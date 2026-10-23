"""응용 모의10회 실행 진입점. 분석별 상태와 결과 표시를 별도 모듈에 위임합니다."""

from pathlib import Path
import json
import sys
import pandas as pd

REPO = next(p for p in Path(__file__).parents if (p / "src/Recipes/loader.py").exists())
sys.path.insert(0, str(REPO))
from src.output import _json_value
from regression_tasks import RegressionTasks
from classification_tasks import ClassificationTasks
from time_series_tasks import TimeSeriesTasks
from presentation import present_result


class Context(RegressionTasks, ClassificationTasks, TimeSeriesTasks):
    """한 회차의 데이터·설정·순차 실행 상태를 보존합니다."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.plan = json.loads((self.folder / "실행설정.json").read_text())
        self.seed = self.plan["seed"]
        self.number = self.plan["number"]
        self.results = []
        self.out = self.folder / "실행결과"
        self.out.mkdir(exist_ok=True)
        self.reg = pd.read_csv(self.folder / "data/regression.csv")
        self.cls = pd.read_csv(self.folder / "data/classification.csv")
        self.ts = (
            pd.read_csv(self.folder / "data/timeseries.csv", parse_dates=["date"])
            .set_index("date")
            .target.asfreq("D")
        )

    def run(self, qid, task):
        """문항을 순서대로 실행하고 전체 수치 저장과 읽기용 표를 분리합니다."""
        groups = [
            [
                "reg_design",
                "reg_baseline",
                "reg_diagnostics",
                "reg_selection",
                "reg_tuning",
                "reg_final",
            ],
            ["cls_design", "cls_tuning", "cls_logit", "cls_final", "cls_decision"],
            ["ts_design", "ts_tuning", "ts_diagnostics", "ts_final"],
        ]
        sequence = next((g for g in groups if task in g), None)
        if sequence is None:
            raise ValueError("알 수 없는 문항 실행 단계: " + str(task))
        completed = {r["task"] for r in self.results}
        missing = [t for t in sequence[: sequence.index(task)] if t not in completed]
        if missing:
            raise ValueError("이 문항 전에 실행할 단계: " + ", ".join(missing))
        value = _json_value(getattr(self, task)())
        self.results.append(dict(question=qid, task=task, result=value))
        (self.out / "답안수치.json").write_text(
            json.dumps(self.results, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
        present_result(task, value)
        return value
