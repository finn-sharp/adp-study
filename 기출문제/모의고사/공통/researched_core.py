"""
================================================================================
researched_core.py - 모의16~20회 실행·원행·수치 기록. 계산은 분야별 파일에서 조합합니다.
================================================================================
설명:
    - 모의16~20회 실행·원행·수치 기록. 계산은 분야별 파일에서 조합합니다.
분석 가정:
    - 제공 자료·문항 명세·실행 환경을 따릅니다. 복원·창작 연습 자료의 수치를 공식 기출 정답으로 해석하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

REPO = next(p for p in Path(__file__).parents if (p / "src/Recipes/loader.py").exists())
sys.path.insert(0, str(REPO))
from src.output import _json_value
from src.Evaluation.regression.metrics import run_regression_metrics


def regression_metrics(y, p):
    """문항의 예측 목적에 맞는 원단위 세 지표만 보고합니다."""
    values = run_regression_metrics(y, p).value
    return {name: values[name] for name in ("rmse", "mae", "r2")}


def finish_plot():
    plt.tight_layout()
    plt.show()
    plt.close("all")


class ResearchContext:
    """각 대문항의 상태를 따로 보존하고 실행 순서 위반을 거부합니다."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.spec = json.loads((self.folder / "문항명세.json").read_text())
        plan = json.loads((self.folder / "실행설정.json").read_text())
        expected = dict(
            seed=self.spec["seed"],
            sections=[
                dict(family=s["family"], settings=s["settings"], context=s["context"])
                for s in self.spec["sections"]
            ],
        )
        if plan != expected:
            raise ValueError(
                "지문과 실행설정이 다릅니다. 생성 명세를 수정한 뒤 재생성하십시오."
            )
        self.seed = plan["seed"]
        self.states = {}
        self.results = []
        self.out = self.folder / "실행결과"
        self.out.mkdir(exist_ok=True)

    def run(self, qid, task):
        section_id, step = map(int, qid.split("-"))
        if not 1 <= section_id <= len(self.spec["sections"]):
            raise ValueError("명세에 없는 대문항입니다.")
        sec = self.spec["sections"][section_id - 1]
        if not 1 <= step <= len(sec["questions"]):
            raise ValueError("명세에 없는 소문항입니다.")
        expected = sec["questions"][step - 1]
        if expected["task"] != task:
            raise ValueError("문항과 실행 단계가 다릅니다.")
        if step > 1 and not any(
            r["question"] == "%d-%d" % (section_id, step - 1) for r in self.results
        ):
            raise ValueError("앞 문항부터 실행하십시오.")
        if any(r["question"] == qid for r in self.results):
            raise ValueError("재실행은 새 Context에서 시작하십시오.")
        if sec["family"] in ["reg", "cls"]:
            from researched_prediction import execute
        elif sec["family"] in ["ts", "risk", "lag", "energy"]:
            from researched_temporal import execute
        else:
            from researched_descriptive import execute
        result = _json_value(
            execute(self, sec, step, self.states.setdefault(section_id, {}))
        )
        self.results.append(dict(question=qid, task=task, result=result))
        (self.out / "답안수치.json").write_text(
            json.dumps(self.results, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
        # Large arrays remain in JSON; notebooks foreground metrics and decisions.
        for key, value in result.items():
            if key in (
                "truth",
                "predictions",
                "probabilities",
                "train_indices",
                "test_indices",
                "folds",
                "daily_rows",
                "interval_lower",
                "interval_upper",
                "rows",
                "tokens",
                "scores",
            ):
                continue
            if isinstance(value, list) and value and isinstance(value[0], dict):
                display(pd.DataFrame(value).head(16))
            elif isinstance(value, dict) and {"data", "columns", "index"} <= set(value):
                display(
                    pd.DataFrame(
                        value["data"], columns=value["columns"], index=value["index"]
                    ).head(16)
                )
            elif isinstance(value, dict):
                display(
                    pd.DataFrame([dict(metric=k, value=v) for k, v in value.items()])
                )
            else:
                print(key + ":", value)
        return result
