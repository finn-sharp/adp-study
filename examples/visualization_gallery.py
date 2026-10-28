"""
================================================================================
visualization_gallery.py - 각 모듈 __main__에서 PNG를 저장하고 전체 시각화 갤러리를 재생성합니다.
================================================================================
설명:
    - 각 모듈 __main__에서 PNG를 저장하고 전체 시각화 갤러리를 재생성합니다.
분석 가정:
    - 입력의 형상·타입·자료 순서를 호출자가 확인합니다. 계산과 파일 쓰기의 조건은 각 공개 함수의 계약을 따릅니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

import argparse
import ast
from contextlib import contextmanager
from hashlib import sha256
import importlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/images/visualization"


def registered_plots():
    """등록부의 모든 시각화 모듈을 중복 없이 반환합니다.

    Args:
        None: 현재 등록부 사용.
    Returns:
        dict: 상대 모듈 경로→레시피 키.
    Raises:
        ValueError: 한 모듈의 중복 등록.
    """
    from src.Recipes.loader import _REGISTRY

    plots = {}
    for key, (module, _, _, _) in _REGISTRY.items():
        if module.startswith("src.Visualization."):
            path = module[len("src.Visualization.") :]
            if path in plots:
                raise ValueError("중복 시각화 모듈: " + path)
            plots[path] = key
    return dict(sorted(plots.items()))


@contextmanager
def _style():
    """예제와 외부 subplot에 재사용할 수 있는 보고서 테마를 적용합니다."""
    from src.Visualization.theme import analysis_theme

    with analysis_theme():
        yield


def _figure(width=9, height=6):
    """pyplot 전역 Figure 등록 없이 Agg 캔버스를 가진 소유 Figure를 생성합니다."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    return figure


def _description(path):
    """모듈 명세의 실제 기능 제목을 사용합니다."""
    source = ROOT / ("src/Visualization/" + path.replace(".", "/") + ".py")
    doc = ast.get_docstring(ast.parse(source.read_text())) or ""
    return next(
        line.split(".py -", 1)[1].strip()
        for line in doc.splitlines()
        if ".py -" in line
    )


_GROUPS = {
    "numeric": ("수치 분포", "중심·산포·분포의 형태를 함께 살펴봅니다."),
    "categorical": ("범주별 비교", "집단의 크기와 분포, 발생률을 비교합니다."),
    "classification": ("분류 성능", "식별력·확률·임계값을 같은 자료에서 확인합니다."),
    "clustering": ("군집 구조", "집단의 분리도와 군집 수 후보를 검토합니다."),
    "decomposition": ("차원과 성분", "설명분산과 변수별 성분 계수를 비교합니다."),
    "quality": ("데이터 품질", "누락의 크기와 공동 결측 패턴을 확인합니다."),
    "regression": ("회귀모형 진단", "잔차·계수·영향 관측을 함께 확인합니다."),
    "relationships": ("변수 간 관계", "관측의 밀도와 변수 사이의 관계를 살펴봅니다."),
    "selection": ("변수와 학습량", "중요도와 학습량별 점수를 비교합니다."),
    "time_series": ("시간 패턴", "추세와 반복 구조, 지연 관계를 확인합니다."),
    "qq": ("분포별 Q–Q", "관측 분위수와 이론 분위수를 비교합니다."),
    "association": ("연관규칙", "빈도·신뢰도·향상도를 함께 살펴봅니다."),
    "networks": ("연결 구조", "연결된 집단과 고립된 관측을 확인합니다."),
    "optimization": ("최적화 과정", "세대별 실행가능 최우수 목적값을 살펴봅니다."),
}
_TITLES = {
    "histogram": "값의 분포",
    "kde": "분포 밀도",
    "ecdf": "누적 분포",
    "qq": "정규분포 Q–Q",
    "count": "범주별 관측 수",
    "pareto": "범주별 비중과 누적 비중",
    "grouped_distribution": "집단별 분포",
    "contingency": "범주 간 구성비",
    "mosaic": "범주별 구성과 규모",
    "category_rate": "범주별 양성 비율",
    "missingness": "변수별 결측률",
    "missing_matrix": "관측별 결측 패턴",
    "missing_association": "공동 결측의 연관",
    "scatter": "변수 간 관계",
    "hexbin": "관측 밀도",
    "correlation": "상관계수",
    "dendrogram": "계층적 군집 구조",
    "silhouette": "군집 분리도",
    "k_distance": "이웃 거리",
    "elbow": "군집 수별 개선 폭",
    "explained_variance": "성분별 설명분산",
    "loadings": "변수별 성분 계수",
    "confusion": "실제와 예측의 구성비",
    "roc": "ROC 곡선",
    "precision_recall": "정밀도·재현율",
    "threshold": "임계값별 성능",
    "calibration": "예측확률과 실제 발생률",
    "rules": "빈도·신뢰도·향상도",
    "graph": "관측 간 연결",
    "feature_importance": "변수 중요도",
    "validation_curve": "학습량별 점수",
    "timeline": "시간에 따른 변화",
    "lag": "지연 관계",
    "autocorrelation": "자기상관",
    "seasonal": "요일별 분포",
    "coefficient_intervals": "회귀계수와 구간",
    "cooks_distance": "관측별 영향력",
    "leverage": "레버리지와 잔차",
    "model_comparison": "같은 검증 구간의 모형 비교",
    "observed_predicted": "관측값과 적합값",
    "partial_residual": "변수의 부분 잔차",
    "prediction_interval": "평균 구간과 예측구간",
    "residual_fitted": "적합값과 잔차",
    "residual_order": "관측 순서별 잔차",
    "scale_location": "잔차의 크기",
    "vif": "변수 간 공선성",
    "glm_residuals": "GLM 잔차",
    "convergence": "최우수 목적값의 변화",
}


def _title(path, variant=None):
    group, name = path.split(".")
    title = (
        (name.replace("_", " ").title() + " Q–Q")
        if group == "qq"
        else _TITLES.get(name, name.replace("_", " ").title())
    )
    if variant:
        title += " · " + variant.replace("_", " / ")
    return title


def _value(value):
    if isinstance(value, int):
        return format(value, ",")
    if isinstance(value, float):
        return format(value, ".3g")
    return str(value)


def _metrics(group, entries):
    """실제 DTO의 진단값만 요약합니다. 없으면 지표를 만들어 넣지 않습니다."""
    choices = {
        "numeric": [
            ("histogram", "n", "관측 수"),
            ("histogram", "mean", "평균"),
            ("histogram", "median", "중앙값"),
        ],
        "categorical": [
            ("count", "n", "관측 수"),
            ("count", "n_categories", "범주 수"),
        ],
        "classification": [
            ("roc", "roc_auc", "ROC AUC"),
            ("precision_recall", "average_precision", "Average precision"),
            ("calibration", "brier_score", "Brier score"),
        ],
        "clustering": [
            ("silhouette", "n", "관측 수"),
            ("silhouette", "mean_silhouette", "평균 silhouette"),
        ],
        "decomposition": [
            ("loadings", "n_features", "변수 수"),
            ("loadings", "n_components", "성분 수"),
        ],
        "quality": [
            ("missingness", "n_rows", "관측 수"),
            ("missingness", "complete_rows", "완전 관측"),
            ("missing_matrix", "missing_cells", "결측 셀"),
        ],
        "regression": [
            ("observed_predicted", "r2", "학습 R²"),
            ("cooks_distance", "flagged", "영향 관측 후보"),
            ("observed_predicted", "n", "관측 수"),
        ],
        "relationships": [
            ("scatter", "n", "관측 수"),
            ("correlation", "n_features", "변수 수"),
        ],
        "networks": [
            ("graph", "n_nodes", "노드 수"),
            ("graph", "n_edges", "연결 수"),
            ("graph", "isolates", "고립 노드"),
        ],
        "selection": [("feature_importance", "n_features", "변수 수")],
        "time_series": [
            ("timeline", "n", "관측 수"),
            ("timeline", "rolling_window_in_rows", "후행 요약 구간"),
        ],
        "association": [
            ("rules", "input_rules", "입력 규칙"),
            ("rules", "displayed_rules", "표시 규칙"),
        ],
        "optimization": [("convergence", "best_objective", "최우수 목적값")],
        "qq": [("normal", "n", "분포별 관측 수")],
    }
    result = []
    for name, key, label in choices.get(group, []):
        entry = entries.get(group + "." + name)
        if entry is not None:
            value = entry.get("diagnostics", {}).get(key)
            if value is not None:
                result.append((label, _value(value)))
    return result[:3]


def _rule(figure, y, left=0.055, right=0.95):
    from matplotlib.lines import Line2D

    figure.add_artist(
        Line2D(
            [left, right],
            [y, y],
            transform=figure.transFigure,
            color="#E3E9ED",
            linewidth=0.7,
        )
    )


def _draw(path, plotter, case, output, variant=None):
    """한 예제를 기존 plot 함수에 전달하고 PNG와 진단 메타데이터를 만듭니다."""
    with _style():
        figure = _figure()
        ax = figure.subplots()
        try:
            cax = (
                figure.add_axes((0.87, 0.19, 0.025, 0.67))
                if case.get("colorbar_label")
                else None
            )
            kwargs = {"cax": cax} if cax is not None else {}
            result = plotter(
                *case["args"], ax=ax, overrides=case["overrides"], **kwargs
            )
            if cax is not None:
                cax.set_ylabel(case["colorbar_label"], fontsize=10)
            if result.axes is not ax or len(figure.axes) != (
                2 if cax is not None else 1
            ):
                raise AssertionError("예제는 전달한 단일 ax를 사용해야 합니다.")
            from src.Visualization.theme import style_axes

            style_axes(ax, kind=path)
            ax.set_title("")
            group = path.split(".")[0]
            figure.text(
                0.075,
                0.955,
                _GROUPS[group][0],
                fontsize=9,
                color="#2E746E",
                weight="bold",
            )
            figure.text(
                0.075,
                0.902,
                _title(path, variant),
                fontsize=17,
                color="#293D46",
                weight="bold",
            )
            n = result.diagnostics.get(
                "n",
                result.diagnostics.get("n_samples", result.diagnostics.get("n_rows")),
            )
            if n is not None:
                figure.text(
                    0.935, 0.945, "관측 수", fontsize=8, color="#788891", ha="right"
                )
                figure.text(
                    0.935, 0.901, _value(n), fontsize=16, color="#293D46", ha="right"
                )
            _rule(figure, 0.845, left=0.075, right=0.935)
            _rule(figure, 0.13, left=0.075, right=0.935)
            figure.text(
                0.075,
                0.052,
                textwrap.fill(case["note"], width=75),
                fontsize=8,
                color="#788891",
                ha="left",
                va="bottom",
            )
            bottom = (
                0.27
                if any(label.get_rotation() for label in ax.get_xticklabels())
                else 0.22
            )
            figure.subplots_adjust(
                left=0.115,
                right=0.82 if cax is not None else 0.935,
                bottom=bottom,
                top=0.79,
            )
            if cax is not None:
                cax.set_position([0.865, bottom, 0.016, 0.79 - bottom])
                for spine in cax.spines.values():
                    spine.set_visible(False)
                cax.tick_params(length=0, labelsize=8, colors="#788891")
                cax.yaxis.label.set_color("#63757F")
            group, name = path.split(".")
            filename = name + (("__" + variant) if variant else "") + ".png"
            destination = output / group / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(
                str(destination),
                dpi=150,
                metadata={"Software": "adp-study visualization examples"},
            )
            payload = result.to_dict(json_safe=True)
            return dict(
                image=destination.relative_to(output).as_posix(),
                width=1350,
                height=900,
                sha256=sha256(destination.read_bytes()).hexdigest(),
                parameters=payload["parameters"],
                diagnostics=payload["diagnostics"],
                warnings=payload["warnings"],
                note=case["note"],
            )
        finally:
            figure.clear()


def save_example(path, plotter, output_dir=None):
    """각 모듈 __main__에서 자기 plot 함수를 넘겨 PNG를 생성합니다.

    Args:
        path: numeric.histogram 같은 등록 시각화 상대 경로.
        plotter: 그 모듈에 정의된 실제 plot 함수.
        output_dir: 출력 폴더. None이면 __main__ CLI의 --output-dir 또는 기본 docs/images/visualization.
    Returns:
        dict: 기본 PNG와 선택 변형의 파일·진단·설정 메타데이터.
    Raises:
        ValueError/TypeError/OSError: 사례 누락·그림·파일 저장 오류.
    """
    if output_dir is None:
        parser = argparse.ArgumentParser(
            description="이 시각화 모듈의 고정 합성 예제를 PNG로 저장합니다."
        )
        parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
        output_dir = parser.parse_args().output_dir
    output = Path(output_dir)
    from examples.visualization_gallery_inputs import example_cases

    cases = example_cases(path.split(".")[0])
    if path not in cases:
        raise ValueError("시각화 예제 누락: " + path)
    entry = _draw(path, plotter, cases[path], output)
    entry["variants"] = {
        name: _draw(path, plotter, case, output, name)
        for name, case in cases[path].get("variants", {}).items()
    }
    print(str(output / entry["image"]))
    return entry


def _dashboard(group, paths, cases, output):
    """호출자가 소유한 subplots에 영역별 시각화를 조합해 저장합니다."""
    columns = 2 if len(paths) == 4 else min(3, len(paths))
    rows = (len(paths) + columns - 1) // columns
    with _style():
        height = 4.25 * rows + 1.65
        figure = _figure(5.8 * columns, height)
        axes = figure.subplots(rows, columns, squeeze=False)
        try:
            entries = {}
            from src.Visualization.theme import style_axes

            for path, ax in zip(paths, axes.flat):
                module = importlib.import_module("src.Visualization." + path)
                cax = (
                    ax.inset_axes([1.02, 0, 0.025, 1])
                    if cases[path].get("colorbar_label")
                    else None
                )
                kwargs = {"cax": cax} if cax is not None else {}
                result = getattr(module, "plot_" + path.split(".")[-1])(
                    *cases[path]["args"],
                    ax=ax,
                    overrides=cases[path]["overrides"],
                    **kwargs,
                )
                if cax is not None:
                    cax.set_ylabel(cases[path]["colorbar_label"], fontsize=8)
                    cax.tick_params(labelsize=7)
                if result.axes is not ax:
                    raise AssertionError("subplot 소유권 오류")
                style_axes(ax, kind=path)
                ax.set_title(_title(path), fontsize=11, loc="left", pad=14)
                entries[path] = {
                    "diagnostics": result.to_dict(json_safe=True)["diagnostics"]
                }
                if cax is not None:
                    for spine in cax.spines.values():
                        spine.set_visible(False)
                    cax.tick_params(length=0, labelsize=7, colors="#788891")
            for ax in list(axes.flat)[len(paths) :]:
                ax.set_axis_off()
            title, subtitle = _GROUPS[group]
            figure.text(
                0.055,
                1 - 0.38 / height,
                title,
                fontsize=18,
                weight="bold",
                color="#293D46",
            )
            figure.text(0.055, 1 - 0.77 / height, subtitle, fontsize=9, color="#788891")
            metrics = _metrics(group, entries)
            for i, (label, value) in enumerate(metrics):
                x = 0.65 + 0.12 * i
                figure.text(
                    x,
                    1 - 0.28 / height,
                    label,
                    fontsize=7.5,
                    color="#788891",
                    ha="center",
                )
                figure.text(
                    x,
                    1 - 0.66 / height,
                    value,
                    fontsize=16,
                    weight="bold",
                    color="#293D46",
                    ha="center",
                )
            _rule(figure, 1 - 1.08 / height)
            _rule(figure, 0.48 / height)
            figure.text(
                0.055,
                0.18 / height,
                "고정 합성 자료 예시 · 자료 설명과 해석 조건은 개별 이미지 및 갤러리에서 확인하세요.",
                fontsize=8,
                color="#788891",
            )
            figure.tight_layout(
                rect=(0.025, 0.67 / height, 0.985, 1 - 1.26 / height),
                h_pad=3.0,
                w_pad=2.8,
            )
            destination = output / "dashboards" / (group + ".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(
                str(destination),
                dpi=120,
                metadata={"Software": "adp-study visualization examples"},
            )
            return dict(
                image=destination.relative_to(output).as_posix(),
                modules=paths,
                sha256=sha256(destination.read_bytes()).hexdigest(),
            )
        finally:
            figure.clear()


def _markdown(manifest):
    """갤러리·단일 이미지·소스·직접 실행 명령을 연결합니다."""
    lines = [
        "# 전체 시각화 PNG 예시",
        "",
        "등록된 시각화 **"
        + str(len(manifest["modules"]))
        + "개 전체**의 기본 PNG와 분포별 GLM 변형5개, 영역별 subplot 조합14개를 제공합니다. 모든 자료는 고정 합성 자료이며 실제 업무 자료나 모형 우수성의 근거가 아닙니다.",
        "",
        '각 모듈의 `if __name__ == "__main__":` 아래에서 자기 plot 함수로 PNG를 저장합니다. import와 일반 plot 호출에는 파일 저장 부작용이 없습니다. 색상으로 수치를 표시하는 예시는 호출자가 별도 cax를 만들어 색상 범례를 연결합니다. 그림 생성·저장은 예제 계층의 책임이며 plot 함수는 전달한 ax를 재사용합니다.',
        "",
        "```bash",
        "python -m src.Visualization.numeric.histogram",
        "python -m src.Visualization.regression.glm_residuals --output-dir /tmp/adp-plots",
        "python -m examples.visualization_gallery  # 전체 PNG·manifest·이 안내 재생성",
        "python -m examples.visualization_gallery --check  # 파일·해시·등록부 누락 검사",
        "```",
        "",
        "기본 출력은 `docs/images/visualization/`입니다. 단일 모듈은 자기 PNG만 갱신하므로 출력 환경·예제 입력을 변경한 경우 전체 생성 명령으로 manifest와 안내도 갱신하세요. 흰 바탕·얇은 격자·절제된 팔레트의 보고서 테마를 사용하며, 한글은 저장소의 NanumGothic 폰트로 표시합니다. [공통 테마 사용법](VISUALIZATION_THEME.md)을 참고하세요.",
        "",
        "[EDA 명세](EDA_VISUALIZATION_MODULES.md) · [회귀·분포 명세](REGRESSION_DISTRIBUTION_MODULES.md) · [이미지·설정·진단 manifest](images/visualization/manifest.json)",
        "",
        "모수·조건·원 자료의 역할은 각 이미지 아래 자료 설명과 manifest의 note/parameters/diagnostics/warnings로 확인할 수 있습니다. 중요도·학습곡선·inertia·분산비·규칙 지표는 설명용 사전 계산 예시입니다. 회귀 비교는 같은 합성 자료의 동일4-fold 실제 CV, GA는 실제 소규모 탐색 이력입니다. GLM 잔차는 logistic/Poisson/Gamma와 Pearson/deviance를 각각 제공합니다.",
        "",
    ]
    for group, dashboard in manifest["dashboards"].items():
        lines += [
            "## " + group,
            "",
            "![영역별 subplot 예시](images/visualization/" + dashboard["image"] + ")",
            "",
            "| 모듈·기능 | 개별 PNG |",
            "| --- | --- |",
        ]
        for path in dashboard["modules"]:
            entry = manifest["modules"][path]
            source = "../src/Visualization/" + path.replace(".", "/") + ".py"
            images = ["[기본 이미지](images/visualization/" + entry["image"] + ")"]
            images += [
                "[" + name + "](images/visualization/" + info["image"] + ")"
                for name, info in entry["variants"].items()
            ]
            lines += [
                "| [`"
                + path
                + "`]("
                + source
                + ") — "
                + _description(path)
                + " | "
                + " · ".join(images)
                + " |"
            ]
        lines += [""]
    lines += [
        "## 재현과 파일 계약",
        "",
        "seed20261027의 합성 입력으로 네트워크 없이 생성합니다. 개별 이미지는1350×900px이며 PNG/manifest를 저장소에 함께 둡니다. Matplotlib 버전·렌더링 환경에 따라 픽셀이 달라질 수 있어 정확한 바이트 재현은 같은 의존성 환경에서 확인합니다.",
        "",
        "`--check`는 manifest의61개 모듈과 현재 등록부의 완전 대응, PNG의 정상 디코딩·비어 있지 않은 픽셀·크기·해시, 추가/누락 파일을 검사합니다. 새 시각화를 등록하면 예제 입력과 PNG도 추가해야 합니다.",
        "",
    ]
    return "\n".join(lines)


def build_gallery(output_dir=DEFAULT_OUTPUT, markdown_path=None):
    """등록 시각화 전체를 렌더링하고 PNG·manifest·갤러리를 저장합니다.

    Args:
        output_dir: PNG/manifest 출력 폴더.
        markdown_path: 선택 갤러리 Markdown 경로; 기본 출력 폴더일 때 docs에 생성.
    Returns:
        dict: 모든 모듈·변형·subplot 조합의 manifest.
    Raises:
        ValueError/AssertionError/OSError: 사례/등록 누락·축 계약·저장 오류.
    """
    from examples.visualization_gallery_inputs import example_cases

    plots = registered_plots()
    cases = example_cases()
    output = Path(output_dir)
    if set(plots) != set(cases):
        raise ValueError(
            "등록부와 PNG 예제 입력의 누락/추가: " + str(set(plots) ^ set(cases))
        )
    modules = {}
    for path, key in plots.items():
        module = importlib.import_module("src.Visualization." + path)
        modules[path] = save_example(
            path, getattr(module, "plot_" + path.split(".")[-1]), output
        )
        modules[path]["recipe_key"] = key
    groups = {
        group: [p for p in plots if p.split(".")[0] == group]
        for group in sorted({p.split(".")[0] for p in plots})
    }
    dashboards = {g: _dashboard(g, paths, cases, output) for g, paths in groups.items()}
    manifest = dict(
        schema_version=1,
        theme="analysis-report-v2",
        seed=20261027,
        modules=modules,
        dashboards=dashboards,
        module_count=len(modules),
        image_count=len(modules)
        + sum(len(v["variants"]) for v in modules.values())
        + len(dashboards),
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if markdown_path is None and output.resolve() == DEFAULT_OUTPUT.resolve():
        markdown_path = ROOT / "docs/VISUALIZATION_GALLERY.md"
    if markdown_path is not None:
        Path(markdown_path).write_text(_markdown(manifest), encoding="utf-8")
    return manifest


def check_gallery(output_dir=DEFAULT_OUTPUT):
    """저장된 PNG와 등록부·진단 메타데이터·해시의 완전 대응을 검사합니다.

    Args:
        output_dir: 기존 PNG/manifest 폴더.
    Returns:
        dict: 등록 모듈 수와 정상 이미지 수.
    Raises:
        ValueError/OSError: 파일 누락·추가·손상·빈 이미지·해시/크기 불일치.
    """
    from PIL import Image

    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["modules"]) != set(registered_plots()):
        raise ValueError("등록부/갤러리 불일치")
    expected = []
    for item in manifest["modules"].values():
        expected += [item] + list(item["variants"].values())
    expected += list(manifest["dashboards"].values())
    files = set()
    for item in expected:
        path = output / item["image"]
        files.add(path.relative_to(output).as_posix())
        if sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("PNG 해시 불일치: " + str(path))
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or min(image.size) < 400:
                raise ValueError("PNG 형식/크기 오류")
            if "width" in item and image.size != (item["width"], item["height"]):
                raise ValueError("기본 PNG 크기 오류")
            extrema = image.convert("RGB").getextrema()
            if max(hi - lo for lo, hi in extrema) < 30:
                raise ValueError("빈 PNG: " + str(path))
    if files != {p.relative_to(output).as_posix() for p in output.rglob("*.png")}:
        raise ValueError("PNG 누락/추가")
    if manifest["image_count"] != len(files):
        raise ValueError("이미지 집계 불일치")
    return dict(module_count=len(manifest["modules"]), image_count=len(files))


def main():
    """전체 PNG 생성 또는 기존 갤러리 검사를 실행합니다.

    Args:
        CLI: --output-dir, --check.
    Returns:
        None: 집계를 출력합니다.
    Raises:
        ValueError/OSError: 그림 생성 또는 검사 실패.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = (
        check_gallery(args.output_dir) if args.check else build_gallery(args.output_dir)
    )
    print(
        json.dumps(
            {k: result[k] for k in ("module_count", "image_count")}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
