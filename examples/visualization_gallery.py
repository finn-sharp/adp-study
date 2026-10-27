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
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _font_name():
    """저장소에 포함된 한글 폰트를 등록하며 네트워크를 사용하지 않습니다."""
    from matplotlib import font_manager

    path = ROOT / "기출문제/공통/NanumGothic-Regular.ttf"
    font_manager.fontManager.addfont(str(path))
    return font_manager.FontProperties(fname=str(path)).get_name()


@contextmanager
def _style():
    """호출자 전역 스타일과 GUI backend를 변경하지 않는 저장 스타일."""
    import matplotlib

    with matplotlib.rc_context(
        {
            "font.family": _font_name(),
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
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
            figure.suptitle(
                path + ((" / " + variant) if variant else ""),
                fontsize=14,
                fontweight="bold",
                y=0.98,
            )
            figure.text(
                0.06,
                0.035,
                textwrap.fill("예시 자료: " + case["note"], width=65),
                fontsize=9,
                ha="left",
                va="bottom",
            )
            bottom = (
                0.25
                if any(label.get_rotation() for label in ax.get_xticklabels())
                else 0.19
            )
            figure.subplots_adjust(
                left=0.12,
                right=0.82 if cax is not None else 0.96,
                bottom=bottom,
                top=0.86,
            )
            if cax is not None:
                cax.set_position([0.87, bottom, 0.025, 0.86 - bottom])
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
        figure = _figure(6.2 * columns, 4.8 * rows + 1.0)
        axes = figure.subplots(rows, columns, squeeze=False)
        try:
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
                ax.set_title(path.split(".")[-1], fontsize=12)
                ax.tick_params(labelsize=8)
            for ax in list(axes.flat)[len(paths) :]:
                ax.set_axis_off()
            figure.suptitle(
                group + " / subplot examples", fontsize=17, fontweight="bold", y=0.99
            )
            figure.text(
                0.04,
                0.015,
                "고정 합성 예시 자료 · 각 PNG의 자료 설명과 해석 조건을 함께 확인하세요.",
                fontsize=11,
            )
            figure.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=3.0, w_pad=2.0)
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
        "기본 출력은 `docs/images/visualization/`입니다. 단일 모듈은 자기 PNG만 갱신하므로 출력 환경·예제 입력을 변경한 경우 전체 생성 명령으로 manifest와 안내도 갱신하세요. 한글은 저장소의 NanumGothic 폰트를 사용합니다.",
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
