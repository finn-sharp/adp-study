# `src/Visualization/plots.py`

[소스 코드](../../src/Visualization/plots.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
plots.py - 데이터 시각화와 모델 평가 그래프
================================================================================
설명: 데이터시각화 PDF의 기본 차트·히트맵을 Figure/Axes 반환 함수로 구현합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 rc_context 기반 설정과 재사용 차트 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `PlotProfile`

그래프 설정.

Attributes:
    kind: line, scatter, bar, hist, box, heatmap.
    title: 제목.
    xlabel: x축 이름.
    ylabel: y축 이름.
    figsize: 그림 크기 (인치).
    font_family: 설치된 선택적 폰트 이름. None이면 기본 폰트.
    font_size: 글자 크기.
    bins: 히스토그램 구간 수.
    cmap: 히트맵 색상표.

## `PlotOutputDTO`

그래프 결과.

Attributes:
    figure: matplotlib Figure. 저장은 figure.savefig, 해제는 plt.close(figure).
    axes: matplotlib Axes.
    kind, title, xlabel, ylabel: 요청한 차트 유형과 제목·축 라벨.

## `run_plot`

기본 그래프를 생성합니다. show/save를 자동 수행하지 않습니다.

Args:
    x: 단일 분포 데이터 또는 x축 데이터. heatmap에서는 2차원 행렬.
    y: line/scatter/bar의 y값. line에서 None이면 x를 y값으로 사용.
    profile: 그래프 설정.
Returns:
    PlotOutputDTO: Figure와 Axes. 전역 rcParams를 변경하지 않습니다.
Raises:
    ValueError: 미지원 kind, 필요한 y 누락, 설치되지 않은 폰트 또는 잘못된 행렬.

## `PairPlotOutputDTO`

쌍별 시각화 출력 DTO.

Attributes:
    figure: matplotlib Figure(runtime), 저장은 figure.savefig.
    axes: Axes 배열(runtime).
    grid: 원본 seaborn PairGrid(runtime).
    feature_names: 입력 열 이름, 원래 순서.
    hue: 요청한 그룹 열 또는 None.
Notes:
    이전 grid.fig 접근을 fig 속성으로 지원합니다. 원본 PairGrid 메서드는 grid로 접근합니다.

### `fig`



## `run_pairplot`

수치 특성 간 쌍별 분포·산점도를 생성합니다.

Args:
    data: pandas DataFrame.
    hue: 선택적 그룹 열 이름.
Returns:
    PairPlotOutputDTO: Figure/Axes/PairGrid 및 입력 열·hue 정보. 저장/표시는 호출자가 수행합니다.
Raises:
    ValueError: 잘못된 데이터/그룹 설정.
    ImportError: seaborn 미설치.
