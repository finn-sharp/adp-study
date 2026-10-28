# 공통 시각화 테마

흰 배경, 얇은 격자, 청록 중심의 제한된 팔레트를 사용합니다. 제목·핵심 수치·차트·자료 설명을 분리하고, 영역별 subplot에서도 같은 위계와 여백을 유지합니다. [PNG 갤러리](VISUALIZATION_GALLERY.md)에서 전체 적용 예시를 볼 수 있습니다.

## 제공 기능과 사용 범위

`src.Visualization.theme.analysis_theme()`은 Matplotlib 설정을 컨텍스트 안에 적용하고 종료 시 복원합니다. 한글 폰트는 사용 시 등록하며, import만으로 그림이나 파일을 만들지 않습니다.

`style_axes(ax, kind)`는 전달한 2D 직교 축의 격자·테두리·범례와 지원 색상표를 정리하고 같은 축을 반환합니다. 자료, 축 척도, 기준선 위치와 색상 정규화 범위는 유지합니다. 히트맵은 값의 크기와 부호를 나타내는 색상표를 구분하고 주석 대비를 조정합니다. 사용자 지정 색상표는 교체하지 않습니다.

```python
import matplotlib.pyplot as plt
from src.Visualization.theme import analysis_theme, style_axes

with analysis_theme():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist([1, 2, 2, 3, 4], bins=4)
    axes[1].plot([1, 2, 3], [2, 4, 3])
    style_axes(axes[0], kind="numeric.histogram")
    style_axes(axes[1], kind="timeseries.series")
    fig.tight_layout()
```

기존 모듈의 `plot_*(..., ax=axes[i])`도 같은 컨텍스트에서 실행한 뒤 해당 축에 `style_axes`를 적용하면 됩니다. figure 생성·subplot 배치·저장은 호출자가 담당합니다. 기존 Profile·DTO와 계산 API는 유지됩니다.

## 예시 자료의 구성

각 시각화 모듈의 직접 실행 예제와 갤러리 생성기가 공통 테마를 사용합니다. 개별 그림에는 읽기 쉬운 제목과 자료 설명을 배치하고, 영역별 조합에는 DTO에서 얻은 핵심 수치를 표시합니다. 새 지표를 계산하거나 학습 성능을 평가 성능으로 바꾸지 않습니다. 입력·진단·경고는 갤러리 manifest에 함께 보존합니다.
