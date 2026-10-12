"""레이아웃 생성은 예제의 책임이며 개별 plot 모듈은 기존 Axes만 받습니다."""
from dataclasses import dataclass
from typing import Any,Dict
import numpy as np
from matplotlib.axes import Axes
from src.output import OutputDTO

@dataclass
class EDADashboardOutputDTO(OutputDTO):
    """재사용 가능한 subplot과 개별 시각화 근거.

    Attributes:
        figure (Any): 예제 또는 사용자가 소유한 Figure; JSON 제외.
        axes (Any): 연결한 원 Axes 배열; JSON 제외.
        panels (Dict[str, Any]): 이름별 전용 Plot DTO; JSON에서는 수치만 직렬화.
    """
    figure: Any
    axes: Any
    panels: Dict[str,Any]


def render_panels(calls,axes=None,ncols=2):
    """예제 레이아웃을 생성하거나 사용자가 전달한 축을 순서대로 사용합니다.

    Args:
        calls: (pane 이름, ax를 받는 callable) 목록.
        axes: 기존 Axes 배열 또는 None.
        ncols: 예제가 생성할 양의 열 수.
    Returns:
        EDADashboardOutputDTO: 패널과 각 수치 결과.
    Raises:
        ValueError/TypeError: 빈 호출·레이아웃·축/figure 오류.
    """
    if not calls or ncols<1 or len({name for name,_ in calls})!=len(calls):raise ValueError('패널 이름/레이아웃 오류')
    owned=axes is None
    if owned:
        from matplotlib import pyplot as plt
        rows=(len(calls)+ncols-1)//ncols
        figure,axes=plt.subplots(rows,ncols,figsize=(5.4*ncols,4.1*rows),squeeze=False)
    flat=np.asarray(axes,dtype=object).reshape(-1)
    if len(flat)<len(calls) or any(not isinstance(ax,Axes) for ax in flat) or len({id(ax) for ax in flat})!=len(flat) or len({id(ax.figure) for ax in flat})!=1:raise ValueError('같은 figure의 서로 다른 충분한 Axes가 필요합니다.')
    figure=flat[0].figure;panels={name:call(ax) for (name,call),ax in zip(calls,flat)}
    if owned:
        for ax in flat[len(calls):]:ax.set_axis_off()
        figure.tight_layout()
    return EDADashboardOutputDTO(figure,axes,panels)
