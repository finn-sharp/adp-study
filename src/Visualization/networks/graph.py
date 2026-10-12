"""
================================================================================
graph.py - 연결 구조·분리 요소·허브 후보를 검토합니다.
================================================================================
설명:
    - Profile → JSON → 단일 ax 시각화 → 수치 근거·runtime DTO.
분석 가정:
    - 사용자가 전달한 2D Axes에만 그리며 figure 생성·show·save·예측 모델 재학습을 수행하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    Axes 조합 EDA·의사결정 시각화 추가
================================================================================
"""
__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.Recipes import resolve_profile
from src.Visualization._common import PlotOutputDTO, validate_profile, start, finish, vector, frame, paired, levels, heatmap, binary_scores

# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class NetworksGraphProfile:
    """단일 pane 시각화 설정.

    Attributes:
        title (str): 사용자 제목; 빈 문자열이면 기본 목적 제목. 기본 ''.
        color (str): Matplotlib 기본 선/표식 색. 기본 'C0'.
        alpha (float): artist 투명도 [0,1]. 기본 0.8.
        max_nodes (int): 표시 가능한 최대 노드; 자동 부분 그래프 생성 안 함. 기본 100.
        random_state (int): spring 배치 재현성 seed. 기본 1234.
        show_labels (bool): 노드 label 표시. 기본 True.
    """
    title: str = ''
    color: str = 'C0'
    alpha: float = 0.8
    max_nodes: int = 100
    random_state: int = 1234
    show_labels: bool = True

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class NetworksGraphOutputDTO(PlotOutputDTO):
    """그린 축과 의사결정 수치.

    Attributes:
        figure (Any): 원 ax의 Figure; runtime으로 JSON 제외.
        axes (Any): 전달한 동일 Axes; runtime으로 JSON 제외.
        artists (List[Any]): 이 호출에서 추가한 artist; JSON 제외.
        data (pd.DataFrame): 표시/계산한 수치와 원 label 순서.
        diagnostics (Dict[str, Any]): 표본 수·분모·선택 기준·추가 근거.
        parameters (Dict[str, Any]): 적용 Profile 설정.
        warnings (List[str]): 해석 가정·미표시/표본 제한 정보.
    """
    figure: Any
    axes: Any
    artists: List[Any]
    data: pd.DataFrame
    diagnostics: Dict[str, Any]
    parameters: Dict[str, Any]
    warnings: List[str]

# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _validate(profile):
    """전용 Profile과 공통 스타일을 확인합니다.

    Args:
        profile: NetworksGraphProfile.
    Returns:
        None: 유효 설정; 기법별 입력은 그리기 전에 추가 검증.
    Raises:
        ValueError/TypeError: Profile 타입·필드·스타일 오류.
    """
    validate_profile(profile, NetworksGraphProfile)

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _draw(graph, ax, profile, before):
    """전용 수치 계산과 한 Axes 렌더링.

    Args:
        graph: 명시 NetworkX Graph; 비음수 유한 edge weight.
        ax: 검증한 대상 축.
        profile: 전용 설정.
        before: 기존 artist 식별자.
    Returns:
        NetworksGraphOutputDTO: 축과 수치 결과.
    Raises:
        ValueError/TypeError: 기법별 입력·설정 정의역 오류.
    """
    import networkx as nx
    if not isinstance(graph,nx.Graph) or graph.number_of_nodes()<1 or graph.number_of_nodes()>profile.max_nodes or profile.max_nodes<1:raise ValueError('Graph/노드 한도 오류')
    for u,v,attributes in graph.edges(data=True):
        weight=attributes.get('weight',1.)
        if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not np.isfinite(weight) or weight<0:raise ValueError('spring 가중치는 비음수 유한 실수입니다.')
    positions=nx.spring_layout(graph,seed=profile.random_state,weight='weight');degrees=dict(graph.degree());nodes=list(graph.nodes());sizes=[40+25*degrees[node] for node in nodes]
    nx.draw_networkx(graph,pos=positions,ax=ax,node_size=sizes,node_color=profile.color,alpha=profile.alpha,with_labels=profile.show_labels,font_size=8,arrows=graph.is_directed());ax.set_axis_off()
    data=pd.DataFrame({'node':[str(n) for n in nodes],'x':[positions[n][0] for n in nodes],'y':[positions[n][1] for n in nodes],'degree':[degrees[n] for n in nodes]});diagnostics={'n_nodes':len(nodes),'n_edges':graph.number_of_edges(),'directed':graph.is_directed(),'isolates':len(list(nx.isolates(graph)))};notes=['배치 거리와 node 크기는 탐색 표현입니다. degree 크기는 영향력·인과를 뜻하지 않습니다.']
    return finish(ax, before, data, diagnostics, profile, NetworksGraphOutputDTO, 'Networks — Graph', notes)

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def plot_graph(graph, *, ax, profile=None, recipe_path=None, overrides=None) -> NetworksGraphOutputDTO:
    """연결 구조·분리 요소·허브 후보를 검토합니다.

    Args:
        graph: 명시 NetworkX Graph; 비음수 유한 edge weight.
        ax: 필수 기존 2D Axes; 다른 축과 figure를 만들지 않음.
        profile: 전용 Profile 또는 None.
        recipe_path: 사용자 JSON 경로 또는 None.
        overrides: Profile 생략 시 레시피 재정의.
    Returns:
        NetworksGraphOutputDTO: 동일 ax·수치·진단·설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·축·레시피 오류.
    """
    profile=resolve_profile('plot-networks-graph', profile, recipe_path=recipe_path, overrides=overrides)
    _validate(profile)
    before=start(ax)
    return _draw(graph, ax, profile, before)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
# 실행 예제는 src.Visualization.skeleton.networks에 있으며 사용자 subplots의 축에 연결합니다.
# 예: result = plot_graph(..., ax=axes[0, 0]); result.axes is axes[0, 0]
