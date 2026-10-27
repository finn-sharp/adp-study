"""
================================================================================
sna.py - SNA 사회연결망 분석
================================================================================
설명:
    - SNA 사회연결망 분석을 Profile·전용 Output DTO·공개 실행기로 제공합니다.
    - 실행 설정은 src/Recipes의 모델별 JSON에서 별도로 관리합니다.
분석 가정:
    - 가중치는 연결 강도이며 거리 기반 중심성에는 역수를 사용합니다. SNA는 신경망 모델이 아닙니다.
    - 결측값과 범주형 인코딩은 실행 전에 처리합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-11
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-11  김재현(Finn)     v1.0.0    모델별 레시피·전체 명세·실행 예제 작성
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.output import OutputDTO
from src.Recipes import load_profile, resolve_profile
from src.MachineLearning.common import validate_features, make_scaler
from src.MachineLearning.analysis_common import (
    positive_int,
    finite_number,
    validate_scaling,
    feature_names,
)


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class SNAProfile:
    """SNA 사회연결망 분석 입력 조건.

    Attributes:
        directed (bool): 유향 그래프 여부. False이면 무향. 기본 False.
        nodes (Optional[List[str]]): 고립 노드를 포함할 선택적 문자열 노드 목록. None이면 연결에서 추출. 기본 None.
        pagerank_alpha (float): PageRank 감쇠 계수. 0 초과 1 미만. 기본 0.85.
        pagerank_max_iter (int): PageRank 반복 한도. 양의 정수. 기본 1000.
        pagerank_tol (float): PageRank 수렴 허용오차. 양수. 기본 1e-6.
    """

    directed: bool = False
    nodes: Optional[List[str]] = None
    pagerank_alpha: float = 0.85
    pagerank_max_iter: int = 1000
    pagerank_tol: float = 1e-6


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class SNAOutputDTO(OutputDTO):
    """SNA 사회연결망 분석 결과. 배열은 원 입력 행 순서.

    Attributes:
        model (Any): 구성된 networkx Graph/DiGraph. 기본 JSON/사전에서 제외. 예측 모델이 아님.
        nodes (List[str]): 노드 결과 행 순서. 지정 노드 후 연결의 최초 등장 순서.
        node_metrics (pd.DataFrame): 노드별 degree, strength, degree_centrality, betweenness, closeness, pagerank, community.
        communities (List[List[str]]): 무향 투영 그래프의 greedy modularity 커뮤니티. 고립점은 단독 커뮤니티.
        density (float): 노드 대비 실제 연결 밀도.
        n_components (int): 무향 연결 요소 또는 유향 weakly-connected 요소 수.
        modularity (float): 무향 투영 그래프의 strength 가중 모듈성. 연결이 없으면 0.
    """

    model: Any
    nodes: List[str]
    node_metrics: pd.DataFrame
    communities: List[List[str]]
    density: float
    n_components: int
    modularity: float


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _graph(edges: Any, profile: SNAProfile) -> Any:
    """연결 목록을 단순 그래프로 구성하고 중복 연결 강도를 합산합니다.

    Args:
        edges: (source,target) 또는 (source,target,strength) 문자열 노드 연결 목록.
        profile: 고립 노드와 방향 조건.
    Returns:
        Any: networkx Graph/DiGraph. 노드 최소 2개.
    Raises:
        ValueError: 노드 타입·자기 연결·가중치·입력 오류.
        ImportError: networkx 미설치.
    """
    import networkx as nx

    if not isinstance(edges, (list, tuple)):
        raise ValueError("edges는 연결 목록이어야 합니다.")
    G = nx.DiGraph() if profile.directed else nx.Graph()
    if profile.nodes is not None:
        if not isinstance(profile.nodes, (list, tuple)) or any(
            not isinstance(n, str) or not n.strip() for n in profile.nodes
        ):
            raise ValueError("nodes는 비어 있지 않은 문자열 목록이어야 합니다.")
        if len(set(profile.nodes)) != len(profile.nodes):
            raise ValueError("nodes에 중복이 있습니다.")
        G.add_nodes_from(profile.nodes)
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) not in (2, 3):
            raise ValueError("연결은 2개 노드와 선택적 강도로 구성합니다.")
        left, right = edge[:2]
        if any(not isinstance(n, str) or not n.strip() for n in (left, right)):
            raise ValueError("노드는 비어 있지 않은 문자열입니다.")
        if left == right:
            raise ValueError("자기 연결은 지원하지 않습니다.")
        weight = edge[2] if len(edge) == 3 else 1.0
        finite_number(weight, "strength", positive=True)
        weight += G.get_edge_data(left, right, {}).get("weight", 0)
        if not np.isfinite(weight):
            raise ValueError("합산 연결 강도는 유한해야 합니다.")
        distance = 1.0 / weight
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("연결 강도의 역수는 유한한 양수여야 합니다.")
        G.add_edge(left, right, weight=weight, distance=distance)
    if len(G) < 2:
        raise ValueError("그래프에 최소 두 노드가 필요합니다.")
    return G


def _validate_inputs(edges: Any, profile: SNAProfile) -> Any:
    """자료와 Profile 조건을 검증합니다.

    Args:
        edges: 문자열 노드 연결 목록.
        profile: SNAProfile 조건.
    Returns:
        Any: 검증된 networkx 그래프.
    Raises:
        TypeError: 잘못된 Profile 타입.
        ValueError: 입력·설정 범위 또는 자료 조건 오류.
    """
    if not isinstance(profile, SNAProfile):
        raise TypeError("SNAProfile이 필요합니다.")
    if not isinstance(profile.directed, bool):
        raise ValueError("directed는 bool이어야 합니다.")
    finite_number(profile.pagerank_alpha, "pagerank_alpha", positive=True)
    finite_number(profile.pagerank_tol, "pagerank_tol", positive=True)
    positive_int(profile.pagerank_max_iter, "pagerank_max_iter")
    if profile.pagerank_alpha >= 1:
        raise ValueError("pagerank_alpha는 1 미만이어야 합니다.")
    arr = _graph(edges, profile)
    return arr


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _run_model(edges: Any, profile: SNAProfile) -> SNAOutputDTO:
    """검증된 입력을 적합하고 알고리즘별 결과를 구성합니다.

    Args:
        edges: 문자열 노드 연결 목록.
        profile: 검증된 SNAProfile.
    Returns:
        SNAOutputDTO: 실행 객체와 원 행 순서의 분석 결과.
    Raises:
        ValueError: 적합 실패 또는 알고리즘 자료 조건 불충족.
        ImportError: 선택적 라이브러리 미설치.
    """
    import networkx as nx

    G = _graph(edges, profile)
    nodes = list(G.nodes())
    projection = nx.Graph()
    projection.add_nodes_from(nodes)
    for left, right, data in G.edges(data=True):
        current = projection.get_edge_data(left, right, {}).get("weight", 0)
        projection.add_edge(left, right, weight=current + data["weight"])
    communities = (
        list(
            nx.algorithms.community.greedy_modularity_communities(
                projection, weight="weight"
            )
        )
        if projection.number_of_edges()
        else [frozenset([n]) for n in nodes]
    )
    communities = sorted([sorted(c) for c in communities], key=lambda c: c[0])
    membership = {
        node: i for i, community in enumerate(communities) for node in community
    }
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)
    closeness = nx.closeness_centrality(G, distance="distance")
    pagerank = nx.pagerank(
        G,
        alpha=profile.pagerank_alpha,
        max_iter=profile.pagerank_max_iter,
        tol=profile.pagerank_tol,
        weight="weight",
    )
    metrics = pd.DataFrame(
        [
            dict(
                node=n,
                degree=G.degree(n),
                strength=G.degree(n, weight="weight"),
                degree_centrality=degree[n],
                betweenness=betweenness[n],
                closeness=closeness[n],
                pagerank=pagerank[n],
                community=membership[n],
            )
            for n in nodes
        ]
    ).set_index("node")
    components = (
        nx.number_weakly_connected_components(G)
        if profile.directed
        else nx.number_connected_components(G)
    )
    modularity = (
        nx.algorithms.community.modularity(
            projection, [set(c) for c in communities], weight="weight"
        )
        if projection.number_of_edges()
        else 0.0
    )
    return SNAOutputDTO(
        G,
        nodes,
        metrics,
        communities,
        float(nx.density(G)),
        components,
        float(modularity),
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_sna(
    edges: Any,
    profile: Optional[SNAProfile] = None,
    *,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> SNAOutputDTO:
    """분석 조건을 검증하고 실행합니다.

    Args:
        edges: 문자열 노드의 (source,target[,strength]) 연결 목록. 강도는 양수.
        profile: 명시적 SNAProfile. None이면 sna.json을 읽음.
        recipe_path: Profile 생략 시 사용할 사용자 평면 JSON 경로.
        overrides: Profile 생략 시 레시피 위에 적용할 일회성 설정.
    Returns:
        SNAOutputDTO: 분석 결과. to_dict()/to_json()은 실행 객체를 제외.
    Raises:
        TypeError: 잘못된 Profile 또는 overrides 타입.
        ValueError: 입력·설정·적합 오류 또는 Profile과 레시피 옵션 동시 지정.
        FileNotFoundError/OSError: 레시피 읽기 실패.
        ImportError: 선택적 라이브러리 미설치.
    """
    profile = resolve_profile(
        "sna", profile, recipe_path=recipe_path, overrides=overrides
    )
    _validate_inputs(edges, profile)
    return _run_model(edges, profile)


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    edges = [("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")]
    result = run_sna(edges, load_profile("sna"))
    print(result.node_metrics)
    print(result.to_json(indent=2)[:500])
