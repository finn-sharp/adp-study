"""Statistics.pdf 분포 16종×2 예제의 교정된 실행 기록.

연속 점확률은0이고 PDF는 밀도입니다. 이하/이상은 CDF/SF로 계산합니다.
불충분한 모수를 가진 문항에는 추가한 가정을 결과의 assumptions에 명시합니다.
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"
import json
import math
from scipy.special import gamma
from src.stats.distributions.bernoulli import run_bernoulli
from src.stats.distributions.binomial import run_binomial
from src.stats.distributions.normal import run_normal
from src.stats.distributions.hypergeometric import run_hypergeometric
from src.stats.distributions.geometric import run_geometric
from src.stats.distributions.negative_binomial import run_negative_binomial
from src.stats.distributions.poisson import run_poisson
from src.stats.distributions.exponential import run_exponential
from src.stats.distributions.weibull import run_weibull
from src.stats.distributions.chi_square import run_chi_square
from src.stats.distributions.gamma import run_gamma
from src.stats.distributions.beta import run_beta
from src.stats.distributions.student_t import run_student_t
from src.stats.distributions.cauchy import run_cauchy
from src.stats.distributions.lognormal import run_lognormal
from src.stats.distributions.f import run_f


def run_pdf_examples():
    """32개 문항의 입력·연산 의미·가정과 결과를 반환합니다.

    Args:
        없음.
    Returns:
        list: 문항 id, 연산 의미, 보충 가정, 엄격 JSON 호환 결과.
    Raises:
        ValueError: 분포 모수·조회 오류.
    """
    records = []

    def add(number, letter, meaning, assumptions, result, point_probability=None):
        record = {
            "id": str(number) + letter,
            "operation": meaning,
            "assumptions": assumptions,
            "result": result.to_dict(json_safe=True),
        }
        if point_probability is not None:
            record["continuous_point_probability"] = point_probability
        records.append(record)

    add(1, "a", "PMF 성공", [], run_bernoulli(x=1, overrides={"p": 0.7}))
    add(1, "b", "PMF 불량", [], run_bernoulli(x=1, overrides={"p": 0.05}))
    add(2, "a", "PMF 10회 중4회", [], run_binomial(x=4, overrides={"n": 10, "p": 0.5}))
    add(2, "b", "PMF 20명 중5명", [], run_binomial(x=5, overrides={"n": 20, "p": 0.3}))
    add(
        3,
        "a",
        "80에서의 밀도; 정확히80의 연속 점확률은0",
        ["연속 정규 수명 모형"],
        run_normal(x=80, overrides={"mu": 75, "sigma": 10}),
        0.0,
    )
    add(
        3,
        "b",
        "85에서의 밀도; 정확히85의 연속 점확률은0",
        ["시험 점수를 연속 모형으로 해석; 정수 점수 확률은 별도 구간 필요"],
        run_normal(x=85, overrides={"mu": 70, "sigma": 10}),
        0.0,
    )
    add(
        4,
        "a",
        "PMF 비복원5개 중불량2개",
        [],
        run_hypergeometric(x=2, overrides={"N": 50, "K": 10, "n": 5}),
    )
    add(
        4,
        "b",
        "PMF 비복원4명 중관리자1명",
        [],
        run_hypergeometric(x=1, overrides={"N": 30, "K": 8, "n": 4}),
    )
    add(
        5,
        "a",
        "PMF 첫성공까지4회",
        [],
        run_geometric(x=4, overrides={"p": 0.2, "counting": "trials"}),
    )
    add(
        5,
        "b",
        "PMF 첫성공까지3회",
        [],
        run_geometric(x=3, overrides={"p": 0.5, "counting": "trials"}),
    )
    add(
        6,
        "a",
        "PMF 네번째 성공이총10회째",
        [],
        run_negative_binomial(x=10, overrides={"r": 4, "p": 0.3, "counting": "trials"}),
    )
    add(
        6,
        "b",
        "PMF 다섯번째 성공이총15회째",
        [],
        run_negative_binomial(x=15, overrides={"r": 5, "p": 0.6, "counting": "trials"}),
    )
    add(7, "a", "PMF 평균3회 중2회", [], run_poisson(x=2, overrides={"rate": 3}))
    add(7, "b", "PMF 평균5회 중3회", [], run_poisson(x=3, overrides={"rate": 5}))
    add(
        8,
        "a",
        "CDF 고장까지2시간 이내",
        [],
        run_exponential(x=2, overrides={"rate": 0.25}),
    )
    add(
        8,
        "b",
        "CDF 다음방문5분 이내",
        [],
        run_exponential(x=5, overrides={"rate": 0.1}),
    )
    add(
        9,
        "a",
        "CDF 수명800 이하",
        ["평균1000에서 shape1.5를 이용해 scale 역산"],
        run_weibull(
            x=800, overrides={"shape": 1.5, "scale": 1000 / gamma(1 + 1 / 1.5)}
        ),
    )
    add(
        9,
        "b",
        "CDF 수명600 이하",
        ["평균500에서 shape2를 이용해 scale 역산"],
        run_weibull(x=600, overrides={"shape": 2.0, "scale": 500 / gamma(1.5)}),
    )
    add(10, "a", "SF 자유도9에서15 초과", [], run_chi_square(x=15, overrides={"df": 9}))
    add(
        10,
        "b",
        "SF 자유도5에서10 초과",
        ["문항의 결과를 상측확률로 해석"],
        run_chi_square(x=10, overrides={"df": 5}),
    )
    add(
        11,
        "a",
        "SF 대기5시간 이상",
        [],
        run_gamma(x=5, overrides={"shape": 1.0, "scale": 3.0}),
    )
    add(
        11,
        "b",
        "SF 수익률7% 이상",
        ["평균5%만으로 shape 미확정. 예제에 shape2,scale0.025를 추가 가정"],
        run_gamma(x=0.07, overrides={"shape": 2.0, "scale": 0.025}),
    )
    add(
        12,
        "a",
        "SF 잠재 재구매확률0.7 이상",
        ["평균0.6만으로 alpha/beta 미확정. 예제에 집중도5를 가정하여 alpha3,beta2"],
        run_beta(x=0.7, overrides={"alpha": 3.0, "beta": 2.0}),
    )
    add(
        12,
        "b",
        "잠재 성공확률분포 조회",
        ["임계값도 미지정. 예제에 임계값0.5와 집중도5를 추가:alpha2,beta3"],
        run_beta(x=0.5, overrides={"alpha": 2.0, "beta": 3.0}),
    )
    add(
        13,
        "a",
        "t=2의좌/우꼬리; 양측은2*SF(abs(t))",
        ["문항에서 대립 방향 미지정"],
        run_student_t(x=2, overrides={"df": 15}),
    )
    add(
        13,
        "b",
        "t=1.5의좌/우꼬리; 양측은2*SF(abs(t))",
        ["문항에서 대립 방향 미지정"],
        run_student_t(x=1.5, overrides={"df": 10}),
    )
    add(
        14,
        "a",
        "x0.5에서의 코시 밀도; 점확률0",
        [],
        run_cauchy(x=0.5, overrides={"x0": 0.0, "gamma": 1.0}),
        0.0,
    )
    add(
        14,
        "b",
        "x1.2에서의 코시 밀도; 점확률0",
        ["문항의 모수가 부족해 x0=0,gamma=1 가정"],
        run_cauchy(x=1.2),
        0.0,
    )
    add(
        15,
        "a",
        "SF 지정수익값 이상",
        ["문항에 mu/sigma/임계값이 미지정. 예제에 mu0,sigma1,임계값2 가정"],
        run_lognormal(x=2, overrides={"mu": 0.0, "sigma": 1.0}),
    )
    add(
        15,
        "b",
        "SF 수명1000 이상",
        ["로그 좌표 mu7,sigma0.5를 추가 가정"],
        run_lognormal(x=1000, overrides={"mu": 7.0, "sigma": 0.5}),
    )
    add(
        16,
        "a",
        "SF F통계량3.5 이상",
        ["문항의 확률을 상측 검정확률로 해석"],
        run_f(x=3.5, overrides={"df_num": 10, "df_den": 15}),
    )
    add(
        16,
        "b",
        "SF F통계량4.2 이상",
        ["문항의 확률을 상측 검정확률로 해석"],
        run_f(x=4.2, overrides={"df_num": 8, "df_den": 12}),
    )
    return records


def main():
    """통계 분야에서 기존 32개 분포 예제를 실행합니다.

    Args:
        None: 명시된 예제 모수와 보충 가정을 사용합니다.
    Returns:
        list: 확률/밀도·모수·해석·보충 가정 결과.
    Raises:
        ValueError/OSError: 분포 모수·레시피 오류.
    """
    return run_pdf_examples()


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, allow_nan=False, indent=2))
