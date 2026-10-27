"""
================================================================================
scenario_inference.py - 상황형 풀이의 가설·실행 수치·의사결정 기록. Python 3.7 호환.
================================================================================
설명:
    - 상황형 풀이의 가설·실행 수치·의사결정 기록. Python 3.7 호환.
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
import json
from pathlib import Path
import numpy as np
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests

_REPORTS = []
_PATH = None
_QUESTION = None


def reset_reports(directory):
    global _REPORTS, _PATH
    _REPORTS = []
    _PATH = Path(directory) / '가설검정_답안.json'
    _PATH.write_text('[]\n', encoding='utf-8')


def set_question(identifier):
    global _QUESTION
    _QUESTION = identifier


def report_test(method, h0, h1, statistic, p_value, alternative_claim,
                assumptions, df=None, adjusted_p=None, alpha=.05):
    """실제로 계산한 p값만 보고하며 기각 실패를 동일성 증명으로 쓰지 않는다."""
    p = float(p_value)
    used = p if adjusted_p is None else float(adjusted_p)
    if not np.isfinite([p, used]).all() or not (0 <= p <= 1 and 0 <= used <= 1):
        raise ValueError('검정 p값은 유한한 [0, 1] 값이어야 합니다.')
    reject = used < alpha
    row = dict(question=_QUESTION, method=method, H0=h0, H1=h1,
               statistic=float(statistic), p_value=p, alpha=float(alpha),
               reject_H0=bool(reject), assumptions=assumptions,
               conclusion=('귀무가설을 기각한다. ' + alternative_claim + '는 통계적 근거가 있다.'
                           if reject else '귀무가설을 기각하지 못한다. ' + alternative_claim +
                           '는 통계적 근거가 충분하지 않다. 귀무가설의 참을 입증한 것은 아니다.'))
    if df is not None:
        row['df'] = np.asarray(df).tolist()
    if adjusted_p is not None:
        row['adjusted_p_value'] = used
        row['decision_basis'] = 'Holm 보정 p값'
    print(json.dumps(row, ensure_ascii=False, indent=2))
    _REPORTS.append(row)
    if _PATH is not None:
        _PATH.write_text(json.dumps(_REPORTS, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return row


def report_anova(model, response, factors):
    table = anova_lm(model, typ=2)
    for effect, row in table.iterrows():
        if effect == 'Residual':
            continue
        interaction = ':' in effect
        label = (' × '.join(factors) + ' 상호작용' if interaction else
                 next((f for f in factors if 'C(' + f + ')' == effect), effect) + ' 주효과')
        report_test('독립 집단 요인 모형의 Type II F 검정: ' + label,
                    label + '가 없다', label + '가 있다', row['F'], row['PR(>F)'],
                    response + '에 대한 ' + label + '가 있다',
                    '독립 관측·정규 오차·등분산·식별 가능한 설계. 상호작용 유의 시 셀 평균과 조건부 효과를 우선 해석한다.',
                    df=[float(row['df']), float(table.loc['Residual', 'df'])])


def report_correlations(df, target, features):
    results = [stats.pearsonr(df[target], df[x]) for x in features]
    corrected = multipletests([r[1] for r in results], method='holm')[1]
    for name, result, p_holm in zip(features, results, corrected):
        report_test('Pearson 선형 상관: ' + name, 'rho(' + target + ',' + name + ')=0',
                    'rho(' + target + ',' + name + ') != 0', result[0], result[1],
                    target + '와 ' + name + ' 사이에 선형 관계가 있다',
                    '독립 관측·선형 관계·이변량 정규 추론, 이상값 점검. 다중 비교 Holm 보정; 인과가 아님.',
                    adjusted_p=p_holm)


def report_adf(series, label):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series)
    report_test('ADF: ' + label, '단위근이 존재한다', '단위근이 없다', result[0], result[1],
                label + '에 단위근이 없다',
                '상수항 포함·AIC 시차 선택. 기각이 모든 형태의 정상성 또는 정규성을 입증하지 않는다.')


def report_residuals(residuals, label, lags, model_df=0):
    import statsmodels.api as sm
    residuals = np.asarray(residuals)
    normal = stats.shapiro(residuals)
    report_test('Shapiro-Wilk: ' + label, '잔차의 주변 분포가 정규이다',
                '잔차의 주변 분포가 정규가 아니다', normal.statistic, normal.pvalue,
                label + '가 정규 분포에서 벗어난다',
                '적합 잔차의 진단용 p값이며 독립 표본 검정과 같은 확증 해석에 주의한다.')
    lb = sm.stats.acorr_ljungbox(residuals, lags=lags, model_df=model_df, return_df=True)
    for lag, row in lb.iterrows():
        report_test('Ljung-Box: ' + label + ', lag=' + str(lag),
                    'lag 1~' + str(lag) + '의 자기상관이 모두 0이다',
                    '해당 시차 중 하나 이상 자기상관이 0이 아니다', row.lb_stat, row.lb_pvalue,
                    label + '에 잔여 시차 의존성이 있다',
                    '적합한 AR·MA 계수 수를 자유도에서 차감한다. 정규성·등분산 검정과 구분한다.',
                    df=int(lag) - model_df)


def report_regression_diagnostics(model):
    normal = stats.shapiro(model.resid)
    bp = het_breuschpagan(model.resid, model.model.exog)
    report_test('Shapiro-Wilk 잔차 진단', '잔차 분포가 정규이다', '잔차 분포가 정규가 아니다',
                normal.statistic, normal.pvalue, '잔차 분포가 정규에서 벗어난다',
                '선택된 모형의 잔차를 이용한 탐색 진단이며 선택 후 추론의 한계가 있다.')
    report_test('Breusch-Pagan 잔차 진단', '오차 분산이 일정하다', '오차 분산이 일정하지 않다',
                bp[0], bp[1], '잔차에 이분산이 존재한다',
                '독립 관측과 지정된 분산 설명 구조에 의존하는 점근적 진단이다.',
                df=model.model.exog.shape[1] - 1)
