"""반응 분포·링크별 GLM 적합, 우도·잔차·조건부 영향도 계약."""

import warnings
from dataclasses import asdict
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from scipy.stats import chi2
from src.stats.regression._common import numeric_data


def fit_family(X, y, X_new, offset, new_offset, profile, family, dto):
    """반응 정의역과 full-rank 설계를 검사한 후 고정 family를 적합합니다.

    Args:
        X,y: 같은 행의 유한 수치 Feature와 대응 반응.
        X_new: 선택 예측 Feature; 같은 열 이름/순서.
        offset,new_offset: 알려진 고정 선형예측자 offset. Poisson만 허용.
        profile: 전용 GLM 설정.
        family: logistic/poisson/gamma 고정 family 이름.
        dto: 전용 결과 타입.
    Returns:
        OutputDTO: 조건부 계수·우도·분포에 맞는 잔차·새 평균 예측.
    Raises:
        ValueError/TypeError: 정의역·rank·열·노출 offset·미수렴/분리 오류.
    """
    raw, target = numeric_data(X, y)
    if (np.ptp(raw.values, axis=0) == 0).any() or "Intercept" in raw:
        raise ValueError("상수 Feature를 제거하세요.")
    design = raw.copy()
    design.insert(0, "Intercept", 1.0)
    if (
        len(raw) <= design.shape[1] + 1
        or np.linalg.matrix_rank(design) != design.shape[1]
    ):
        raise ValueError("full-rank 설계와 잔차 자유도가 필요합니다.")
    if family == "logistic":
        if set(np.unique(target)) != {0.0, 1.0}:
            raise ValueError("logistic 반응은 두 값 모두 포함한 0/1입니다.")
        distribution = sm.families.Binomial()
    elif family == "poisson":
        if (target < 0).any() or (target != np.rint(target)).any():
            raise ValueError("Poisson 반응은 비음수 정수 횟수입니다.")
        distribution = sm.families.Poisson()
    else:
        if (target <= 0).any():
            raise ValueError("Gamma 반응은 양수 연속량입니다.")
        distribution = sm.families.Gamma(
            link=getattr(sm.families.links, "Log", sm.families.links.log)()
        )
    if family != "poisson" and (offset is not None or new_offset is not None):
        raise ValueError("이 API의 offset은 Poisson 노출에만 허용합니다.")

    def check_offset(value, n, label):
        if value is None:
            return None
        a = np.asarray(value, dtype=float)
        if a.shape != (n,) or not np.isfinite(a).all():
            raise ValueError(label + "은 같은 행 수의 유한 log exposure입니다.")
        return a

    off = check_offset(offset, len(raw), "offset")
    if new_offset is not None and X_new is None:
        raise ValueError("X_new 없이 new_offset을 지정할 수 없습니다.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = sm.GLM(target, design, family=distribution, offset=off).fit(
                maxiter=profile.maxiter, cov_type=profile.cov_type
            )
        except PerfectSeparationError as exc:
            raise ValueError("완전 분리로 GLM 추론이 불가능합니다.") from exc
    if (
        not model.converged
        or not np.isfinite(model.params).all()
        or not np.isfinite(model.bse).all()
        or any("separation" in str(w.message).lower() for w in caught)
    ):
        raise ValueError(
            "미수렴 또는 완전/준완전 분리로 안정적인 GLM 추론이 불가능합니다."
        )
    interval = np.asarray(model.conf_int(alpha=profile.alpha))
    coef = np.asarray(model.params)
    table = pd.DataFrame(
        dict(
            term=design.columns,
            coefficient=coef,
            standard_error=np.asarray(model.bse),
            z_statistic=np.asarray(model.tvalues),
            p_value=np.asarray(model.pvalues),
            ci_lower=interval[:, 0],
            ci_upper=interval[:, 1],
            exp_coefficient=np.exp(coef),
        )
    )
    influence = model.get_influence(observed=True)
    diagnostics = pd.DataFrame(
        dict(
            row_position=np.arange(len(raw)),
            fitted_mean=np.asarray(model.fittedvalues),
            pearson_residual=np.asarray(model.resid_pearson),
            deviance_residual=np.asarray(model.resid_deviance),
            leverage=np.asarray(influence.hat_matrix_diag),
            cooks_distance=np.asarray(influence.cooks_distance[0]),
        ),
        index=raw.index,
    )
    metrics = dict(
        family=family,
        link="logit" if family == "logistic" else "log",
        llf=float(model.llf),
        aic=float(model.aic),
        bic=float(-2 * model.llf + np.log(len(raw)) * len(coef)),
        criterion_parameter_count=len(coef),
        deviance=float(model.deviance),
        pearson_dispersion=float(model.pearson_chi2 / model.df_resid),
        scale=float(model.scale),
        mcfadden_pseudo_r2=None,
        lr_statistic=None,
        lr_p_value=None,
    )
    if family in ("logistic", "poisson"):
        null = sm.GLM(
            target, np.ones((len(raw), 1)), family=distribution, offset=off
        ).fit(maxiter=profile.maxiter)
        lr = max(0.0, 2 * (model.llf - null.llf))
        metrics.update(
            mcfadden_pseudo_r2=float(1 - model.llf / null.llf),
            lr_statistic=float(lr),
            lr_p_value=float(chi2.sf(lr, raw.shape[1])),
        )
    predictions = None
    if X_new is not None:
        future, _ = numeric_data(X_new)
        if future.columns.tolist() != raw.columns.tolist():
            raise ValueError("새 Feature 이름/순서 오류")
        if off is not None and new_offset is None:
            raise ValueError("학습 offset 사용 시 새 노출의 log offset도 필요합니다.")
        future.insert(0, "Intercept", 1.0)
        new_off = check_offset(new_offset, len(future), "new_offset")
        predictions = np.asarray(model.predict(future, offset=new_off))
        if not np.isfinite(predictions).all():
            raise ValueError("예측 overflow")
    notes = [
        "GLM은 OLS 잔차 정규성·보통 R²·t/전체 F를 요구하는 모형이 아닙니다. z·우도 검정은 점근적이며 선택 후 추론을 보정하지 않습니다.",
        "AIC/BIC는 동일 family·반응·관측 행·offset 조건에서 비교하세요. Gamma의 추정 Pearson scale에 조건부인 기준은 근사입니다.",
        "GLM Cook은 GLM 가중 곡률의 근사 영향이며 OLS Cook과 구분합니다. 임계값 초과가 오류·삭제 사유를 증명하지 않습니다.",
    ]
    if family == "logistic":
        notes.append(
            "확률이 아닌 log odds에 대한 사양·독립성·분리·보정·판별력을 검토하세요. pseudo R²를 보통 R²로 비교하지 않습니다."
        )
    if family == "poisson":
        notes.append(
            "조건부 평균=분산·독립 횟수 가정을 점검하세요. 노출이 다르면 알려진 log exposure offset을 지정합니다. 과산포/과도한0은 다른 분포 검토 대상입니다."
        )
    if family == "gamma":
        notes.append(
            "양수 연속 반응·log 평균 링크·Var(Y|X)=phi*mu²를 가정합니다. 평균/분산 사양과 Pearson/deviance 잔차를 확인하세요."
        )
    notes.extend(
        str(w.message) for w in caught if "deprecated" not in str(w.message).lower()
    )
    return dto(model, table, metrics, diagnostics, predictions, asdict(profile), notes)
