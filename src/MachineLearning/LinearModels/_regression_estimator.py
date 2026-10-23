"""Feature 전처리·VIF·명시 Cook 처리·타깃 변환을 CV fit 안에 두는 OLS estimator."""
from sklearn.base import BaseEstimator,RegressorMixin,clone
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
from src.Preprocessing._common import feature_array
from src.Preprocessing.numeric.imputation import build_numeric_imputation_transformer
from src.Preprocessing.numeric.scaling import build_numeric_scaling_transformer
from src.Preprocessing.numeric.power import build_numeric_power_transformer
from src.Preprocessing.feature_pipeline import make_feature_spec,build_feature_preprocessor
from src.Preprocessing.regression.polynomial import build_polynomial_transformer,PolynomialProfile
from src.Preprocessing.regression.target_transform import build_target_transformer,TargetTransformProfile
from src.Preprocessing.regression.vif_selection import build_vif_selector,VIFSelectionProfile
from src.Preprocessing.regression.influence_filter import run_influence_filter,InfluenceFilterProfile
from src.stats.regression.ols import run_ols,OLSProfile


class RegressionEstimator(RegressorMixin,BaseEstimator):
    """CV가 clone해서 폴드 내부에서 모든 선택·통계 적합을 수행합니다."""
    def __init__(self,preprocessor=None,power='none',degree=1,target_method='none',vif_threshold=None,influence_action='flag',cook_threshold=None,max_remove_fraction=.1,cov_type='nonrobust',hac_maxlags=1,smearing=False,alpha=.05):
        """미적합 설정을 보존합니다.

        Args:
            preprocessor: 별도 FeatureSpec ColumnTransformer 또는 None.
            power: 기본 numeric Feature의 none/yeo-johnson/box-cox.
            degree: 기본 numeric polynomial 차수 1~3.
            target_method: none/log/log1p/yeo-johnson/box-cox.
            vif_threshold: 선택 VIF 기준 또는 None.
            influence_action: flag/drop; drop만 학습 행을 처리.
            cook_threshold: 선택 양수 Cook 기준 또는 None=4/n.
            max_remove_fraction: 삭제 비율 한도.
            cov_type: OLS 공분산 방법.
            hac_maxlags: HAC 최대 지연.
            smearing: log/log1p 예측의 학습 잔차 Duan 보정.
            alpha: 계수/구간 유의수준.
        Returns:
            None: 미적합 설정.
        Raises:
            None: fit에서 검증.
        """
        self.preprocessor=preprocessor;self.power=power;self.degree=degree;self.target_method=target_method;self.vif_threshold=vif_threshold;self.influence_action=influence_action;self.cook_threshold=cook_threshold;self.max_remove_fraction=max_remove_fraction;self.cov_type=cov_type;self.hac_maxlags=hac_maxlags;self.smearing=smearing;self.alpha=alpha

    def _raw(self,X):
        """원 Feature와 열 순서를 확인합니다.

        Args:
            X: DataFrame/2차원 배열; custom 전처리는 범주도 허용.
        Returns:
            DataFrame: 원 열을 보존한 자료.
        Raises:
            ValueError/TypeError: shape/열 오류.
        """
        if self.preprocessor is None:
            a,names=feature_array(X,True,True);return pd.DataFrame(a,index=X.index if isinstance(X,pd.DataFrame) else None,columns=names)
        raw=X.copy() if isinstance(X,pd.DataFrame) else pd.DataFrame(X)
        if raw.empty or not raw.columns.is_unique or any(not isinstance(c,str) for c in raw.columns):raise ValueError('custom Feature 전처리는 고유 문자열 열의 DataFrame을 사용하세요.')
        return raw

    def _prepare(self,raw,target):
        """지정 학습 행에서만 Feature·타깃·선택 변환기를 적합합니다.

        Args:
            raw: 현재 학습 원 Feature.
            target: 현재 학습 원 타깃.
        Returns:
            tuple: 선택 설계 DataFrame, 변환 타깃.
        Raises:
            ValueError: custom Feature 설정·전처리·rank 오류.
        """
        if self.preprocessor is None:
            steps=[('impute',build_numeric_imputation_transformer()),('drop_constant',build_vif_selector(VIFSelectionProfile(threshold=None)))]
            if self.power!='none':steps.append(('power',build_numeric_power_transformer(overrides={'method':self.power})))
            steps.extend([('scale',build_numeric_scaling_transformer()),('polynomial',build_polynomial_transformer(PolynomialProfile(degree=self.degree)))])
            feature=make_feature_spec('numeric',raw.columns.tolist(),steps);processor=build_feature_preprocessor([feature])
        else:
            if self.power!='none' or self.degree!=1:raise ValueError('custom Feature 전처리의 power/polynomial은 해당 Feature 단계에서 지정하세요.')
            processor=clone(self.preprocessor)
        self.preprocessor_=processor.fit(raw,target);matrix=self.preprocessor_.transform(raw)
        if hasattr(matrix,'toarray'):matrix=matrix.toarray()
        names=self.preprocessor_.get_feature_names_out().tolist();expanded=pd.DataFrame(matrix,index=raw.index,columns=names)
        self.selector_=build_vif_selector(VIFSelectionProfile(threshold=self.vif_threshold)).fit(expanded)
        selected=self.selector_.transform(expanded);self.output_features_=self.selector_.get_feature_names_out().tolist()
        self.target_transformer_=build_target_transformer(TargetTransformProfile(method=self.target_method)).fit(target)
        return pd.DataFrame(selected,index=raw.index,columns=self.output_features_),self.target_transformer_.transform(target)

    def fit(self,X,y):
        """학습/CV 폴드 안에서 선택과 적합을 완료합니다.

        Args:
            X: 원 학습 Feature; 결측은 학습 imputer에 전달.
            y: 유한 원 단위 타깃.
        Returns:
            RegressionEstimator: 적합한 자신.
        Raises:
            ValueError/TypeError: 입력·전처리·제거·추론 정의역 오류.
        """
        if np.iscomplexobj(y):raise ValueError('실수 타깃이 필요합니다.')
        raw=self._raw(X);target=np.asarray(y,dtype=float)
        if isinstance(X,pd.DataFrame) and isinstance(y,pd.Series) and not X.index.equals(y.index):raise ValueError('타깃 인덱스 불일치')
        if target.ndim!=1 or len(target)!=len(raw) or not np.isfinite(target).all() or np.ptp(target)==0:raise ValueError('대응 비상수 타깃이 필요합니다.')
        if self.power not in ('none','yeo-johnson','box-cox') or self.influence_action not in ('flag','drop') or not isinstance(self.smearing,bool):raise ValueError('모형 처리 설정 오류')
        if self.smearing and self.target_method not in ('log','log1p'):raise ValueError('smearing은 log/log1p에서만 지원합니다.')
        self.feature_names_in_=np.asarray(raw.columns,dtype=object);self.n_features_in_=raw.shape[1]
        design,transformed=self._prepare(raw,target);self.kept_positions_=np.arange(len(raw));self.removed_positions_=np.array([],dtype=int);self.influence_review_=None
        if self.influence_action=='drop':
            reviewed=run_influence_filter(design,transformed,InfluenceFilterProfile(action='drop',cook_threshold=self.cook_threshold,max_remove_fraction=self.max_remove_fraction))
            self.kept_positions_=reviewed.kept_positions;self.removed_positions_=reviewed.removed_positions;self.influence_review_=reviewed.review_table
            # 삭제가 명시된 경우 유지한 원 행에서 전처리·타깃·VIF를 다시 적합합니다.
            raw=raw.iloc[self.kept_positions_];target=target[self.kept_positions_];design,transformed=self._prepare(raw,target)
        self.inference_=run_ols(design,transformed,profile=OLSProfile(alpha=self.alpha,cov_type=self.cov_type,hac_maxlags=self.hac_maxlags))
        self.model_=self.inference_.model;self.smearing_factor_=float(np.mean(np.exp(self.model_.resid))) if self.smearing else 1.
        if not np.isfinite(self.smearing_factor_):raise ValueError('smearing overflow')
        self.n_training_samples_=len(raw)
        return self

    def transform_features(self,X):
        """새 Feature에 학습 변환·선택만 적용합니다.

        Args:
            X: 같은 원 열 순서의 새 자료.
        Returns:
            DataFrame: OLS가 사용할 최종 Feature.
        Raises:
            ValueError/NotFittedError: 미적합·스키마 오류.
        """
        check_is_fitted(self,'model_');raw=self._raw(X)
        if raw.columns.tolist()!=self.feature_names_in_.tolist():raise ValueError('새 원 Feature 이름/순서 불일치')
        matrix=self.preprocessor_.transform(raw)
        if hasattr(matrix,'toarray'):matrix=matrix.toarray()
        names=self.preprocessor_.get_feature_names_out().tolist();expanded=pd.DataFrame(matrix,index=raw.index,columns=names)
        return pd.DataFrame(self.selector_.transform(expanded),index=raw.index,columns=self.output_features_)

    def predict(self,X):
        """새 행의 예측을 원 y 단위로 반환합니다.

        Args:
            X: 원 Feature 새 자료.
        Returns:
            ndarray: 원 단위 예측; log는 선택 smearing 적용.
        Raises:
            ValueError/NotFittedError: 스키마·역변환 오류.
        """
        design=self.transform_features(X);design.insert(0,'Intercept',1.);prediction=np.asarray(self.model_.predict(design))
        if self.smearing:
            result=np.exp(prediction)*self.smearing_factor_-(1 if self.target_method=='log1p' else 0)
        else:result=self.target_transformer_.inverse_transform(prediction)
        if not np.isfinite(result).all():raise ValueError('예측 inverse overflow')
        return result
