"""등록부와 소스의 기능 설명에서 단계별 기능 카탈로그를 생성합니다.

분석 의존성을 import하거나 모형을 학습하지 않습니다. --check는 누락·중복·생성 차이를 검사합니다.
"""
import ast
from collections import OrderedDict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STAGES = OrderedDict([
 ('exploration', ('1. 데이터 이해·EDA·상관분석', '분포·결측·집단 차이·연관을 탐색합니다. 목표와의 관계로 처리 규칙을 정하면 학습 자료에서만 탐색하고 CV 안에서 검증합니다. 상관·그림만으로 인과나 자동 변수 제거를 결정하지 않습니다.')),
 ('features', ('2. Feature 정의·전처리', '대상 열과 처리 순서를 선언합니다. 미적합 factory를 모델 Pipeline에 연결하고 각 학습 폴드에서 fit합니다. 목표 변환과 시퀀스 임베딩은 일반 표 Feature 처리와 분리합니다.')),
 ('sampling', ('3. 학습 표본 처리', 'X와 y의 행을 함께 처리합니다. 학습 폴드에만 fit_resample하고 검증·최종 평가 행과 발생률은 보존합니다. Cook 기준 제거는 기본 동작이 아닌 명시적 학습 정책입니다.')),
 ('selection', ('4. Feature 선택', '열 선택 역시 학습/CV 안에서 수행합니다. VIF는 공선성, AIC/BIC는 동일 반응·동일 관측의 우도와 복잡도, wrapper는 CV 예측 점수를 사용하므로 기준을 혼용하지 않습니다.')),
 ('models', ('5. 모형 적합·예측', '입력 형태·타깃 정의역·분할 방식은 기법별로 다릅니다. IID 내부 분할 runner와 외부 Pipeline용 미적합 estimator를 구분합니다. 범주 입력·새 관측 예측 지원 여부는 상세 명세에서 확인합니다.')),
 ('search', ('6. 검증·튜닝·후보 선택', '최종 평가 자료를 전달하지 않습니다. IID·시간·그룹 설계를 정하고 같은 fold/scorer로 비교합니다. 선택된 CV 점수와 최종 보류 성능은 별도 결과입니다.')),
 ('inference', ('7. 추론·가정 진단·확률 계산', '관측 설계와 입력 단위를 먼저 정합니다. 가설 검정·구간 추정·확률 계산을 구별하며, 비기각은 가정의 증명이 아닙니다. 진단 후 변경한 모형은 학습/CV 단계로 돌아가 검증합니다.')),
 ('metrics', ('8. 예측 평가·모형 비교', '적합·샘플링 없이 같은 행 순서의 정답과 예측을 평가합니다. 확률 열은 명시한 클래스 순서와 대응해야 합니다. 최종 평가 결과로 후보를 계속 고르지 않습니다.')),
 ('plots', ('9. 진단·평가·설명 시각화', '호출자가 만든 ax에만 그립니다. DTO의 수치/표와 실행 객체를 구분하고 subplots 조합·저장은 호출자가 수행합니다. 데이터나 이미 계산한 지표를 사용하며 그림 자체가 모형을 교정하지 않습니다.')),
 ('decision', ('10. 목적함수 기반 의사결정 최적화', '예측모형의 하이퍼파라미터 탐색과 별개입니다. 목적함수·제약·변수 범위를 입력하고 해의 실행가능성과 최적성 근거를 구분합니다.')),
])
FUNCTIONS = {
 'numeric-imputation':'평균·중앙값·최빈값·상수 결측 대치와 선택적 결측 indicator',
 'numeric-scaling':'standard·minmax·robust·maxabs·행 normalize 또는 항등 변환',
 'numeric-power':'Yeo–Johnson·Box–Cox와 선택적 표준화',
 'categorical-imputation':'범주형 최빈값·상수 결측 대치',
 'categorical-encoding':'one-hot·ordinal·frequency 인코딩, 학습 vocabulary와 미지 범주 정책',
 'feature-preprocessing':'FeatureSpec의 열·처리 순서를 ColumnTransformer로 결합',
 'rope':'시퀀스 임베딩의 회전 위치 인코딩; 표 범주 인코딩과 별도',
 'correlation-pearson':'수치 Feature 쌍의 선형 상관·p값·선택적 다중비교 보정',
 'correlation-spearman':'수치 Feature 순위의 단조 연관·p값·선택적 보정',
 'correlation-kendall':'수치 Feature 순위의 Kendall tau·정확/근사 p값',
 'correlation-cramers-v':'범주 Feature 교차표의 Cramér’s V·독립성 p값',
 'correlation-phi':'이진 범주 쌍의 부호 있는 Phi·독립성 p값',
 'one-sample-mean':'단일 평균의 Z/t 검정·평균 신뢰구간',
 'two-sample-mean':'독립 두 평균의 Student/Welch/Z 검정·차이 구간',
 'paired-t':'대응 관측 차이의 평균 t 검정·차이 구간',
 'one-sample-variance':'정규 모집단의 모분산 카이제곱 검정·구간',
 'one-sample-proportion':'단일 비율 exact/score 검정·Clopper–Pearson/Wilson 구간',
 'two-sample-proportion':'독립 비율의 pooled Z·Newcombe 차이 구간',
 'sign':'단일 기준·대응 차이의 부호 정확 이항 검정',
 'wilcoxon':'단일·대응 차이의 signed-rank 정확/점근 검정',
 'runs':'순서 이진열의 runs 무작위성 정규근사 검정',
 'mann-whitney':'독립 두 집단의 Mann–Whitney U 정확/점근 검정',
 'kruskal-wallis':'독립 여러 집단의 순위 Kruskal–Wallis 검정',
 'mood-median':'공동 중앙값 위/아래 빈도를 이용한 독립 집단 비교',
 'friedman':'완전 대응 블록×조건의 Friedman 순위 검정',
 'repeated-measures-anova':'완전 균형 일원 반복측정 ANOVA·선택적 GG 보정',
 'anderson-darling':'지정 분포 적합 후 Anderson–Darling 임계값 판단; p값 없음',
 'kolmogorov-smirnov':'고정 모수 KS 또는 정규/지수 추정 모수 Lilliefors',
 'hotelling':'독립 두 다변량 평균 벡터의 Hotelling T² 검정',
 'box-m':'독립 다변량 집단의 공분산 동질성 Box M 근사 검정',
 'odds-ratio':'독립 2×2 표의 OR·log-Wald 검정·구간',
 'risk-ratio':'cohort 2×2 표의 RR·log-Wald 검정·구간',
 'fisher-exact':'고정 주변합 Fisher exact·RxC 열거/Monte Carlo',
 'cmh':'층화 2×2 표의 공통 OR·CMH 검정',
 'bowker':'같은 대상의 전후 다범주 표 대칭성 Bowker 검정',
 'cochran-q':'같은 대상의 여러 이진 조건 성공확률 Cochran Q 검정',
 'adf':'시간순 자료의 단위근 ADF 검정·차수 선택',
 'chi-square-homogeneity':'독립 모집단의 범주 분포 동질성 Pearson 검정',
}
METRICS = {
 'accuracy':'전체 정분류 비율', 'precision':'예측 양성의 정답 비율', 'recall':'실제 양성의 검출 비율',
 'f1':'precision·recall 조화평균', 'fbeta':'재현율 가중 F-beta', 'specificity':'실제 음성의 정분류 비율',
 'balanced-accuracy':'클래스별 recall 평균', 'mcc':'혼동행렬 기반 Matthews 상관', 'cohen-kappa':'우연 일치 보정 분류 일치도',
 'confusion-matrix':'지정 클래스 순서의 오분류 빈도표', 'roc-auc':'양성 점수 순위의 ROC 면적',
 'average-precision':'recall 증가 가중 평균 precision(AP)', 'pr-auc':'PR 곡선 사다리꼴 면적; AP와 구분',
 'log-loss':'실제 클래스 확률의 음의 로그손실', 'brier-score':'이진 양성 확률의 제곱오차', 'gini':'ROC AUC 기반 순위 Gini',
 'ks':'누적 양성·음성 분리의 최대 차이', 'lift':'선별 구간 양성률/전체 양성률', 'gain':'선별 구간의 전체 양성 포착 비율',
 'lift-curve':'선별 비율별 lift 곡선 수치', 'gain-curve':'선별 비율별 cumulative gain 곡선 수치',
 'roc-curve':'임계값별 FPR·TPR·threshold 배열', 'precision-recall-curve':'임계값별 precision·recall 배열',
 'calibration':'확률 bin별 평균 예측·실제 발생률; 보정기 적합 없음',
 'mae':'절대오차 평균', 'mse':'제곱오차 평균', 'rmse':'제곱오차 평균의 제곱근', 'r2':'평가 정답 평균 대비 오차 감소 비율',
 'adjusted-r2':'표본 수·설명변수 수 보정 R²', 'median-absolute-error':'절대오차 중앙값',
 'mape':'실제값 기준 절대 백분율 오차', 'smape':'실제·예측 크기를 함께 쓰는 대칭 백분율 오차',
 'wape':'절대오차 합/실제 절댓값 합', 'msle':'비음수 실제·예측의 log1p 제곱오차', 'rmsle':'MSLE 제곱근',
 'explained-variance':'잔차 분산 기준 설명력', 'max-error':'최대 절대오차', 'pinball-loss':'지정 분위수 예측의 pinball 손실',
 'silhouette':'같은 군집·다른 군집의 거리 기반 분리도', 'davies-bouldin':'군집 내부 산포 대비 군집 간 분리 비율',
 'calinski-harabasz':'군집 간/내 산포비', 'ari':'정답 군집과 우연 보정 Rand 일치도', 'ami':'우연 보정 상호정보량',
 'nmi':'정규화 상호정보량', 'homogeneity':'예측 군집 안의 정답 클래스 순도', 'completeness':'동일 정답 클래스의 군집 결집',
 'v-measure':'homogeneity·completeness 조화평균',
 'mase':'학습 naive 절대 차분으로 나눈 평가 MAE', 'rmsse':'학습 naive 제곱 차분으로 나눈 평가 RMSE',
 'horizon-errors':'예측 horizon별 오차 집계',
}
INPUTS = {
 'one-sample-mean':'수치 표본 x·귀무 평균·선택적 알려진 sigma', 'two-sample-mean':'독립 수치 표본 x/y·귀무 차이',
 'paired-t':'같은 대상의 동일 길이 x/y', 'one-sample-variance':'수치 표본·귀무 분산',
 'one-sample-proportion':'성공 수·시행 수·귀무 비율', 'two-sample-proportion':'두 집단의 성공/시행 수',
 'chi-square':'관측 도수·선택적 기대도수', 'independence':'독립 관측 교차 빈도표', 'mcnemar':'대응 이진 2×2 표',
 'f-test':'독립 정규 수치 표본 x/y', 'normality':'비상수 수치 표본', 'homogeneity':'독립 수치 집단들',
 'friedman':'행=블록, 열=조건의 완전 대응 행렬', 'repeated-measures-anova':'행=피험자, 열=조건의 완전 대응 행렬',
 'cochran-q':'피험자×조건의 0/1 행렬', 'cmh':'층×2×2 도수표', 'bowker':'전후 같은 범주 k×k 대응표',
 'odds-ratio':'노출/비노출×사건/비사건 2×2 표', 'risk-ratio':'cohort 노출/비노출×사건/비사건 표',
 'hotelling':'독립 두 (표본,변수) 행렬', 'box-m':'여러 집단의 (표본,변수) 행렬',
 'oneway-anova':'독립 수치 집단들', 'twoway-anova':'수치 반응·두 범주 요인', 'manova':'다변량 반응·집단 라벨',
 'posthoc':'독립 집단들·선택적 집단 이름', 'glm-logistic':'수치 설계 X·0/1 y·선택적 X_new',
 'glm-poisson':'수치 설계 X·비음수 정수 y·선택적 log 노출 offset',
 'glm-gamma':'수치 설계 X·양의 연속 y·선택적 X_new',
 'nonlinear-regression':'설명변수 1개·수치 y·모형/초기값/경계',
 'naive-bayes-categorical':'원 범주 Feature X·클래스 y', 'naive-bayes-bernoulli':'0/1 X 또는 명시 임계값·클래스 y',
 'naive-bayes-binomial':'0..n_trials 정수 X·클래스 y', 'naive-bayes-poisson':'동일 노출의 비음수 정수 X·클래스 y',
 'naive-bayes-multinomial':'비음수 빈도/가중 빈도 X·클래스 y', 'naive-bayes-complement':'비음수 빈도 X·클래스 y',
 'feature-preprocessing':'열 이름 있는 X_train·FeatureSpec 목록·선택적 X_new',
 'rope':'시퀀스 임베딩·위치 정보', 'regression-target-transform':'학습 수치 y·선택적 새 y',
 'regression-information-selection':'유한 full-rank X·동일 행의 원 반응 y',
 'regression-vif-selection':'학습 수치 Feature X', 'regression-polynomial':'학습 X·선택적 X_new',
 'regression-influence-filter':'학습 X/y·적합 OLS 진단/처리 정책',
 'regression-collinearity':'수치 설계 X', 'regression-ols':'수치 설계 X/y·선택적 X_new',
 'regression-weighted':'수치 X/y·선택적 X_new',
 'apriori':'거래별 항목 목록', 'fp-growth':'거래별 항목 목록', 'eclat':'거래별 항목 목록',
 'sna':'노드·가중 연결 목록', 'genetic-algorithm':'목적함수·변수 경계·정수열·제약함수',
 'production-allocation':'설비 비용/처리량·예산·가용 수·GA 설정',
 'metric-comparison':'동일 정답·모형별 정렬된 예측/점수', 'metric-cv-summary':'fold별 수치 평가값·선택적 크기',
}

def registry():
 tree=ast.parse((ROOT/'src/Recipes/loader.py').read_text())
 return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_REGISTRY' for t in n.targets))

def stage(key,module):
 if module.startswith('src.Optimization.'):return 'decision'
 if key.startswith('optimization-') or key in ('regression-workflow','timeseries-rolling-selection'):return 'search'
 if key.startswith('resampling-') or key=='regression-influence-filter':return 'sampling'
 if key in ('regression-vif-selection','regression-information-selection','wrapper-selection','regularized-selection'):return 'selection'
 if key.startswith('correlation-') or key.startswith(('plot-numeric-','plot-quality-','plot-categorical-','plot-relationships-')):return 'exploration'
 if module.startswith('src.Preprocessing.'):return 'features'
 if module.startswith('src.Visualization.'):return 'plots'
 if module.startswith('src.Evaluation.'):return 'metrics'
 if module.startswith('src.stats.'):return 'inference'
 return 'models'

def purpose(key,tree):
 if key in FUNCTIONS:return FUNCTIONS[key]
 if key.startswith('metric-'):
  for family in ('classification','regression','clustering','time-series'):
   prefix='metric-'+family+'-'
   if key.startswith(prefix):
    name=key[len(prefix):]
    return family+' 분야 지표 묶음' if name=='metrics' else METRICS[name]
 doc=ast.get_docstring(tree) or ''
 heading=next((line.strip().split('.py -',1)[1].strip() for line in doc.splitlines() if '.py -' in line),None)
 if not heading:raise ValueError('기능 설명 누락: '+key)
 return heading

def inputs(key,module,group):
 if key in INPUTS:return INPUTS[key]
 if key.startswith('distribution-'):return '분포 모수·선택적 x/확률 q/구간·난수 개수'
 if key.startswith('correlation-'):return '상관행렬' if key.endswith('matrix-transform') else '두 Feature x/y 또는 열 이름 있는 표(행렬 API)'
 if key.startswith('survival-'):return '시간·사건 지시자·집단/공변량(기법별)'
 if key.startswith('regression-') and group=='inference':return '적합 OLS DTO·필요시 제약/축소 모형'
 if key.startswith('timeseries-') or key=='tf-time-series':return '시간순 정규 간격 시계열·선택적 외생변수'
 if key.startswith('tf-'):return '수치 X/y; Dense 2D, CNN 3D/4D, RNN 3D'
 if key.startswith('resampling-'):return '학습 X/y·자료형/이웃 조건에 맞는 설정'
 if key.startswith('plot-'):return '기법별 원자료/계산 결과·호출자 ax'
 if key.startswith('metric-classification-'):return '같은 행 정답·예측 라벨/점수/확률'
 if key.startswith('metric-regression-'):return '같은 행·같은 단위 y_true/y_pred'
 if key.startswith('metric-clustering-'):return '군집 라벨·좌표 X 또는 참 라벨'
 if key.startswith('metric-time-series-'):return '시간/horizon 정렬 정답·예측·학습 기준열'
 if module.startswith('src.Preprocessing.'):return '학습 Feature·선택적 새 Feature'
 if group=='selection' or '.LinearModels.' in module or '.NaiveBayes.' in module or '.Trees.' in module or '.NeuralNetworks.' in module:return '수치 Feature X·타깃 y'
 if '.Decomposition.' in module:return '수치 X·LDA에는 클래스 y도 필요'
 if '.Clustering.' in module:return '유한 수치 Feature X'
 if group=='search':return '학습 X/y·미적합 estimator/Pipeline·후보 공간'
 if group=='inference':return '기법별 표본/집단/순서열(상세 명세 확인)'
 raise ValueError('입력 설명 누락: '+key)

def outputs(tree):
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name.endswith('OutputDTO'))
 names=[n.target.id for n in cls.body if isinstance(n,ast.AnnAssign)]
 exclude={'method','task','parameters','warnings','n_samples','sample_size','elapsed_seconds'}
 selected=[n for n in names if n not in exclude]
 # 기법별 첫 핵심 필드는 실제 DTO에서 추출. 전체 필드는 링크의 해당 모듈 소스/상세 명세에 있음.
 return ' · '.join('`'+n+'`' for n in selected[:4])

def build():
 reg=registry(); buckets={k:[] for k in STAGES}
 for key,(module,profile,task,tasks) in sorted(reg.items()):
  path=ROOT/(module.replace('.','/')+'.py');tree=ast.parse(path.read_text());group=stage(key,module)
  buckets[group].append('| [`'+key+'`](../'+path.relative_to(ROOT).as_posix()+') | '+purpose(key,tree).replace('|','／')+' | '+inputs(key,module,group)+' | '+outputs(tree)+' |')
 recipes=len(list((ROOT/'src/Recipes').glob('*.json')))
 text='# 단계별 전체 모듈 기능 카탈로그\n\n등록 모듈은 '+str(len(reg))+'개, 모델·목적별 JSON 레시피는 '+str(recipes)+'개입니다. 모든 등록 키를 아래 한 단계에 한 번씩 배치했습니다. 단계는 활용 목적에 따른 분류이며, 모든 모듈을 차례로 실행해야 한다는 의미가 아닙니다.\n\n'
 text+='[전체 파이프라인](ANALYSIS_PIPELINE.md)에서 책임과 자료 경계를 먼저 확인하고, [문서 안내](README.md)에서 분야별 상세 명세를 찾으세요. 표의 결과 열은 실제 전용 DTO의 대표 필드이며 전체 필드 목록은 아닙니다. 모듈 키 링크는 해당 기능의 구현·Profile·전체 DTO 계약으로 연결됩니다.\n\n'
 text+='Dataset·FeatureSpec·Recipe 로더·OutputDTO는 공통 기반이므로 등록 모듈 수에 포함하지 않습니다. 이들의 기능은 [기반 계약](ANALYSIS_PIPELINE.md#공통-기반의-역할)에 따로 정리했습니다. 같은 함수가 여러 목적을 지원하더라도 등록 키당 한 번 셉니다. 빈 파일·호환 import·내부 helper는 구현 모듈로 중복 집계하지 않습니다.\n\n'
 text+='| 단계 | 모듈 수 |\n| --- | ---: |\n'+''.join('| '+STAGES[k][0]+' | '+str(len(v))+' |\n' for k,v in buckets.items())+'| 합계 | '+str(len(reg))+' |\n\n'
 for group,rows in buckets.items():
  title,desc=STAGES[group];text+='## '+title+'\n\n'+desc+'\n\n| 모듈 | 구현 기능 | 필요한 입력 | 대표 결과(DTO) |\n| --- | --- | --- | --- |\n'+'\n'.join(rows)+'\n\n'
 text+='## 기능 명세와 API 필드 참조의 구분\n\n이 목록은 기능·자료·결과를 선택하기 위한 명세입니다. [핵심 API 필드](reference/CORE_API.md)와 분야별 문서는 파라미터의 정확한 타입·기본값·배열 대응·실패 조건을 보충합니다. 현재 기능 범위는 등록부와 소스 기준이며 제안 단계의 기능을 포함하지 않습니다.\n\n생성: `python tools/docs/build_module_catalog.py`; 누락·중복·생성 차이 검사: `python tools/docs/build_module_catalog.py --check`. 이 작업은 모형을 적합하거나 선택 패키지를 로드하지 않습니다.\n'
 return text,len(reg)

if __name__=='__main__':
 content,count=build();path=ROOT/'docs/ANALYSIS_CATALOG.md'
 if '--check' in sys.argv:
  if path.read_text()!=content:raise SystemExit('기능 카탈로그를 재생성하세요.')
  print('기능 카탈로그 일치: %d개 등록 모듈'%count)
 else:path.write_text(content);print('생성: %d개 등록 모듈'%count)
