"""복원 연습 문제 공통 풀이. 원자료가 없는 문항은 명시된 합성 자료로 실행합니다.

Python 3.7 문법. 모델 평가 전처리는 학습 내부에서 적합합니다.
"""
import re
import inspect
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve,
                             mean_squared_error, r2_score, silhouette_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt


def numeric_correlations(df, target, features):
    rows = []
    for name in features:
        r, p = stats.pearsonr(df[target], df[name])
        rows.append({'feature': name, 'r': r, 'p_value': p})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    corr = df[[target] + features].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=65)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax); fig.tight_layout(); plt.show()
    return table


def select_ols(df, target, features, method='stepwise'):
    """절편 포함 OLS AIC 엄격 개선 선택. 선택 후 p값은 탐색적이며 인과가 아닙니다."""
    if method not in ('forward', 'backward', 'stepwise'):
        raise ValueError('method 오류')
    def fit(names):
        X = sm.add_constant(df[names].astype(float), has_constant='add')
        if np.linalg.matrix_rank(X) != X.shape[1]:
            return None
        return sm.OLS(df[target].astype(float), X).fit()
    selected = list(features) if method == 'backward' else []
    model = fit(selected)
    if model is None:
        raise ValueError('설계행렬이 완전 다중공선성입니다.')
    history = []
    for step in range(100):
        candidates = []
        if method != 'backward':
            for name in features:
                if name not in selected:
                    names = selected + [name]; candidate = fit(names)
                    if candidate is not None: candidates.append((candidate.aic, names, candidate))
        if method != 'forward':
            for name in selected:
                names = [x for x in selected if x != name]; candidate = fit(names)
                if candidate is not None: candidates.append((candidate.aic, names, candidate))
        if not candidates: break
        aic, names, best = min(candidates, key=lambda x: x[0])
        if aic >= model.aic - 1e-8: break
        selected, model = names, best
        history.append({'step': step + 1, 'features': ','.join(names), 'AIC': aic})
    print('선택 변수:', selected); print(pd.DataFrame(history)); print(model.summary())
    print('선택 후 계수 추론은 다중 탐색의 영향을 받습니다. p값을 확증 검정으로 단정하지 않습니다.')
    return model


def residual_diagnostics(model):
    normal = stats.shapiro(model.resid)
    bp = het_breuschpagan(model.resid, model.model.exog)
    print({'Shapiro_p': normal.pvalue, 'Breusch_Pagan_p': bp[1],
           'Durbin_Watson': sm.stats.durbin_watson(model.resid)})
    print('p>=0.05는 귀무가설을 기각하지 못함입니다. 독립성은 관측 설계·시간 순서도 확인합니다.')
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].scatter(model.fittedvalues, model.resid, s=12); ax[0].axhline(0, color='red')
    ax[0].set(xlabel='Fitted', ylabel='Residual')
    sm.qqplot(model.resid, line='s', ax=ax[1]); fig.tight_layout(); plt.show()


def anova(df, target, group, second=None):
    from statsmodels.formula.api import ols
    formula = '%s ~ C(%s)' % (target, group)
    if second: formula += ' * C(%s)' % second
    model = ols(formula, df).fit()
    table = anova_lm(model, typ=2)
    print(table)
    print('유의하면 적어도 한 평균/효과가 다릅니다. 모든 쌍이 다르다는 뜻은 아닙니다.')
    if not second:
        groups = [v[target].to_numpy() for _, v in df.groupby(group, observed=True)]
        print('등분산 Levene:', stats.levene(*groups))
        if table.iloc[0]['PR(>F)'] < .05:
            print(pairwise_tukeyhsd(df[target], df[group], alpha=.05))
        else: print('ANOVA 유의하지 않아 사후검정을 주 결론으로 수행하지 않습니다.')
    print('합성 자료의 예시 추론. 정규성·등분산·독립성 및 표본 설계를 실제 자료에서 재검증하세요.')
    return model


def tabular_pipeline(X, estimator):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X if c not in numeric]
    kw = {'sparse_output': False} if 'sparse_output' in inspect.signature(OneHotEncoder).parameters else {'sparse': False}
    prep = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('encoder', OneHotEncoder(handle_unknown='ignore', **kw))]), categorical)])
    return Pipeline([('preprocessor', prep), ('model', estimator)])


def classify(df, target, features, out, seed=1234, algorithms=None):
    """전처리를 train에만 적합. 동일 7:3 test에서 비교, 재튜닝하지 않습니다."""
    X = df[features].replace('', np.nan).copy(); y = df[target]
    if 'Date' in X: X['Date'] = pd.to_datetime(X['Date']).map(pd.Timestamp.toordinal)
    train, test = train_test_split(np.arange(len(df)), test_size=.3, random_state=seed, stratify=y)
    algorithms = algorithms or {'logistic': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1),
        'gradient_boosting': GradientBoostingClassifier(random_state=seed)}
    results = []; fitted = {}; fig, ax = plt.subplots(figsize=(6, 4))
    for name, estimator in algorithms.items():
        pipe = tabular_pipeline(X, estimator); pipe.fit(X.iloc[train], y.iloc[train])
        pred = pipe.predict(X.iloc[test]); prob = pipe.predict_proba(X.iloc[test])
        row = {'model': name, 'accuracy': accuracy_score(y.iloc[test], pred),
               'f1_weighted': f1_score(y.iloc[test], pred, average='weighted')}
        if len(pipe.classes_) == 2:
            binary = (y.iloc[test] == pipe.classes_[1]).astype(int)
            row['ROC_AUC'] = roc_auc_score(binary, prob[:, 1])
            fpr, tpr, _ = roc_curve(binary, prob[:, 1]); ax.plot(fpr, tpr, label=name)
        pd.DataFrame({'row_id': test, 'actual': y.iloc[test].to_numpy(), 'prediction': pred}).to_csv(out/(name+'_predictions.csv'), index=False)
        results.append(row); fitted[name] = pipe
    ax.plot([0, 1], [0, 1], '--'); ax.set(xlabel='FPR', ylabel='TPR'); ax.legend(); plt.show()
    table = pd.DataFrame(results); print(table); table.to_csv(out/'model_comparison.csv', index=False)
    return table, fitted, train, test


def cluster(X, out, original=None, k_values=range(2, 6)):
    scaled = StandardScaler().fit_transform(X)
    choices = []
    for k in k_values:
        model = KMeans(k, random_state=1234, n_init=10).fit(scaled)
        choices.append({'k': k, 'silhouette': silhouette_score(scaled, model.labels_, sample_size=min(1000, len(X)), random_state=1234),
                        'inertia': model.inertia_})
    table = pd.DataFrame(choices); print(table)
    k = int(table.loc[table.silhouette.idxmax(), 'k'])
    labels = KMeans(k, random_state=1234, n_init=10).fit_predict(scaled)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3)); ax[0].plot(table.k, table.inertia, 'o-'); ax[1].plot(table.k, table.silhouette, 'o-'); plt.show()
    data = (original.copy() if original is not None else X.copy()); data['clust'] = labels
    data.to_csv(out/'cluster_labels.csv', index=False)
    print(data.select_dtypes(include=np.number).groupby(data.clust).mean())
    return data, table


def clean_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^가-힣A-Za-z\s]', ' ', text)).strip()


def lexical_pos(texts, lexicon, pos='Noun'):
    """동봉 합성 텍스트의 명시 POS 사전으로 분석. 일반 한국어 형태소 분석기가 아닙니다."""
    words = set(lexicon.loc[lexicon.pos == pos, 'word'])
    return [[word for word in clean_text(text).split() if word in words] for text in texts]


def word_plot(tokens, top=10):
    counts = Counter(word for row in tokens for word in row)
    table = pd.DataFrame(counts.most_common(top), columns=['word', 'count'])
    print(table); fig, ax = plt.subplots(figsize=(8, 3)); ax.bar(table.word, table['count']); ax.tick_params(axis='x', rotation=45); plt.tight_layout(); plt.show()
    return table


def seasonal_forecast(df, out, period=12, orders=None):
    """학습 앞 80%에서만 SARIMA 차수 선정; validation 없음은 AIC로 선정."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    values = pd.Series(df['value'].to_numpy(), index=pd.to_datetime(df['date']))
    train = values.iloc[:int(len(values)*.8)].copy(); test = values.iloc[len(train):]
    train = train.interpolate().ffill().bfill()
    candidates = []
    for order in (orders or [(1,0,0), (1,1,0), (0,1,1)]):
        model = SARIMAX(train, order=order, seasonal_order=(1,0,0,period)).fit(disp=False, maxiter=100)
        candidates.append((model.aic, order, model))
    table = pd.DataFrame([{'order': order, 'AIC': aic,
                          'converged': model.mle_retvals.get('converged', False)}
                         for aic, order, model in candidates])
    print(table);table.to_csv(out/'sarima_candidates.csv', index=False)
    eligible = [entry for entry in candidates if np.isfinite(entry[0]) and entry[2].mle_retvals.get('converged', False)]
    if not eligible:
        raise ValueError('수렴한 유한 AIC 후보가 없습니다. 후보나 최대 반복 수를 검토하세요.')
    _, order, best = min(eligible, key=lambda x: x[0])
    pred = np.asarray(best.forecast(len(test))); actual = test.to_numpy(); mask = np.isfinite(actual)
    metrics = {'order': order, 'RMSE': float(np.sqrt(mean_squared_error(actual[mask], pred[mask]))),
               'MAE': float(np.mean(np.abs(actual[mask]-pred[mask]))), 'ADF_train_p': float(adfuller(train)[1])}
    print(metrics)
    pd.DataFrame({'date': test.index, 'actual': actual, 'predicted': pred}).to_csv(out/'time_series_predictions.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 3)); ax.plot(values.index, values, label='actual'); ax.plot(test.index, pred, label='forecast'); ax.legend(); plt.show()
    return best, metrics


def kaplan_meier(df):
    """동일 시각 사건을 검열보다 먼저 처리하는 KM 및 log-log Greenwood CI."""
    rows=[]; survival=1.; greenwood=0.
    for t in sorted(df.time.unique()):
        risk = int((df.time >= t).sum()); events = int(((df.time == t) & (df.event == 1)).sum())
        censor = int(((df.time == t) & (df.event == 0)).sum())
        survival *= 1 - events / risk
        if events and risk > events: greenwood += events / (risk*(risk-events))
        low, high = survival, survival
        if 0 < survival < 1:
            z = stats.norm.ppf(.975); se = np.sqrt(greenwood)/abs(np.log(survival))
            low = np.exp(-np.exp(np.log(-np.log(survival))+z*se))
            high = np.exp(-np.exp(np.log(-np.log(survival))-z*se))
        rows.append({'time': t, 'at_risk': risk, 'events': events, 'censored': censor, 'survival': survival, 'low': low, 'high': high})
    result=pd.DataFrame(rows);print(result)
    fig, ax=plt.subplots(figsize=(7,3));ax.step([0]+result.time.tolist(),[1]+result.survival.tolist(),where='post');ax.fill_between(result.time,result.low,result.high,step='post',alpha=.2);ax.set(xlabel='Time',ylabel='Survival',ylim=(0,1.05));plt.show()
    return result
