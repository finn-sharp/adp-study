"""
================================================================================
researched_descriptive.py - 제공 사전·거래·설문의 기술 분석. 원 관측 단위와 분모를 보존합니다.
================================================================================
설명:
    - 제공 사전·거래·설문의 기술 분석. 원 관측 단위와 분모를 보존합니다.
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

import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.Association.apriori import run_apriori
from src.MachineLearning.Decomposition.pca import run_pca
from src.stats.tests.parametric.Average.two_sample import run_two_sample_mean
from researched_core import finish_plot


def execute(ctx, sec, step, state):
    return {"text": text, "assoc": association, "survey": survey}[sec["family"]](
        ctx, sec, step, state
    )


def text(ctx, sec, step, s):
    if step == 1:
        reviews = pd.read_csv(ctx.folder / "data/reviews.csv", keep_default_na=False)
        lex = pd.read_csv(ctx.folder / "data/lexicon.csv")
        assert reviews.id.is_unique and lex.token.is_unique
        tokens = [re.findall("[a-z]+", v.lower()) for v in reviews.text]
        s.update(reviews=reviews, lex=lex.set_index("token"), tokens=tokens)
        return dict(
            documents=len(reviews),
            empty_documents=int(reviews.text.str.strip().eq("").sum()),
            unique_ids=bool(reviews.id.is_unique),
            conclusion="제공 사전의 통제 어휘만 사용한다. 실제 문맥 품사 분석·부정 범위 해석은 수행하지 않는다.",
        )
    if step == 2:
        lex = s["lex"]
        pos = sec["settings"]["pos"]
        vocab = set(lex.index[lex.pos == pos])
        tf = Counter(t for row in s["tokens"] for t in row if t in vocab)
        df = Counter(t for row in s["tokens"] for t in set(row) if t in vocab)
        table = pd.DataFrame(
            [
                dict(token=t, frequency=n, document_frequency=df[t])
                for t, n in tf.most_common()
            ]
        )
        s["frequency"] = table
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(table.token, table.frequency)
        ax.set_title(pos + " token frequency")
        finish_plot()
        return dict(
            part_of_speech=pos,
            frequency=table,
            denominator_documents=len(s["reviews"]),
            conclusion="빈도는 반복 토큰 수이고 문서빈도는 등장 문서 수다. 두 집계를 같은 비율로 해석하지 않는다.",
        )
    if step == 3:
        rows = []
        lex = s["lex"]
        unknown = Counter()
        for i, tokens in enumerate(s["tokens"]):
            known = [t for t in tokens if t in lex.index]
            adjectives = [
                float(lex.loc[t, "polarity"])
                for t in known
                if lex.loc[t, "pos"] == "ADJ"
            ]
            unknown.update(t for t in tokens if t not in lex.index)
            rows.append(
                dict(
                    id=int(s["reviews"].id.iloc[i]),
                    group=s["reviews"].group.iloc[i],
                    token_count=len(tokens),
                    dictionary_matches=len(known),
                    coverage=len(known) / len(tokens) if tokens else np.nan,
                    adjective_count=len(adjectives),
                    sentiment=float(np.mean(adjectives)) if adjectives else np.nan,
                )
            )
        s["scores"] = pd.DataFrame(rows)
        return dict(
            scored_documents=int(s["scores"].sentiment.notna().sum()),
            unscored_documents=int(s["scores"].sentiment.isna().sum()),
            mean_coverage=s["scores"].coverage.mean(),
            unknown_terms=[dict(token=t, count=n) for t, n in unknown.most_common()],
            scores=s["scores"],
            conclusion="미일치 문서는 결측 점수로 남겼다. not great도 단순 사전에서는great의 양성으로 남으므로 문맥 감성을 판정하지 못한다.",
        )
    scores = s["scores"]
    scores.to_csv(ctx.out / "document_scores.csv", index=False)
    table = (
        scores.groupby("group")
        .agg(
            documents=("id", "size"),
            scored=("sentiment", "count"),
            mean_sentiment=("sentiment", "mean"),
            mean_coverage=("coverage", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(table.group, table.mean_sentiment)
    ax.set_title("Dictionary sentiment (valid documents)")
    finish_plot()
    return dict(
        summary=table,
        output="document_scores.csv",
        conclusion="집단별 유효 문서 수와 일치율을 함께 비교한다. 이 합성 사전 점수는 실제 상영관 만족도의 추정치가 아니다.",
    )


def association(ctx, sec, step, s):
    if step == 1:
        d = pd.read_csv(ctx.folder / "data/baskets.csv", keep_default_na=False)
        baskets = [set(v.split("|")) - {""} for v in d["items"]]
        assert d.id.is_unique
        s["baskets"] = baskets
        return dict(
            transactions=len(baskets),
            empty_transactions=sum(not b for b in baskets),
            conclusion="빈 거래를 포함한100건이 지지도 분모다. 거래 내 반복 품목은 한 번만 센다.",
        )
    if step == 2:
        fit = run_apriori(
            [sorted(b) for b in s["baskets"]],
            overrides=dict(
                min_support=0.15, min_confidence=0.5, min_lift=1.0, max_length=3
            ),
        )
        s["rules"] = fit.rules
        return dict(
            itemsets=fit.itemsets,
            rules=fit.rules,
            rule_count=len(fit.rules),
            conclusion="임계값을 사전 고정한 기술 연관 규칙이다. 출력 규칙 개수는 성과나 매출 효과가 아니다.",
        )
    if step == 3:
        rules = s["rules"]
        if rules.empty:
            return dict(
                rule_exists=False,
                conclusion="조건을 만족하는 규칙이 없어 검산할 규칙이 없다.",
            )
        r = rules.iloc[0]
        a = set(r.antecedents)
        b = set(r.consequents)
        n = len(s["baskets"])
        na = sum(a <= t for t in s["baskets"])
        nb = sum(b <= t for t in s["baskets"])
        nab = sum((a | b) <= t for t in s["baskets"])
        support = nab / n
        confidence = nab / na
        lift = confidence / (nb / n)
        return dict(
            rule_exists=True,
            antecedents=sorted(a),
            consequents=sorted(b),
            n=n,
            count_a=na,
            count_b=nb,
            count_ab=nab,
            support=support,
            confidence=confidence,
            lift=lift,
            reported_support=r.support,
            reported_confidence=r.confidence,
            reported_lift=r.lift,
            conclusion="동시 구매와 조건부 비율을 원 거래에서 검산했다. 높은 lift를 구매 유발의 인과로 읽지 않는다.",
        )
    basket = {"bread"}
    recommend = []
    for r in s["rules"].itertuples():
        if set(r.antecedents) <= basket:
            for item in sorted(set(r.consequents) - basket):
                recommend.append(dict(item=item, confidence=r.confidence, lift=r.lift))
    recommend = sorted(
        recommend, key=lambda r: (-r["confidence"], -r["lift"], r["item"])
    )
    recommend = (
        pd.DataFrame(recommend).drop_duplicates("item")
        if recommend
        else pd.DataFrame(columns=["item", "confidence", "lift"])
    )
    return dict(
        basket=sorted(basket),
        recommendations=recommend,
        conclusion="이미 구매한bread는 제외했다. 적용 규칙이 없으면 추천을 비우고, 추가 매출은 별도 운영 실험으로 확인해야 한다.",
    )


def alpha_values(frame):
    """과제에서 지정한 두 공식을 직접 계산합니다. 결측 없는 문항행렬만 입력."""
    k = frame.shape[1]
    cov = frame.cov().to_numpy()
    corr = frame.corr().to_numpy()
    rbar = (corr.sum() - k) / (k * (k - 1))
    raw = k / (k - 1) * (1 - np.trace(cov) / cov.sum())
    standardized = k * rbar / (1 + (k - 1) * rbar)
    return dict(
        items=k,
        alpha=raw,
        standardized_alpha=standardized,
        mean_interitem_correlation=rbar,
        total_score_variance=cov.sum(),
    )


def survey(ctx, sec, step, s):
    if step == 1:
        d = pd.read_csv(ctx.folder / "data/survey.csv")
        raw = d[["q1", "q2", "q3", "q4"]]
        assert raw.stack().between(1, 5).all()
        mask = raw.notna().all(axis=1)
        x = raw.loc[mask].copy()
        correct = x.copy()
        correct[["q2", "q4"]] = 6 - correct[["q2", "q4"]]
        g = d.loc[mask, "group"]
        s.update(raw=x, correct=correct, group=g)
        table = (
            correct.assign(group=g)
            .groupby("group")
            .agg(["mean", "std", "skew", lambda v: v.kurt()])
        )
        table.columns = [
            "%s_%s" % (a, b if b != "<lambda_0>" else "kurtosis")
            for a, b in table.columns
        ]
        return dict(
            complete_rows=len(correct),
            excluded_rows=int((~mask).sum()),
            group_descriptives=table,
            conclusion="결측 응답%d명을 제외하고 q2/q4 방향을 맞췄다. 집단은 관측 범주이며 처치 효과를 뜻하지 않는다."
            % ((~mask).sum()),
        )
    if step == 2:
        before = alpha_values(s["raw"])
        after = alpha_values(s["correct"])
        return dict(
            before=before,
            after=after,
            conclusion="방향 보정 뒤 공분산 alpha %.4f, 표준화 alpha %.4f. 두 식은 척도의 일관성 지표이며 단일차원·타당도 증명이 아니다."
            % (after["alpha"], after["standardized_alpha"]),
        )
    if step == 3:
        fit = run_pca(
            s["correct"],
            overrides=dict(
                n_components=2, scaling="standard", whiten=False, random_state=ctx.seed
            ),
        )
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].bar([1, 2], fit.explained_variance_ratio)
        axes[0].set_xticks([1, 2], labels=["PC1", "PC2"])
        axes[0].set_title("PCA explained variance")
        for group in sorted(s["group"].unique()):
            mask = (s["group"] == group).to_numpy()
            axes[1].scatter(fit.embedding[mask, 0], fit.embedding[mask, 1], label=group)
        axes[1].legend(title="Group")
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].set_title("PCA scores (observed groups)")
        finish_plot()
        return dict(
            explained_variance_ratio=fit.explained_variance_ratio,
            components=fit.components,
            reconstruction_error=fit.reconstruction_error,
            conclusion="PCA2축은 총분산 %.3f를 설명한다. 부호는 임의이며 PCA를 공통요인분석으로 부르지 않는다."
            % sum(fit.explained_variance_ratio),
        )
    scores = s["correct"].mean(axis=1)
    a = scores[s["group"] == "A"]
    b = scores[s["group"] == "B"]
    fit = run_two_sample_mean(
        a,
        b,
        overrides=dict(
            equal_var=False, known_var=False, alternative="two-sided", alpha=0.05
        ),
    )
    result = fit.to_dict()
    result["conclusion"] = (
        "A−B 평균 점수 차이의 양측 Welch 검정 p=%.4g. 관측 표본의 비교이며 인과·모집단 대표성을 주장하지 않는다."
        % fit.p_value
    )
    return result
