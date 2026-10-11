"""
================================================================================
text.py - 텍스트 정제, Count/TF-IDF, 문서 분류 및 LDA 토픽
================================================================================
설명:
    - 텍스트마이닝 PDF의 벡터화·토픽 모델링과 08장 텍스트 분류를 모듈화합니다.
    - 기본 실행에는 외부 코퍼스 다운로드, Java, gensim이 필요하지 않습니다.
    - 한국어 형태소 분석은 tokenizer 콜백으로 연결합니다.
작성 정보: 작성일 2026-10-10, 버전 1.0.0
변경 이력: 2026-10-10 v1.0.0 학습 문서 전용 벡터화와 토픽 결과 명세 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가
"""
__version__ = "1.1.0"
__status__ = "Development"

import re
from src.output import OutputDTO
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import type_of_target
from src.MachineLearning.common import SupervisedResult, _metrics


@dataclass
class TextClassificationOutputDTO(SupervisedResult):
    """TextClassification 모듈 전용 반환 DTO.

    Attributes:
        공통 SupervisedOutputDTO(Result 호환 별칭)의 모든 데이터·모델·지표 필드를 상속합니다.
        to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.
    """
    pass

def clean_text(text: str) -> str:
    """HTML/URL을 제거하고 공백을 정리합니다.

    Args:
        text: 입력 문장.
    Returns:
        str: 한글·영문·숫자·문장부호를 보존한 정제 문자열.
    Raises:
        ValueError: 문자열이 아닌 입력.
    """
    if not isinstance(text, str):
        raise ValueError("text는 문자열이어야 합니다.")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class TextProfile:
    """텍스트 벡터화·분류 조건.

    Attributes:
        vectorizer: count 또는 tfidf.
        tokenizer: 선택적 단어/형태소 분리 콜백. 반환값은 토큰 목록.
        stop_words: None, english 또는 명시적 불용어 목록.
        ngram_range: (최소 n, 최대 n).
        min_df: 최소 문서 빈도 (정수 수 또는 0~1 비율).
        max_df: 최대 문서 빈도 (정수 수 또는 0~1 비율).
        max_features: 최대 어휘 수. None이면 제한 없음.
        test_size: 분류 테스트 비율 (0, 1).
        random_state: 분할/모델 시드.
        C: 로지스틱 분류 규제 역수.
        max_iter: 분류 최대 반복 수.
        param_grid: 선택적 Pipeline 탐색 공간. vectorizer__/model__ 접두사 필수.
        cv: 분류 탐색 폴드 수 (최소 2).
    """
    vectorizer: str = "tfidf"
    tokenizer: Optional[Callable] = None
    stop_words: Any = None
    ngram_range: Tuple[int, int] = (1, 1)
    min_df: Any = 1
    max_df: Any = 1.0
    max_features: Optional[int] = None
    test_size: float = 0.3
    random_state: int = 1234
    C: float = 1.0
    max_iter: int = 1000
    param_grid: Optional[Dict[str, Any]] = None
    cv: int = 3


@dataclass
class VectorizationOutputDTO(OutputDTO):
    """문서 벡터화 결과.

    Attributes:
        vectorizer: 학습 문서에만 적합한 벡터화 객체.
        train_matrix: 학습 문서 희소 행렬.
        test_matrix: 선택적 테스트 문서 희소 행렬. 없으면 None.
        feature_names: 희소 행렬 열 순서의 토큰 이름.
    """
    vectorizer: Any
    train_matrix: Any
    test_matrix: Any
    feature_names: np.ndarray


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
VectorizationResult = VectorizationOutputDTO


def _documents(documents):
    if isinstance(documents, str):
        raise ValueError("문자열 하나가 아니라 문서 목록을 입력하세요.")
    docs = list(documents)
    if not docs or any(not isinstance(d, str) for d in docs):
        raise ValueError("비어 있지 않은 문자열 문서 목록이 필요합니다.")
    return docs


def _vectorizer(profile):
    if profile.vectorizer not in ("count", "tfidf"):
        raise ValueError("vectorizer는 count 또는 tfidf여야 합니다.")
    klass = CountVectorizer if profile.vectorizer == "count" else TfidfVectorizer
    return klass(tokenizer=profile.tokenizer, token_pattern=None if profile.tokenizer else r"(?u)\b\w\w+\b",
                 preprocessor=clean_text, stop_words=profile.stop_words, ngram_range=profile.ngram_range,
                 min_df=profile.min_df, max_df=profile.max_df, max_features=profile.max_features)


def _names(vectorizer):
    method = getattr(vectorizer, "get_feature_names_out", None) or vectorizer.get_feature_names
    return np.asarray(method())


def run_vectorization(train_documents: Any, profile: Optional[TextProfile] = None,
                       test_documents: Any = None) -> VectorizationOutputDTO:
    """학습 문서에서 어휘/IDF를 적합하고 테스트 문서는 변환만 합니다.

    Args:
        train_documents: 학습 문서 목록.
        profile: 벡터화 설정.
        test_documents: 선택적 테스트 문서 목록. 새 토큰은 학습 어휘에 추가되지 않음.
    Returns:
        VectorizationOutputDTO: 적합 벡터화 객체, 희소 행렬 및 토큰 목록.
    Raises:
        ValueError: 잘못된 입력/설정 또는 빈 어휘.
    """
    p = profile if profile is not None else TextProfile()
    vectorizer = _vectorizer(p)
    train = vectorizer.fit_transform(_documents(train_documents))
    test = vectorizer.transform(_documents(test_documents)) if test_documents is not None else None
    return VectorizationOutputDTO(vectorizer, train, test, _names(vectorizer))


def run_text_classification(documents: Any, y: Any,
                             profile: Optional[TextProfile] = None) -> TextClassificationOutputDTO:
    """층화 분할 후 벡터화 → 로지스틱 분류 Pipeline을 적합합니다.

    Args:
        documents: 문자열 문서 목록.
        y: 같은 길이의 1차원 분류 라벨. 각 클래스는 분할 가능한 표본 수가 필요합니다.
        profile: 벡터화·분류·CV 설정.
    Returns:
        TextClassificationOutputDTO: 분류 Pipeline, 학습/테스트 지표, 예측, 분할 위치 및 어휘.
            feature_names는 원 특성이 아닌 학습 어휘 순서입니다. AUC는 이 실행기에서 생략합니다.
    Raises:
        ValueError: 입력/라벨/폴드/모델 조건 오류 또는 빈 어휘.

    Notes:
        튜닝 시 어휘 및 IDF도 각 학습 폴드에서 적합합니다. 테스트 문서는 튜닝에 사용하지 않습니다.
    """
    p = profile if profile is not None else TextProfile()
    docs, target = np.asarray(_documents(documents), dtype=object), np.asarray(y)
    if target.ndim != 1 or len(target) != len(docs) or type_of_target(target) not in ("binary", "multiclass") or len(np.unique(target)) < 2:
        raise ValueError("문서와 길이가 같은 최소 두 클래스의 1차원 라벨이 필요합니다.")
    if not 0 < p.test_size < 1 or not isinstance(p.cv, int) or p.cv < 2:
        raise ValueError("test_size 또는 cv를 확인하세요.")
    train, test = train_test_split(np.arange(len(docs)), test_size=p.test_size,
                                    stratify=target, random_state=p.random_state)
    pipeline = Pipeline([("vectorizer", _vectorizer(p)),
                          ("model", LogisticRegression(C=p.C, max_iter=p.max_iter, random_state=p.random_state))])
    params, cvscore, cvresults = {}, None, None
    if p.param_grid is not None:
        if not isinstance(p.param_grid, dict) or not p.param_grid or np.unique(target[train], return_counts=True)[1].min() < p.cv:
            raise ValueError("유효한 param_grid와 폴드별 클래스 표본 수가 필요합니다.")
        cv = StratifiedKFold(p.cv, shuffle=True, random_state=p.random_state)
        search = GridSearchCV(pipeline, p.param_grid, cv=cv, scoring="accuracy", n_jobs=1, error_score="raise")
        search.fit(docs[train], target[train])
        pipeline, params, cvscore, cvresults = search.best_estimator_, search.best_params_, float(search.best_score_), search.cv_results_
    else:
        pipeline.fit(docs[train], target[train])
    predictions = pipeline.predict(docs[test]); classes = pipeline.named_steps["model"].classes_
    return TextClassificationOutputDTO.from_result(SupervisedResult("TextLogisticRegression", "classification", pipeline,
             _metrics(target[train], pipeline.predict(docs[train]), "classification"),
             _metrics(target[test], predictions, "classification"), target[test], predictions,
             train, test, list(_names(pipeline.named_steps["vectorizer"])), classes=classes,
             probabilities=pipeline.predict_proba(docs[test]), confusion=confusion_matrix(target[test], predictions, labels=classes),
             best_params=params, best_cv_score=cvscore, cv_results=cvresults))


@dataclass
class TopicProfile:
    """LDA 토픽 조건.

    Attributes:
        n_topics: 토픽 수 (양수).
        top_n_words: 토픽당 반환할 단어 수 (양수).
        max_iter: LDA 반복 수.
        random_state: 시드.
    """
    n_topics: int = 2
    top_n_words: int = 10
    max_iter: int = 20
    random_state: int = 1234


@dataclass
class TopicOutputDTO(OutputDTO):
    """LDA 결과.

    Attributes:
        vectorizer: 학습 CountVectorizer.
        model: sklearn LatentDirichletAllocation. gensim 구현과 수치는 다를 수 있음.
        document_topics: 문서 x 토픽 확률 행렬. 각 행 합은 1.
        top_words: 토픽별 가중치 상위 단어 목록.
        feature_names: 입력 어휘 순서.
    """
    vectorizer: Any
    model: Any
    document_topics: np.ndarray
    top_words: List[List[str]]
    feature_names: np.ndarray


# 기존 Result 이름을 동일 클래스의 호환 별칭으로 유지합니다.
TopicResult = TopicOutputDTO


def run_topic_model(documents: Any, profile: Optional[TopicProfile] = None,
                    text_profile: Optional[TextProfile] = None) -> TopicOutputDTO:
    """정수 단어 빈도에 LDA를 적합합니다.

    Args:
        documents: 학습 문서 목록.
        profile: 토픽 수와 학습 설정.
        text_profile: Count 벡터화 설정. vectorizer='count'여야 합니다.
    Returns:
        TopicOutputDTO: 모델, 학습 문서 토픽 분포 및 상위 단어.
    Raises:
        ValueError: 잘못된 토픽 조건, TF-IDF 설정 또는 빈 어휘.
    """
    p = profile if profile is not None else TopicProfile()
    tp = text_profile if text_profile is not None else TextProfile(vectorizer="count")
    if tp.vectorizer != "count" or not isinstance(p.n_topics, int) or p.n_topics < 1 or not isinstance(p.top_n_words, int) or p.top_n_words < 1:
        raise ValueError("LDA에는 Count 벡터와 양의 정수 토픽/단어 수가 필요합니다.")
    vectors = run_vectorization(documents, tp)
    model = LatentDirichletAllocation(n_components=p.n_topics, max_iter=p.max_iter,
                  learning_method="batch", random_state=p.random_state)
    distribution = model.fit_transform(vectors.train_matrix)
    words = [[str(vectors.feature_names[i]) for i in component.argsort()[::-1][:p.top_n_words]] for component in model.components_]
    return TopicOutputDTO(vectors.vectorizer, model, distribution, words, vectors.feature_names)


def make_korean_tokenizer() -> Callable:
    """선택적 KoNLPy Okt 명사 토크나이저를 생성합니다.

    Args:
        없음.
    Returns:
        Callable: TextProfile(tokenizer=...)에 전달할 Okt.nouns.
    Raises:
        ImportError: konlpy 미설치. JVM 미설정은 KoNLPy의 환경 오류가 전달됩니다.
    """
    from konlpy.tag import Okt
    return Okt().nouns


def make_english_stemmer(stop_words: Optional[List[str]] = None) -> Callable:
    """코퍼스 다운로드 없이 사용할 정규식 단어 분리/Porter stemming 콜백을 생성합니다.

    Args:
        stop_words: 호출자가 제공할 불용어 목록. None이면 제거하지 않음.
    Returns:
        Callable: TextProfile(tokenizer=...)에 전달할 토크나이저.
    Raises:
        ImportError: nltk 미설치.

    Notes:
        PDF의 word_tokenize/POS/WordNet lemmatization과 다릅니다. 해당 기능은 별도 NLTK 자원이 필요합니다.
    """
    from nltk.stem import PorterStemmer
    stemmer, stop = PorterStemmer(), set(stop_words or [])
    def tokenize(text):
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        return [stemmer.stem(t) for t in tokens if t not in stop]
    return tokenize


@dataclass
class EnglishAnalysisOutputDTO(OutputDTO):
    """영어 토큰/문장/품사/표제어 출력 DTO.

    Attributes:
        mode: tokens/sentences/pos/lemma.
        items: 해당 mode의 문자열 또는 (토큰,품사) 목록, 원래 순서.
    """
    mode: str
    items: List[Any]


@dataclass
class TextExplanationOutputDTO(OutputDTO):
    """LIME 문서 설명 출력 DTO.

    Attributes:
        explanation: 원 LIME Explanation(runtime).
        classes: 모델의 실제 클래스 순서.
        probabilities: 설명할 문서의 클래스별 원 모델 예측 확률.
        contributions: 클래스 라벨 문자열별 (단어,국소 가중치) 목록.
        intercepts: 클래스 라벨 문자열별 국소 절편.
    """
    explanation: Any
    classes: List[Any]
    probabilities: np.ndarray
    contributions: Dict[str, List[Tuple[str, float]]]
    intercepts: Dict[str, float]


def analyze_english(text: str, mode: str = "tokens") -> EnglishAnalysisOutputDTO:
    """NLTK의 단어/문장/POS/표제어 분석을 선택적으로 실행합니다.

    Args:
        text: 영어 입력 문자열.
        mode: tokens, sentences, pos, lemma.
    Returns:
        EnglishAnalysisOutputDTO: mode와 토큰/문장/표제어/(토큰,품사) items 목록.
    Raises:
        ValueError: 잘못된 문자열 또는 mode.
        ImportError: nltk 미설치.
        LookupError: NLTK 토크나이저/POS/WordNet 자원 미설치. 자동 다운로드하지 않습니다.

    Notes:
        lemma는 PDF 예제처럼 기본 명사 표제어를 반환하며 문맥 기반 품사 추론은 하지 않습니다.
    """
    import nltk
    if not isinstance(text, str):
        raise ValueError("text는 문자열이어야 합니다.")
    if mode == "sentences":
        return EnglishAnalysisOutputDTO(mode,nltk.sent_tokenize(text))
    if mode not in ("tokens", "pos", "lemma"):
        raise ValueError("mode는 tokens, sentences, pos, lemma 중 하나여야 합니다.")
    tokens = nltk.word_tokenize(text)
    if mode == "tokens":
        return EnglishAnalysisOutputDTO(mode,tokens)
    if mode == "pos":
        return EnglishAnalysisOutputDTO(mode,nltk.pos_tag(tokens))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return EnglishAnalysisOutputDTO(mode,[lemmatizer.lemmatize(t) for t in tokens])


def explain_text(model: Any, text: str, num_features: int = 5,
                 random_state: int = 1234) -> TextExplanationOutputDTO:
    """적합 문서 분류 Pipeline의 LIME 설명을 생성합니다.

    Args:
        model: raw 문서 목록을 받는 predict_proba 지원 분류 Pipeline.
        text: 설명할 문서 한 개.
        num_features: 설명 토큰 수.
        random_state: perturbation 시드.
    Returns:
        TextExplanationOutputDTO: 클래스 순서/확률/단어 가중치/국소 절편과 원 LIME 객체.
    Raises:
        ImportError: lime 미설치.
        ValueError: 문자열/특성 수 오류 또는 predict_proba 미지원.

    Notes:
        국소 설명이며 인과적 기여를 의미하지 않습니다. 분류 클래스를 모두 설명합니다.
    """
    from lime.lime_text import LimeTextExplainer
    if not isinstance(text, str) or not isinstance(num_features, int) or num_features < 1 or not hasattr(model, "predict_proba"):
        raise ValueError("문서, num_features 및 분류 모델을 확인하세요.")
    classes = list(model.classes_)
    explainer = LimeTextExplainer(class_names=[str(c) for c in classes], random_state=random_state)
    explanation = explainer.explain_instance(text, model.predict_proba, num_features=num_features,
                                      labels=tuple(range(len(classes))))
    return TextExplanationOutputDTO(explanation,classes,np.asarray(explanation.predict_proba),
        {str(label):explanation.as_list(label=i) for i,label in enumerate(classes)},
        {str(label):float(explanation.intercept[i]) for i,label in enumerate(classes)})
