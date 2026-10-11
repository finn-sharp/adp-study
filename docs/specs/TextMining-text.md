# `src/TextMining/text.py`

[소스 코드](../../src/TextMining/text.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

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

## `TextClassificationOutputDTO`

TextClassification 모듈 전용 반환 DTO.

Attributes:
    공통 SupervisedOutputDTO(Result 호환 별칭)의 모든 데이터·모델·지표 필드를 상속합니다.
    to_dict/to_json은 저장 가능한 데이터 필드만 기본으로 반환합니다.

## `clean_text`

HTML/URL을 제거하고 공백을 정리합니다.

Args:
    text: 입력 문장.
Returns:
    str: 한글·영문·숫자·문장부호를 보존한 정제 문자열.
Raises:
    ValueError: 문자열이 아닌 입력.

## `TextProfile`

텍스트 벡터화·분류 조건.

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

## `VectorizationOutputDTO`

문서 벡터화 결과.

Attributes:
    vectorizer: 학습 문서에만 적합한 벡터화 객체.
    train_matrix: 학습 문서 희소 행렬.
    test_matrix: 선택적 테스트 문서 희소 행렬. 없으면 None.
    feature_names: 희소 행렬 열 순서의 토큰 이름.

## `run_vectorization`

학습 문서에서 어휘/IDF를 적합하고 테스트 문서는 변환만 합니다.

Args:
    train_documents: 학습 문서 목록.
    profile: 벡터화 설정.
    test_documents: 선택적 테스트 문서 목록. 새 토큰은 학습 어휘에 추가되지 않음.
Returns:
    VectorizationOutputDTO: 적합 벡터화 객체, 희소 행렬 및 토큰 목록.
Raises:
    ValueError: 잘못된 입력/설정 또는 빈 어휘.

## `run_text_classification`

층화 분할 후 벡터화 → 로지스틱 분류 Pipeline을 적합합니다.

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

## `TopicProfile`

LDA 토픽 조건.

Attributes:
    n_topics: 토픽 수 (양수).
    top_n_words: 토픽당 반환할 단어 수 (양수).
    max_iter: LDA 반복 수.
    random_state: 시드.

## `TopicOutputDTO`

LDA 결과.

Attributes:
    vectorizer: 학습 CountVectorizer.
    model: sklearn LatentDirichletAllocation. gensim 구현과 수치는 다를 수 있음.
    document_topics: 문서 x 토픽 확률 행렬. 각 행 합은 1.
    top_words: 토픽별 가중치 상위 단어 목록.
    feature_names: 입력 어휘 순서.

## `run_topic_model`

정수 단어 빈도에 LDA를 적합합니다.

Args:
    documents: 학습 문서 목록.
    profile: 토픽 수와 학습 설정.
    text_profile: Count 벡터화 설정. vectorizer='count'여야 합니다.
Returns:
    TopicOutputDTO: 모델, 학습 문서 토픽 분포 및 상위 단어.
Raises:
    ValueError: 잘못된 토픽 조건, TF-IDF 설정 또는 빈 어휘.

## `make_korean_tokenizer`

선택적 KoNLPy Okt 명사 토크나이저를 생성합니다.

Args:
    없음.
Returns:
    Callable: TextProfile(tokenizer=...)에 전달할 Okt.nouns.
Raises:
    ImportError: konlpy 미설치. JVM 미설정은 KoNLPy의 환경 오류가 전달됩니다.

## `make_english_stemmer`

코퍼스 다운로드 없이 사용할 정규식 단어 분리/Porter stemming 콜백을 생성합니다.

Args:
    stop_words: 호출자가 제공할 불용어 목록. None이면 제거하지 않음.
Returns:
    Callable: TextProfile(tokenizer=...)에 전달할 토크나이저.
Raises:
    ImportError: nltk 미설치.

Notes:
    PDF의 word_tokenize/POS/WordNet lemmatization과 다릅니다. 해당 기능은 별도 NLTK 자원이 필요합니다.

## `EnglishAnalysisOutputDTO`

영어 토큰/문장/품사/표제어 출력 DTO.

Attributes:
    mode: tokens/sentences/pos/lemma.
    items: 해당 mode의 문자열 또는 (토큰,품사) 목록, 원래 순서.

## `TextExplanationOutputDTO`

LIME 문서 설명 출력 DTO.

Attributes:
    explanation: 원 LIME Explanation(runtime).
    classes: 모델의 실제 클래스 순서.
    probabilities: 설명할 문서의 클래스별 원 모델 예측 확률.
    contributions: 클래스 라벨 문자열별 (단어,국소 가중치) 목록.
    intercepts: 클래스 라벨 문자열별 국소 절편.

## `analyze_english`

NLTK의 단어/문장/POS/표제어 분석을 선택적으로 실행합니다.

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

## `explain_text`

적합 문서 분류 Pipeline의 LIME 설명을 생성합니다.

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
