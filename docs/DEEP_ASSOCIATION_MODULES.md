# 딥러닝·연관규칙 기능과 API 명세

Dense·CNN·SimpleRNN/LSTM/GRU·시간 윈도 학습과 Apriori·FP-Growth·Eclat을 다룹니다. 공통 학습 계약은 뒤의 Dense·RNN 절에서, 초기 TF4종의 전체 필드는 [핵심 API 참조](reference/CORE_API.md)에서 확인합니다.
알고리즘마다 별도 구현 파일·전체 Profile·전용 Output DTO·JSON·UCI 스켈레톤을 제공합니다.
새 공통 파일은 학습/분할·거래 검증·결과 표/규칙 계산만 담당합니다.
CNN 아키텍처, SimpleRNN/LSTM/GRU 층 구성, FP-tree 탐색, TID 교집합 탐색은 각각 전용 파일에 있습니다.
시각화 import·그림 생성·저장은 추가하지 않습니다.

## 공개 API와 경로

| 알고리즘 | 구현 (`src/` 기준) | 공개 실행기 | 스켈레톤 (`src/` 기준) |
| --- | --- | --- | --- |
| CNN 1D·2D | DeepLearning/TensorFlow/cnn.py | run_tf_cnn | DeepLearning/skeleton/tf_cnn.py |
| SimpleRNN | DeepLearning/TensorFlow/simple_rnn.py | run_tf_simple_rnn | DeepLearning/skeleton/tf_simple_rnn.py |
| LSTM | DeepLearning/TensorFlow/lstm.py | run_tf_lstm | DeepLearning/skeleton/tf_lstm.py |
| GRU | DeepLearning/TensorFlow/gru.py | run_tf_gru | DeepLearning/skeleton/tf_gru.py |
| FP-Growth | Association/fp_growth.py | run_fp_growth | Association/skeleton/fp_growth.py |
| Eclat | Association/eclat.py | run_eclat | Association/skeleton/eclat.py |

연관규칙은 Apriori·FP-Growth·Eclat 3종입니다. 위 표는 전용 확장6개이며, Dense·일반 RNN·시간 윈도는 아래 공통 계약에서 설명합니다. 전체 기능 목록과 현재 수치는 [기능 카탈로그](ANALYSIS_CATALOG.md)를 따릅니다.

## 딥러닝 입력·학습 계약

- CNN dimension=1: `(표본,시간,채널)`. dimension=2: `(표본,높이,너비,채널)`.
  Conv → MaxPool 블록, Flatten, Dense, 출력 층 순서입니다. Conv는 same padding, Pool은 valid입니다.
  블록 뒤에 공간/시간 축이 사라지는 설정은 학습 전에 거부합니다.
- 전용 순환 모델: `(표본,시간,특성)`. 선택한 파일의 셀을 직접 생성합니다.
  `num_layers>1`이면 마지막 이전 층만 return_sequences=True입니다.
  `bidirectional=True`는 각 층을 Bidirectional로 감쌉니다.
  이 API는 고정 길이 독립 시퀀스의 sequence-to-one 분류/회귀입니다.
- 분류 y는 원 수치/문자 라벨 1차원, 회귀 y는 `(표본,)` 또는 `(표본,타깃)`입니다.
  scaler/LabelEncoder는 학습 표본에만 적합합니다. validation만 조기 종료에 쓰며 test는 최종 평가 전용입니다.
- test_data가 없으면 Profile의 test_size/validation_size는 전체 표본의 비율입니다.
  `test_data=(X_test,y_test)`를 제공하면 이 고정 보류 자료를 나누거나 학습에 사용하지 않습니다.
  이 경우 validation_size는 제공한 학습 풀 X 내부 비율이고 test_size는 분할에 사용하지 않습니다.
- 분할 위치는 원 X 뒤에 고정 X_test를 붙인 배열 기준입니다. test_data 미제공 시에는 원 X 기준입니다.
  parameters에는 Profile과 실제 `effective_test_fraction`, `effective_validation_fraction`이 포함됩니다.
- `model.predict(X_new)`는 학습 scaler·원 클래스 복원을 적용합니다.
  시계열을 자동 window로 바꾸지 않습니다. 중첩 시계열 창을 IID 입력으로 무작위 분리하면 누수가 생기므로
  연속 예측에는 기존 `time_series.py`의 시간 순 분할 계약을 사용합니다.
- `.keras_model`은 원 Keras 객체입니다. JSON은 가중치 저장/복원 형식이 아닙니다.
  가변 길이/padding/masking·stateful·custom cell은 새 API에도 포함하지 않습니다.

## 연관규칙 계약과 수학

FP-Growth는 항목 빈도 순 FP-tree와 조건부 패턴 기반 트리를 구성하여 패턴을 성장시킵니다.
Eclat은 항목별 거래 ID 집합을 만들고 교집합으로 확장합니다. Apriori 호출을 감싼 별칭이 아닙니다.
두 구현은 빈 거래도 분모에 포함하며, 같은 거래의 중복 항목은 한 번만 셉니다.
문자열 항목의 순서는 입력과 무관하게 재현됩니다.

`support(A)=count(A)/N`, `confidence(A→B)=support(A∪B)/support(A)`,
`lift=confidence/support(B)`, `leverage=support(A∪B)-support(A)support(B)`,
`conviction=(1-support(B))/(1-confidence)`입니다. confidence=1은 기존 계약처럼 infinity로 나타내며 JSON에서는 null입니다.
빈발 집합의 items 및 규칙의 antecedents/consequents는 정렬된 tuple입니다.
연관은 인과 관계를 뜻하지 않습니다.

결과는 itemsets/rules DataFrame과 알고리즘별 diagnostics입니다.
FP-tree의 조건부 트리까지 포함한 노드 한도, Eclat의 교집합 검사 한도,
빈발 집합·규칙 후보·규칙 출력 한도를 넘기면 오류를 내고 부분 결과를 반환하지 않습니다.
연관규칙 결과와 탐색 지표는 예측 모델의 보류 성능으로 해석하지 않습니다.

## UCI 스켈레톤

CNN 2D는 [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)의
64 정수 특성을 8×8×1로 복원합니다. 공식 `optdigits.tra` 3,823개에서 학습/검증을 나누고
`optdigits.tes` 1,797개를 test_data로 전달합니다. 원 자료의 수행자별 공식 train/test 구분을 유지합니다.
Alpaydin & Kaynak (1998), DOI [10.24432/C50P49](https://doi.org/10.24432/C50P49), CC BY 4.0.
이미지의 눈에 보이는 출력이나 학습 곡선은 생성하지 않습니다.

CNN 1D와 SimpleRNN/LSTM/GRU는 기존 [Libras Movement](https://archive.ics.uci.edu/dataset/181/libras+movement)의
독립 손동작 45프레임×2좌표를 사용합니다. 여러 표본이 같은 수행자에서 수집되었으므로 새 수행자에 대한 성능 검증은 아닙니다.
FP-Growth/Eclat은 기존 [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) 영수증 정제/상품 제한을 재사용합니다.
UCI 부분 자료의 min_support=0.05는 스켈레톤에 명시되며 원 모델별 JSON을 수정하지 않습니다.
상세 정제·캐시·라이선스는 [UCI 스켈레톤](UCI_SKELETONS.md)을 따릅니다.

```python
from src.DeepLearning.skeleton.tf_cnn import main
result = main(overrides={"epochs": 5, "conv_filters": [16, 32]})
print(result.split_mode, result.test_metrics)

from src.DeepLearning.skeleton.tf_lstm import main
result = main(overrides={"num_layers": 2, "bidirectional": True, "epochs": 5})

from src.Association.skeleton.fp_growth import main
result = main(overrides={"min_support": 0.05, "max_length": 3})
print(result.itemsets, result.rules)
```

## 전체 Profile·Output DTO 필드

### tf-cnn

레시피: `tf-cnn-classification.json`, `tf-cnn-regression.json`

**TFCNNProfile**

| 필드 | 타입 | 기본값 | 계약 |
| --- | --- | --- | --- |
| `random_state` | `<class 'int'>` | `1234` | 0 이상 정수 시드. CPU에서도 완전 결정성을 보장하지 않음. 기본 1234. |
| `epochs` | `<class 'int'>` | `30` | 최대 학습 epoch, 양수. 기본 30. |
| `batch_size` | `<class 'int'>` | `32` | 학습 batch 크기, 양수. 기본 32. |
| `learning_rate` | `<class 'float'>` | `0.001` | Adam 학습률, 양수. 기본 0.001. |
| `patience` | `<class 'int'>` | `5` | validation loss 조기 종료 대기, 0 이상. 기본 5. |
| `test_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 test 비율. 기본 0.2. |
| `validation_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 validation 비율, 두 비율 합 <1. 기본 0.2. |
| `scaling` | `<class 'str'>` | `'standard'` | none/standard; 학습 입력만 적합. 기본 standard. |
| `verbose` | `<class 'int'>` | `0` | Keras 출력 0/1/2. 기본 0. |
| `task` | `<class 'str'>` | `'classification'` | classification/regression. 기본 classification. |
| `dimension` | `<class 'int'>` | `2` | 1은 (표본,시간,채널), 2는 (표본,높이,너비,채널); 기본 2. |
| `conv_filters` | `typing.Tuple[int, ...]` | `(16, 32)` | 합성곱 블록의 양수 필터 수; 기본 (16,32). |
| `kernel_size` | `<class 'int'>` | `3` | same padding 합성곱 크기; 양수, 기본 3. |
| `pool_size` | `<class 'int'>` | `2` | 블록별 valid 최대풀링 크기; 양수, 기본 2. |
| `hidden_layer_sizes` | `typing.Tuple[int, ...]` | `(32,)` | Flatten 이후 Dense 폭; 기본 (32,). |
| `dropout` | `<class 'float'>` | `0.0` | 각 Dense 이후 dropout 비율 [0,1); 기본 0. |

**TFCNNOutputDTO**

| 필드 | 타입 | 기본값 | 계약 |
| --- | --- | --- | --- |
| `task` | `<class 'str'>` | `반환 필드` | classification/regression. 시계열은 regression. |
| `parameters` | `typing.Dict[str, typing.Any]` | `반환 필드` | 적용 Profile 및 effective_test_fraction/effective_validation_fraction. |
| `model` | `typing.Any` | `반환 필드` | 학습 scaler·class encoder를 포함한 predict adapter; keras_model로 원 Keras 모델 접근. JSON 제외. |
| `history` | `typing.Dict[str, typing.List[float]]` | `반환 필드` | epoch별 loss와 val_loss; Keras History 객체가 아닌 수치 목록. |
| `test_metrics` | `typing.Dict[str, float]` | `반환 필드` | 분류 accuracy/log_loss 또는 회귀 mse/rmse/mae. 원 타깃 단위. |
| `predictions` | `<class 'numpy.ndarray'>` | `반환 필드` | test 순서 예측. 분류는 원 라벨, 시계열은 (창,horizon,target). |
| `probabilities` | `typing.Optional[numpy.ndarray]` | `반환 필드` | 분류 (n,classes) 확률; 회귀 None. |
| `classes` | `typing.Optional[numpy.ndarray]` | `반환 필드` | 확률 열 순서의 학습 클래스; 회귀 None. |
| `train_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 학습 표본 번호; 시계열은 원 시점 번호. |
| `validation_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 조기 종료 전용 검증 표본/시점 번호. |
| `test_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 보류 평가 표본/시점 번호. |
| `target_indices` | `typing.Optional[numpy.ndarray]` | `반환 필드` | 시계열 test 각 창의 실제 타깃 시점 (창,horizon), 다른 모델 None. |
| `y_test` | `<class 'numpy.ndarray'>` | `반환 필드` | predictions와 같은 shape/순서의 실제 test 타깃. |
| `input_shape` | `typing.Tuple[int, ...]` | `반환 필드` | Keras 입력 shape(batch 제외). |
| `epochs_trained` | `<class 'int'>` | `반환 필드` | 실제 실행한 epoch 수. |
| `best_epoch` | `<class 'int'>` | `반환 필드` | 최저 val_loss epoch, 1부터 시작. 최적 weights 복원. |
| `split_mode` | `<class 'str'>` | `반환 필드` | random_holdout/provided_test; 고정 test에서는 test_size를 사용하지 않음. |

### tf-simple-rnn

레시피: `tf-simple-rnn-classification.json`, `tf-simple-rnn-regression.json`

**TFSimpleRNNProfile**

| 필드 | 타입 | 기본값 | 계약 |
| --- | --- | --- | --- |
| `random_state` | `<class 'int'>` | `1234` | 0 이상 정수 시드. CPU에서도 완전 결정성을 보장하지 않음. 기본 1234. |
| `epochs` | `<class 'int'>` | `30` | 최대 학습 epoch, 양수. 기본 30. |
| `batch_size` | `<class 'int'>` | `32` | 학습 batch 크기, 양수. 기본 32. |
| `learning_rate` | `<class 'float'>` | `0.001` | Adam 학습률, 양수. 기본 0.001. |
| `patience` | `<class 'int'>` | `5` | validation loss 조기 종료 대기, 0 이상. 기본 5. |
| `test_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 test 비율. 기본 0.2. |
| `validation_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 validation 비율, 두 비율 합 <1. 기본 0.2. |
| `scaling` | `<class 'str'>` | `'standard'` | none/standard; 학습 입력만 적합. 기본 standard. |
| `verbose` | `<class 'int'>` | `0` | Keras 출력 0/1/2. 기본 0. |
| `task` | `<class 'str'>` | `'classification'` | classification/regression. 기본 classification. |
| `units` | `<class 'int'>` | `16` | 순환 층 폭, 양수. 기본 16. |
| `num_layers` | `<class 'int'>` | `1` | 순환 층 수; 양수, 기본 1. 마지막 이전 층은 시퀀스를 반환. |
| `bidirectional` | `<class 'bool'>` | `False` | 각 순환 층의 양방향 처리; 기본 False. |
| `dropout` | `<class 'float'>` | `0.0` | 순환 층 입력 dropout [0,1); 기본 0. |
| `recurrent_dropout` | `<class 'float'>` | `0.0` | 순환 상태 dropout [0,1); 기본 0. |

**TFSimpleRNNOutputDTO**

| 필드 | 타입 | 기본값 | 계약 |
| --- | --- | --- | --- |
| `task` | `<class 'str'>` | `반환 필드` | classification/regression. 시계열은 regression. |
| `parameters` | `typing.Dict[str, typing.Any]` | `반환 필드` | 적용 Profile 및 effective_test_fraction/effective_validation_fraction. |
| `model` | `typing.Any` | `반환 필드` | 학습 scaler·class encoder를 포함한 predict adapter; keras_model로 원 Keras 모델 접근. JSON 제외. |
| `history` | `typing.Dict[str, typing.List[float]]` | `반환 필드` | epoch별 loss와 val_loss; Keras History 객체가 아닌 수치 목록. |
| `test_metrics` | `typing.Dict[str, float]` | `반환 필드` | 분류 accuracy/log_loss 또는 회귀 mse/rmse/mae. 원 타깃 단위. |
| `predictions` | `<class 'numpy.ndarray'>` | `반환 필드` | test 순서 예측. 분류는 원 라벨, 시계열은 (창,horizon,target). |
| `probabilities` | `typing.Optional[numpy.ndarray]` | `반환 필드` | 분류 (n,classes) 확률; 회귀 None. |
| `classes` | `typing.Optional[numpy.ndarray]` | `반환 필드` | 확률 열 순서의 학습 클래스; 회귀 None. |
| `train_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 학습 표본 번호; 시계열은 원 시점 번호. |
| `validation_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 조기 종료 전용 검증 표본/시점 번호. |
| `test_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 보류 평가 표본/시점 번호. |
| `target_indices` | `typing.Optional[numpy.ndarray]` | `반환 필드` | 시계열 test 각 창의 실제 타깃 시점 (창,horizon), 다른 모델 None. |
| `y_test` | `<class 'numpy.ndarray'>` | `반환 필드` | predictions와 같은 shape/순서의 실제 test 타깃. |
| `input_shape` | `typing.Tuple[int, ...]` | `반환 필드` | Keras 입력 shape(batch 제외). |
| `epochs_trained` | `<class 'int'>` | `반환 필드` | 실제 실행한 epoch 수. |
| `best_epoch` | `<class 'int'>` | `반환 필드` | 최저 val_loss epoch, 1부터 시작. 최적 weights 복원. |
| `split_mode` | `<class 'str'>` | `반환 필드` | random_holdout/provided_test; 고정 test에서는 test_size를 사용하지 않음. |

### tf-lstm

레시피: `tf-lstm-classification.json`, `tf-lstm-regression.json`

**TFLSTMProfile**

| 필드 | 타입 | 기본값 | 계약 |
| --- | --- | --- | --- |
| `random_state` | `<class 'int'>` | `1234` | 0 이상 정수 시드. CPU에서도 완전 결정성을 보장하지 않음. 기본 1234. |
| `epochs` | `<class 'int'>` | `30` | 최대 학습 epoch, 양수. 기본 30. |
| `batch_size` | `<class 'int'>` | `32` | 학습 batch 크기, 양수. 기본 32. |
| `learning_rate` | `<class 'float'>` | `0.001` | Adam 학습률, 양수. 기본 0.001. |
| `patience` | `<class 'int'>` | `5` | validation loss 조기 종료 대기, 0 이상. 기본 5. |
| `test_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 test 비율. 기본 0.2. |
| `validation_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 validation 비율, 두 비율 합 <1. 기본 0.2. |
| `scaling` | `<class 'str'>` | `'standard'` | none/standard; 학습 입력만 적합. 기본 standard. |
| `verbose` | `<class 'int'>` | `0` | Keras 출력 0/1/2. 기본 0. |
| `task` | `<class 'str'>` | `'classification'` | classification/regression. 기본 classification. |
| `units` | `<class 'int'>` | `16` | 순환 층 폭, 양수. 기본 16. |
| `num_layers` | `<class 'int'>` | `1` | 순환 층 수; 양수, 기본 1. 마지막 이전 층은 시퀀스를 반환. |
| `bidirectional` | `<class 'bool'>` | `False` | 각 순환 층의 양방향 처리; 기본 False. |
| `dropout` | `<class 'float'>` | `0.0` | 순환 층 입력 dropout [0,1); 기본 0. |
| `recurrent_dropout` | `<class 'float'>` | `0.0` | 순환 상태 dropout [0,1); 기본 0. |

**TFLSTMOutputDTO**

| 필드 | 타입 | 기본값 | 계약 |
| --- | --- | --- | --- |
| `task` | `<class 'str'>` | `반환 필드` | classification/regression. 시계열은 regression. |
| `parameters` | `typing.Dict[str, typing.Any]` | `반환 필드` | 적용 Profile 및 effective_test_fraction/effective_validation_fraction. |
| `model` | `typing.Any` | `반환 필드` | 학습 scaler·class encoder를 포함한 predict adapter; keras_model로 원 Keras 모델 접근. JSON 제외. |
| `history` | `typing.Dict[str, typing.List[float]]` | `반환 필드` | epoch별 loss와 val_loss; Keras History 객체가 아닌 수치 목록. |
| `test_metrics` | `typing.Dict[str, float]` | `반환 필드` | 분류 accuracy/log_loss 또는 회귀 mse/rmse/mae. 원 타깃 단위. |
| `predictions` | `<class 'numpy.ndarray'>` | `반환 필드` | test 순서 예측. 분류는 원 라벨, 시계열은 (창,horizon,target). |
| `probabilities` | `typing.Optional[numpy.ndarray]` | `반환 필드` | 분류 (n,classes) 확률; 회귀 None. |
| `classes` | `typing.Optional[numpy.ndarray]` | `반환 필드` | 확률 열 순서의 학습 클래스; 회귀 None. |
| `train_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 학습 표본 번호; 시계열은 원 시점 번호. |
| `validation_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 조기 종료 전용 검증 표본/시점 번호. |
| `test_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 보류 평가 표본/시점 번호. |
| `target_indices` | `typing.Optional[numpy.ndarray]` | `반환 필드` | 시계열 test 각 창의 실제 타깃 시점 (창,horizon), 다른 모델 None. |
| `y_test` | `<class 'numpy.ndarray'>` | `반환 필드` | predictions와 같은 shape/순서의 실제 test 타깃. |
| `input_shape` | `typing.Tuple[int, ...]` | `반환 필드` | Keras 입력 shape(batch 제외). |
| `epochs_trained` | `<class 'int'>` | `반환 필드` | 실제 실행한 epoch 수. |
| `best_epoch` | `<class 'int'>` | `반환 필드` | 최저 val_loss epoch, 1부터 시작. 최적 weights 복원. |
| `split_mode` | `<class 'str'>` | `반환 필드` | random_holdout/provided_test; 고정 test에서는 test_size를 사용하지 않음. |

### tf-gru

레시피: `tf-gru-classification.json`, `tf-gru-regression.json`

**TFGRUProfile**

| 필드 | 타입 | 기본값 | 계약 |
| --- | --- | --- | --- |
| `random_state` | `<class 'int'>` | `1234` | 0 이상 정수 시드. CPU에서도 완전 결정성을 보장하지 않음. 기본 1234. |
| `epochs` | `<class 'int'>` | `30` | 최대 학습 epoch, 양수. 기본 30. |
| `batch_size` | `<class 'int'>` | `32` | 학습 batch 크기, 양수. 기본 32. |
| `learning_rate` | `<class 'float'>` | `0.001` | Adam 학습률, 양수. 기본 0.001. |
| `patience` | `<class 'int'>` | `5` | validation loss 조기 종료 대기, 0 이상. 기본 5. |
| `test_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 test 비율. 기본 0.2. |
| `validation_size` | `<class 'float'>` | `0.2` | 전체 표본/시점 중 validation 비율, 두 비율 합 <1. 기본 0.2. |
| `scaling` | `<class 'str'>` | `'standard'` | none/standard; 학습 입력만 적합. 기본 standard. |
| `verbose` | `<class 'int'>` | `0` | Keras 출력 0/1/2. 기본 0. |
| `task` | `<class 'str'>` | `'classification'` | classification/regression. 기본 classification. |
| `units` | `<class 'int'>` | `16` | 순환 층 폭, 양수. 기본 16. |
| `num_layers` | `<class 'int'>` | `1` | 순환 층 수; 양수, 기본 1. 마지막 이전 층은 시퀀스를 반환. |
| `bidirectional` | `<class 'bool'>` | `False` | 각 순환 층의 양방향 처리; 기본 False. |
| `dropout` | `<class 'float'>` | `0.0` | 순환 층 입력 dropout [0,1); 기본 0. |
| `recurrent_dropout` | `<class 'float'>` | `0.0` | 순환 상태 dropout [0,1); 기본 0. |

**TFGRUOutputDTO**

| 필드 | 타입 | 기본값 | 계약 |
| --- | --- | --- | --- |
| `task` | `<class 'str'>` | `반환 필드` | classification/regression. 시계열은 regression. |
| `parameters` | `typing.Dict[str, typing.Any]` | `반환 필드` | 적용 Profile 및 effective_test_fraction/effective_validation_fraction. |
| `model` | `typing.Any` | `반환 필드` | 학습 scaler·class encoder를 포함한 predict adapter; keras_model로 원 Keras 모델 접근. JSON 제외. |
| `history` | `typing.Dict[str, typing.List[float]]` | `반환 필드` | epoch별 loss와 val_loss; Keras History 객체가 아닌 수치 목록. |
| `test_metrics` | `typing.Dict[str, float]` | `반환 필드` | 분류 accuracy/log_loss 또는 회귀 mse/rmse/mae. 원 타깃 단위. |
| `predictions` | `<class 'numpy.ndarray'>` | `반환 필드` | test 순서 예측. 분류는 원 라벨, 시계열은 (창,horizon,target). |
| `probabilities` | `typing.Optional[numpy.ndarray]` | `반환 필드` | 분류 (n,classes) 확률; 회귀 None. |
| `classes` | `typing.Optional[numpy.ndarray]` | `반환 필드` | 확률 열 순서의 학습 클래스; 회귀 None. |
| `train_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 학습 표본 번호; 시계열은 원 시점 번호. |
| `validation_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 조기 종료 전용 검증 표본/시점 번호. |
| `test_indices` | `<class 'numpy.ndarray'>` | `반환 필드` | 보류 평가 표본/시점 번호. |
| `target_indices` | `typing.Optional[numpy.ndarray]` | `반환 필드` | 시계열 test 각 창의 실제 타깃 시점 (창,horizon), 다른 모델 None. |
| `y_test` | `<class 'numpy.ndarray'>` | `반환 필드` | predictions와 같은 shape/순서의 실제 test 타깃. |
| `input_shape` | `typing.Tuple[int, ...]` | `반환 필드` | Keras 입력 shape(batch 제외). |
| `epochs_trained` | `<class 'int'>` | `반환 필드` | 실제 실행한 epoch 수. |
| `best_epoch` | `<class 'int'>` | `반환 필드` | 최저 val_loss epoch, 1부터 시작. 최적 weights 복원. |
| `split_mode` | `<class 'str'>` | `반환 필드` | random_holdout/provided_test; 고정 test에서는 test_size를 사용하지 않음. |

### fp-growth

레시피: `fp-growth.json`

**FPGrowthProfile**

| 필드 | 타입 | 기본값 | 계약 |
| --- | --- | --- | --- |
| `min_support` | `<class 'float'>` | `0.2` | 최소 지지도 (0,1]; 기본 0.2, 빈 거래 포함 분모. |
| `min_confidence` | `<class 'float'>` | `0.5` | 최소 신뢰도 [0,1]; 기본 0.5. |
| `min_lift` | `<class 'float'>` | `1.0` | 유한한 최소 lift, 0 이상; 기본 1. |
| `max_length` | `<class 'int'>` | `3` | 최대 패턴 길이, 양수; 기본 3. |
| `max_itemsets` | `<class 'int'>` | `100000` | 빈발 집합 출력 한도; 초과 시 오류, 기본 100000. |
| `max_rules` | `<class 'int'>` | `100000` | 규칙 출력 한도; 초과 시 오류, 기본 100000. |
| `max_rule_candidates` | `<class 'int'>` | `100000` | 전체 규칙 후보 검사 한도; 기본 100000. |
| `max_nodes` | `<class 'int'>` | `100000` | 조건부 트리 포함 생성 노드 한도; 기본 100000. |

**FPGrowthOutputDTO**

| 필드 | 타입 | 기본값 | 계약 |
| --- | --- | --- | --- |
| `algorithm` | `<class 'str'>` | `반환 필드` | 실행 알고리즘 이름. |
| `n_transactions` | `<class 'int'>` | `반환 필드` | 빈 거래 포함 분모. |
| `itemsets` | `<class 'pandas.core.frame.DataFrame'>` | `반환 필드` | items(tuple), count, support, length; 길이/항목 사전 순서. |
| `rules` | `<class 'pandas.core.frame.DataFrame'>` | `반환 필드` | antecedents, consequents, support, confidence, lift, leverage, conviction. |
| `diagnostics` | `typing.Dict[str, int]` | `반환 필드` | 빈발 집합·규칙 후보 및 tree_nodes 수. |

### eclat

레시피: `eclat.json`

**EclatProfile**

| 필드 | 타입 | 기본값 | 계약 |
| --- | --- | --- | --- |
| `min_support` | `<class 'float'>` | `0.2` | 최소 지지도 (0,1]; 기본 0.2, 빈 거래 포함 분모. |
| `min_confidence` | `<class 'float'>` | `0.5` | 최소 신뢰도 [0,1]; 기본 0.5. |
| `min_lift` | `<class 'float'>` | `1.0` | 유한한 최소 lift, 0 이상; 기본 1. |
| `max_length` | `<class 'int'>` | `3` | 최대 패턴 길이, 양수; 기본 3. |
| `max_itemsets` | `<class 'int'>` | `100000` | 빈발 집합 출력 한도; 초과 시 오류, 기본 100000. |
| `max_rules` | `<class 'int'>` | `100000` | 규칙 출력 한도; 초과 시 오류, 기본 100000. |
| `max_rule_candidates` | `<class 'int'>` | `100000` | 전체 규칙 후보 검사 한도; 기본 100000. |
| `max_intersections` | `<class 'int'>` | `100000` | TID 교집합 검사 한도; 기본 100000. |

**EclatOutputDTO**

| 필드 | 타입 | 기본값 | 계약 |
| --- | --- | --- | --- |
| `algorithm` | `<class 'str'>` | `반환 필드` | 실행 알고리즘 이름. |
| `n_transactions` | `<class 'int'>` | `반환 필드` | 빈 거래 포함 분모. |
| `itemsets` | `<class 'pandas.core.frame.DataFrame'>` | `반환 필드` | items(tuple), count, support, length; 길이/항목 사전 순서. |
| `rules` | `<class 'pandas.core.frame.DataFrame'>` | `반환 필드` | antecedents, consequents, support, confidence, lift, leverage, conviction. |
| `diagnostics` | `typing.Dict[str, int]` | `반환 필드` | 빈발 집합·규칙 후보 및 intersections 수. |

## 검증과 공식 근거

- FP-Growth/Eclat: 무작위 거래 50개 사례의 독립 전수 계산·Apriori 일치, 중복/빈 거래/부동소수점 경계,
  confidence/lift/leverage/conviction, 입력 순서, 탐색 한도와 완전한 결과 반환 계약.
- 실제 Retail 부분 자료에서 두 알고리즘 모두 빈발 집합 167개·규칙 80개이며 항목과 수치가 일치합니다.
- 실제 UCI CNN 1D/2D 및 세 전용 순환 스켈레톤을 2 epoch 학습해 예측·DTO 직렬화를 확인했습니다.
- 별도 TF 테스트는 Conv1D/Conv2D 층, 셀 종류, 다층 return_sequences, Bidirectional, dropout,
  다중 회귀 타깃, 원 라벨 복원, 학습 전용 scaler 및 고정 test 위치를 확인합니다.
- 모든 새 모듈에 7개 섹션·전체 Profile/DTO 필드·Args/Returns/Raises와 Python 3.7 구문 검사를 적용합니다.
  실제 시험 서버·GPU 성능을 검증한 기록은 아닙니다. 2 epoch 실행은 예측 품질 평가가 아닙니다.

근거: [FP-growth 원 논문](https://sigmod.org/publications/discs/2001/out/p_miningfrequentpjijiy.htm),
[Eclat 원 논문](https://www.cs.rpi.edu/~zaki/PaperDir/TKDE00.pdf),
[TensorFlow Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D),
[SimpleRNN](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN),
[LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM).

## 분야별 스켈레톤 이관

이번 PR에서 기존 12개 타 분야 스켈레톤도 각 구현 분야로 이관했습니다.
MachineLearning에는 실제 해당 분야 모델의 21개만 남겼습니다.
DeepLearning 8개·Association 3개·Networks 1개는 각각 자신의 skeleton 패키지에 있습니다.
공유 데이터 로더는 src/datasets, 기존 확률분포 예제는 src/stats/skeleton/distributions.py에 있습니다.
도메인별 파일 수·금지된 교차 배치·통계 호환 export·새 import/CLI를 검사합니다.

## Dense·RNN·시간 윈도 공통 학습 계약

첨부한 `41_Tensorflow_classification.ipynb`, `41_Tensorflow_regression.ipynb`,
`41_Tensorflow_rnn.ipynb`, `41_Tensorflow_time_series.ipynb`를 참고해 ADP 분석에 재사용할
내용을 선별했습니다. 데이터 다운로드, 대량 그림, 고급 custom cell 예제는 옮기지 않고
프로젝트의 Profile·Output DTO·7개 섹션·모델별 JSON 레시피 형식으로 새로 작성했습니다.

## 자료 → 프로젝트 모듈

| 참고 자료 | 정리한 내용 | 실행기 |
| --- | --- | --- |
| classification | 수치 특성/평탄화 이미지, Dense, softmax, 정수 인코딩, sparse cross entropy, 원 클래스 확률 | `run_tf_classification(X, y)` |
| regression | 특성/타깃 분리, 학습 전용 정규화, 선형 또는 다층 Dense, MSE 학습, MAE/RMSE 평가 | `run_tf_regression(X, y)` |
| rnn | batch/time/features, SimpleRNN/LSTM/GRU, 마지막 상태 기반 sequence-to-one, 분류/회귀 | `run_tf_rnn(X, y, task=...)` |
| time_series | 시간 순 분할, 입력/미래 타깃 윈도, 단일/다중 스텝·단일/다중 타깃, recurrent single-shot | `run_tf_time_series(series)` |

## 설치와 버전

TensorFlow는 선택 의존성이며 모듈 import/Profile/윈도 검증에는 필요하지 않습니다.
실제 학습 호출 때만 import하며 미설치 시 전용 설치 파일을 안내합니다.
다음 두 환경은 기존 모델 환경과 **별도 가상환경**에 설치합니다. TF 2.16의 NumPy 1.x 제약과
기존 `requirements-models-py312.txt`의 NumPy 2.x 핀을 섞지 않습니다.

```bash
# Python 3.7: CPU TensorFlow 2.11
python -m pip install -r requirements-tensorflow-py37.txt
# Python 3.12: CPU TensorFlow 2.16.2
python -m pip install -r requirements-tensorflow-py312.txt
```

Python 3.7.4 자체와 실제 시험 서버의 설치 가능 여부는 검증 대상과 다릅니다.
CPU 휠 기준이며 Apple Silicon 네이티브 설치 안내는 제공하지 않습니다.
TensorFlow 환경은 통계/트리/군집 전체 패키지를 설치하지 않습니다.

## 공통 흐름과 DTO

1. 결측·범주 인코딩·시간 정렬은 실행 전에 처리합니다. Dense X=(표본,특성), RNN X=(표본,시간,특성).
2. Dense와 독립 시퀀스 RNN은 IID 무작위 분할입니다. classification은 두 단계 모두 층화합니다. test_size/validation_size는 **전체 표본 비율**입니다.
3. StandardScaler는 학습 입력에서만 적합합니다. 3차원 입력은 학습 표본×시간 축을 합쳐 특성별 정규화합니다. 중첩 시계열 윈도에서는 시점이 중복 가중됩니다.
4. Adam과 조기 종료를 사용합니다. validation loss를 모니터링하고 최적 가중치를 복원합니다. test는 최종 평가에만 사용합니다.
5. history에는 실제 epoch별 loss/val_loss, best_epoch은 최저 val_loss의 1-based 위치입니다.
6. DTO는 모델 adapter, 원 라벨/단위 예측, 확률·클래스, 분할 위치, 실제 y_test와 성능을 반환합니다. `to_json()`은 모델을 제외합니다.

Dense 분류는 이진/다중 클래스 모두 softmax입니다. 확률 열은 `result.classes` 순서입니다.
회귀 출력은 무활성 Dense이며 y는 (표본,) 또는 (표본,타깃 수)입니다.
입력 스케일만 변환하며 타깃은 변환하지 않아 MSE/MAE/RMSE가 원 타깃 단위를 사용합니다.
학습/예측이 비유한 값이면 오류입니다. 회귀 타깃의 규모에 따라 학습률을 조정해야 합니다.

`result.model.predict(X_new)`는 학습 scaler와 클래스 라벨 복원을 적용합니다.
`result.model.keras_model`은 원 Keras 모델이므로 **정규화된 입력**을 받아야 합니다.
JSON은 가중치 저장/복원 형식이 아닙니다. `.keras` 저장은 원 Keras 모델만 저장하므로
scaler/encoder/열 순서도 별도로 보관해야 합니다.

## Dense 분류·회귀

```python
import numpy as np
from src.DeepLearning.TensorFlow.classification import run_tf_classification
from src.DeepLearning.TensorFlow.regression import run_tf_regression

X = np.random.RandomState(1234).normal(size=(120, 4))
y = np.where(X[:, 0] > 0, "positive", "negative")
classification = run_tf_classification(X, y, overrides={"epochs": 10})
print(classification.test_metrics, classification.classes)
print(classification.model.predict_proba(X[:3]))
regression = run_tf_regression(X, X[:, 0]*2-X[:, 1],
    overrides={"hidden_layer_sizes": [32, 16], "epochs": 10})
print(regression.test_metrics, regression.model.predict(X[:3]))
# 선형 회귀 신경망: hidden_layer_sizes=[]
```

Fashion MNIST처럼 이미지를 입력하려면 `images.reshape(len(images), -1)`로 평탄화합니다.
이 모듈은 CNN이 아닙니다. 이미 [0,1] 스케일의 입력을 만들었다면 `scaling="none"`을 선택할 수 있습니다.
Auto MPG처럼 범주가 있는 표는 원핫 인코딩·결측 처리 후 수치 배열을 입력합니다.

## RNN 입력과 상태

```python
from src.DeepLearning.TensorFlow.rnn import run_tf_rnn
sequences = np.random.RandomState(9).normal(size=(100, 8, 3))
result = run_tf_rnn(sequences, sequences[:, :, 0].sum(axis=1),
    task="regression", overrides={"cell": "gru", "units": 16, "epochs": 10})
print(result.model.predict(sequences[:2]))
```

`return_sequences=False`는 마지막 상태 (batch,units), True는 매 시점 (batch,time,units)입니다.
현재 실행기는 마지막 상태 기반 sequence-to-one입니다. 쌓은 RNN에서 다음 순환 층에
넘길 때는 앞 층의 return_sequences=True가 필요합니다. 상태 반환(return_state),
stateful·bidirectional·padding/masking·custom cell은 이 실행기 범위에 포함하지 않습니다.

RNN 입력의 표본은 **서로 독립한 시퀀스**여야 합니다. 하나의 연속 시계열에서 겹치는 윈도를
만들어 IID RNN 실행기에 넣으면 누수 가능성이 있습니다. 해당 경우 아래 시계열 실행기를 사용합니다.
stateful RNN은 batch 위치가 이전 batch의 동일 시퀀스에 대응하고 상태 초기화가 필요하며,
예측 시 미래 문맥을 볼 수 없는 상황에는 양방향 모델을 그대로 쓰지 않습니다.

## 시간 순 예측

```python
from src.DeepLearning.TensorFlow.time_series import run_tf_time_series
series = np.stack([np.sin(np.arange(300)/8), np.cos(np.arange(300)/10)], axis=1)
forecast = run_tf_time_series(series,
    overrides={"input_width": 12, "horizon": 3, "target_columns": [0], "epochs": 10})
print(forecast.predictions.shape)  # (test 윈도 수, 3, 1)
print(forecast.target_indices)     # 각 예측에 대응하는 원 시점
```

원 시점을 train → validation → test로 먼저 분리하고 각 구간 내부에서만 윈도를 만듭니다.
입력은 `[i, i+input_width)`, 타깃은 바로 뒤의 `[i+input_width, i+input_width+horizon)`입니다.
각 구간에 input_width+horizon 이상 시점이 필요합니다. 분할 경계 이전의 입력 문맥은
사용하지 않으므로 경계의 초기 타깃들은 평가하지 않습니다. shift/gap 옵션은 없습니다.

train_indices/validation_indices/test_indices는 원 시점 번호이며, predictions는 test 윈도 순서입니다.
y_test/target_indices가 예측과 직접 대응합니다. 여러 윈도가 동일 미래 시점을 예측할 수 있어
MSE/MAE는 윈도×horizon×타깃 전체를 평균합니다. 미래 여러 스텝을 한 번에 예측하는
single-shot 방식이며 예측을 다시 입력하는 autoregressive 방식은 구현하지 않습니다.
`model.predict`는 **원 단위의 이미 구성한 입력 창** (batch,input_width,특성)을 받습니다.

자료의 주기 특성 sin/cos, 이상값 처리와 예측 기준선(persistence), rolling-origin backtest는
문제에 맞춰 준비해야 합니다. 현재는 신경망의 보류 평가만 제공하고 기준선 우월성이나
실제 공정 제어 최적화를 주장하지 않습니다.

## 참고 자료

네 첨부 자료는 TensorFlow Authors 튜토리얼입니다. 다음 공식 자료에서 개념을 확인할 수 있습니다.

- [기초 분류](https://www.tensorflow.org/tutorials/keras/classification)
- [회귀](https://www.tensorflow.org/tutorials/keras/regression)
- [Keras RNN](https://www.tensorflow.org/guide/keras/working_with_rnns)
- [시계열](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [버전별 Python 지원](https://www.tensorflow.org/install/source#tested_build_configurations)
