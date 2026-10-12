# TensorFlow 핵심 정리와 실행 계약

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
