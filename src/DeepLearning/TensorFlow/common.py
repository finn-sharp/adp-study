"""
================================================================================
common.py - TensorFlow 자료 분할·학습 공통 구현
================================================================================
설명:
    - Profile → 모델별 JSON 레시피 → 실행기 → Output DTO 계약을 제공합니다.
분석 가정:
    - IID 독립 시퀀스 또는 정렬된 연속 시계열; test는 최종 평가 전용.
작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    입력 검증·전용 DTO·레시피·예제 추가
================================================================================
"""
__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"
from dataclasses import asdict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, mean_absolute_error
from src.MachineLearning.analysis_common import positive_int, finite_number


def validate_profile(p, kind):
    """Profile의 수치 범위·아키텍처 조건을 검증합니다.

    Args:
        p: 모듈별 학습 Profile.
        kind: classification/regression/rnn/time_series.
    Returns:
        None: 설정이 유효하면 반환.
    Raises:
        ValueError/TypeError: 입력 shape·타입·조건 위반.
    """
    for name in ('random_state', 'patience'):
        positive_int(getattr(p, name), name, 0)
    for name in ('epochs', 'batch_size'):
        positive_int(getattr(p, name), name)
    finite_number(p.learning_rate, 'learning_rate', True)
    for name in ('test_size', 'validation_size'):
        value = getattr(p, name)
        finite_number(value, name, True)
        if value >= 1: raise ValueError(name + '는 1 미만이어야 합니다.')
    if p.test_size+p.validation_size >= 1: raise ValueError('학습 자료 비율이 양수여야 합니다.')
    if p.scaling not in ('none', 'standard'): raise ValueError('scaling은 none/standard입니다.')
    if isinstance(p.verbose, bool) or p.verbose not in (0, 1, 2): raise ValueError('verbose는 0/1/2입니다.')
    if kind in ('classification', 'regression'):
        if not isinstance(p.hidden_layer_sizes, tuple): raise ValueError('hidden_layer_sizes는 tuple입니다.')
        for width in p.hidden_layer_sizes: positive_int(width, 'hidden_layer_sizes')
    else:
        if p.cell not in ('simple_rnn', 'lstm', 'gru'): raise ValueError('cell 오류')
        positive_int(p.units, 'units')
    if kind == 'rnn' and p.task not in ('classification', 'regression'): raise ValueError('task 오류')
    if kind == 'time_series':
        positive_int(p.input_width, 'input_width'); positive_int(p.horizon, 'horizon')
        if p.target_columns is not None:
            if not isinstance(p.target_columns, tuple) or not p.target_columns: raise ValueError('target_columns는 비어있지 않은 tuple입니다.')
            for c in p.target_columns: positive_int(c, 'target_columns', 0)
            if len(set(p.target_columns)) != len(p.target_columns): raise ValueError('target_columns 중복')


def numeric_X(X, ndim):
    """유한한 수치 배열과 고정 rank를 검증합니다.

    Args:
        X: 수치 특성: 2차원 Dense 또는 3차원 시퀀스.
        ndim: 요구하는 입력 rank.
    Returns:
        np.ndarray: float32 수치 입력.
    Raises:
        ValueError/TypeError: 입력 shape·타입·조건 위반.
    """
    a = np.asarray(X, dtype=np.float32)
    if a.ndim != ndim or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError('입력 rank/비어 있는 축/결측을 확인하세요.')
    return a


class PredictionAdapter:
    """원 수치 입력에 학습 scaler를 적용하고 분류 라벨을 복원합니다.

    시계열 predict 입력은 이미 구성한 (창,input_width,특성)입니다. 원 시계열을
    자동 windowing하지 않습니다. keras_model은 scaler를 포함하지 않습니다.
    """
    def __init__(self, model, scaler, shape, encoder=None, scalar_output=False):
        """

        Args:
            model: 적합 Keras 모델.
            scaler: 학습 자료에 적합한 StandardScaler 또는 None.
            shape: batch를 제외한 학습 입력 shape.
            encoder: 학습 클래스 인코더 또는 None.
            scalar_output: 회귀 타깃을 1차원으로 복원할지 여부.
        Returns:
            None: scaler·모델·encoder 상태 보존.
        Raises:
            ValueError/TypeError: 입력 shape·타입·조건 위반.
        """
        self.keras_model = model
        self.scaler = scaler
        self.input_shape = shape
        self.encoder = encoder
        self.scalar_output = scalar_output

    def _input(self, X):
        """

        Args:
            X: 수치 특성: 2차원 Dense 또는 3차원 시퀀스.
        Returns:
            np.ndarray: 학습 기준으로 정규화한 입력.
        Raises:
            ValueError/TypeError: 입력 shape·타입·조건 위반.
        """
        a = numeric_X(X, len(self.input_shape)+1)
        if a.shape[1:] != self.input_shape: raise ValueError('학습과 같은 입력 shape가 필요합니다.')
        if self.scaler is not None:
            a = self.scaler.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape)
        return a.astype(np.float32)

    def _raw(self, X):
        """

        Args:
            X: 수치 특성: 2차원 Dense 또는 3차원 시퀀스.
        Returns:
            np.ndarray: 원 모델 출력.
        Raises:
            ValueError/TypeError: 입력 shape·타입·조건 위반.
        """

        return self.keras_model(self._input(X), training=False).numpy()

    def predict(self, X):
        """원 단위 회귀 예측 또는 원 분류 라벨을 반환합니다.

        Args:
            X: 수치 특성: 2차원 Dense 또는 3차원 시퀀스.
        Returns:
            np.ndarray: 원 클래스 라벨 또는 원 단위 회귀 예측.
        Raises:
            ValueError/TypeError: 입력 shape·타입·조건 위반.
        """
        a = self._raw(X)
        if self.encoder is not None: return self.encoder.inverse_transform(a.argmax(axis=1))
        return a[:, 0] if self.scalar_output else a

    def predict_proba(self, X):
        """학습 클래스 순서의 확률. 회귀에서는 오류입니다.

        Args:
            X: 수치 특성: 2차원 Dense 또는 3차원 시퀀스.
        Returns:
            np.ndarray: 학습 클래스 순서 확률.
        Raises:
            ValueError/TypeError: 입력 shape·타입·조건 위반.
        """
        if self.encoder is None: raise ValueError('회귀는 predict_proba를 지원하지 않습니다.')
        return self._raw(X)


def _tensorflow(seed):
    """선택 의존성을 읽고 Python/NumPy/TF 시드를 설정합니다.

    Args:
        seed: 0 이상 난수 시드.
    Returns:
        Any: TensorFlow 모듈.
    Raises:
        ValueError/TypeError: 입력 shape·타입·조건 위반.
        ImportError: TensorFlow 미설치.
    """
    import random
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError('TensorFlow 전용 requirements-tensorflow-py37/py312.txt를 설치하세요.') from exc
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    return tf


def _dataset(tf, X, y, batch_size):
    """소규모 CPU 예제에서도 thread 폭증을 피하는 batch Dataset.

    Args:
        tf: TensorFlow 모듈.
        X: 수치 특성: 2차원 Dense 또는 3차원 시퀀스.
        y: 분류 라벨 또는 원 단위 수치 타깃.
        batch_size: 양수 batch 크기.
    Returns:
        Any: 제한된 private threadpool의 batch Dataset.
    Raises:
        ValueError/TypeError: 입력 shape·타입·조건 위반.
    """
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y)).batch(batch_size)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return ds.with_options(options)


def _fit(X, y, train, val, test, p, kind, classification, target_indices=None):
    """학습만 정규화·라벨 적합; validation만 조기 종료; test는 최종 평가.

    Args:
        X: 수치 특성: 2차원 Dense 또는 3차원 시퀀스.
        y: 분류 라벨 또는 원 단위 수치 타깃.
        train: 학습 표본 인덱스.
        val: 검증 표본 인덱스.
        test: 보류 표본 인덱스.
        p: 모듈별 학습 Profile.
        kind: classification/regression/rnn/time_series.
        classification: 분류 여부.
        target_indices: 시계열 보류 타깃의 원 시점 배열 또는 None.
    Returns:
        dict: DTO 필드에 대응하는 학습·보류 결과.
    Raises:
        ValueError/TypeError: 입력 shape·타입·조건 위반.
        ImportError: TensorFlow 미설치.
    """
    scaler = StandardScaler().fit(X[train].reshape(-1, X.shape[-1])) if p.scaling == 'standard' else None
    scaled = scaler.transform(X.reshape(-1,X.shape[-1])).reshape(X.shape).astype(np.float32) if scaler is not None else X
    encoder = None
    scalar = not classification and y.ndim == 1
    if classification:
        encoder = LabelEncoder().fit(y[train])
        if len(encoder.classes_) < 2: raise ValueError('학습에 2개 이상 클래스가 필요합니다.')
        target = encoder.transform(y)
        output_shape = (len(encoder.classes_),)
    else:
        target = y[:, None] if scalar else y
        output_shape = target.shape[1:]
    tf = _tensorflow(p.random_state)
    layers = [tf.keras.layers.Input(shape=X.shape[1:])]
    if kind in ('rnn', 'time_series'):
        Cell = {'simple_rnn':tf.keras.layers.SimpleRNN, 'lstm':tf.keras.layers.LSTM, 'gru':tf.keras.layers.GRU}[p.cell]
        layers.append(Cell(p.units))
    else:
        for width in p.hidden_layer_sizes: layers.append(tf.keras.layers.Dense(width, activation='relu'))
    layers.append(tf.keras.layers.Dense(int(np.prod(output_shape)), activation='softmax' if classification else None))
    if len(output_shape) > 1: layers.append(tf.keras.layers.Reshape(output_shape))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p.learning_rate), loss='sparse_categorical_crossentropy' if classification else 'mse')
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=p.patience, restore_best_weights=True)
    history = model.fit(_dataset(tf,scaled[train],target[train],p.batch_size),
                        validation_data=_dataset(tf,scaled[val],target[val],p.batch_size),
                        epochs=p.epochs, callbacks=[callback], verbose=p.verbose, shuffle=False).history
    # TF 2.11 restores only on early stop; also restore after a full epoch budget.
    if callback.best_weights is not None:
        model.set_weights(callback.best_weights)
    if not all(np.isfinite(v).all() for v in history.values()): raise ValueError('비유한 학습 loss입니다. 스케일/학습률을 확인하세요.')
    adapter = PredictionAdapter(model,scaler,X.shape[1:],encoder,scalar)
    predictions = adapter.predict(X[test])
    if not np.isfinite(model(scaled[test],training=False).numpy()).all(): raise ValueError('비유한 예측')
    proba = adapter.predict_proba(X[test]) if classification else None
    metrics = {'accuracy':float(accuracy_score(y[test],predictions)), 'log_loss':float(log_loss(y[test],proba,labels=encoder.classes_))} if classification else {'mse':float(mean_squared_error(y[test].reshape(len(test),-1),predictions.reshape(len(test),-1))), 'mae':float(mean_absolute_error(y[test].reshape(len(test),-1),predictions.reshape(len(test),-1)))}
    if not classification: metrics['rmse'] = float(np.sqrt(metrics['mse']))
    return dict(task='classification' if classification else 'regression', parameters=asdict(p), model=adapter, history={k:[float(v) for v in vs] for k,vs in history.items()}, test_metrics=metrics,
                predictions=predictions, probabilities=proba, classes=None if encoder is None else encoder.classes_,
                train_indices=train,validation_indices=val,test_indices=test,target_indices=target_indices,
                y_test=y[test],input_shape=tuple(X.shape[1:]),epochs_trained=len(history['loss']),best_epoch=int(np.argmin(history['val_loss']))+1)


def run_supervised(X, y, p, kind):
    """IID Dense / 독립 시퀀스 RNN 표본을 학습·검증·평가 분할합니다.

    Args:
        X: 수치 특성: 2차원 Dense 또는 3차원 시퀀스.
        y: 분류 라벨 또는 원 단위 수치 타깃.
        p: 모듈별 학습 Profile.
        kind: classification/regression/rnn/time_series.
    Returns:
        dict: 무작위 IID 분할의 학습 결과.
    Raises:
        ValueError/TypeError: 입력 shape·타입·조건 위반.
        ImportError: TensorFlow 미설치.
    """
    X = numeric_X(X, 3 if kind == 'rnn' else 2)
    classification = kind == 'classification' or (kind == 'rnn' and p.task == 'classification')
    y = np.asarray(y)
    if y.ndim == 0: raise ValueError('타깃은 표본 축이 필요합니다.')
    if len(y) != len(X): raise ValueError('X/y 표본수 불일치')
    if classification:
        if y.ndim != 1 or y.dtype.kind not in 'biufUS': raise ValueError('분류 타깃은 1차원 수치/문자 라벨입니다.')
        if y.dtype.kind in 'f' and not np.isfinite(y).all(): raise ValueError('타깃 결측')
    else:
        y = np.asarray(y,dtype=np.float32)
        if y.ndim not in (1,2) or (y.ndim == 2 and y.shape[1] == 0) or not np.isfinite(y).all(): raise ValueError('유한한 회귀 타깃이 필요합니다.')
    indices = np.arange(len(X))
    remaining, test = train_test_split(indices,test_size=p.test_size,random_state=p.random_state,stratify=y if classification else None)
    train, val = train_test_split(remaining,test_size=p.validation_size/(1-p.test_size),random_state=p.random_state,stratify=y[remaining] if classification else None)
    return _fit(X,y,train,val,test,p,kind,classification)


def make_windows(series, input_width, horizon, target_columns=None, offset=0):
    """연속 미래 horizon 타깃을 생성합니다. 반환은 X,y,원 타깃 시점입니다.

    Args:
        series: 시간 순 (시점,특성) 자료.
        input_width: 과거 입력 창 길이.
        horizon: 즉시 다음 미래 예측 길이.
        target_columns: 타깃 열 위치 tuple 또는 None.
        offset: 반환 타깃 시점에 더할 원 자료 시작 위치.
    Returns:
        tuple: 입력 창, 미래 타깃, 원 타깃 시점.
    Raises:
        ValueError/TypeError: 입력 shape·타입·조건 위반.
    """
    a = numeric_X(series,2)
    positive_int(input_width,'input_width'); positive_int(horizon,'horizon')
    columns = tuple(range(a.shape[1])) if target_columns is None else target_columns
    if not columns or len(set(columns)) != len(columns): raise ValueError('타깃 열 선택 오류')
    for c in columns:
        positive_int(c,'target_columns',0)
        if c >= a.shape[1]: raise ValueError('타깃 열 범위 오류')
    n = len(a)-input_width-horizon+1
    if n <= 0: raise ValueError('각 분할에 input_width+horizon 이상의 시점이 필요합니다.')
    X = np.stack([a[i:i+input_width] for i in range(n)])
    times = np.stack([np.arange(i+input_width,i+input_width+horizon) for i in range(n)])
    y = a[times][:,:,columns]
    return X,y,times+offset


def run_forecast(series, p):
    """원 시점을 먼저 분리하고 각 구간 내부에서만 윈도를 생성합니다.

    Args:
        series: 시간 순 (시점,특성) 자료.
        p: 모듈별 학습 Profile.
    Returns:
        dict: 시간 순 분할 및 타깃 시점의 학습 결과.
    Raises:
        ValueError/TypeError: 입력 shape·타입·조건 위반.
        ImportError: TensorFlow 미설치.
    """
    a = numeric_X(series,2)
    n = len(a); end_train = int(n*(1-p.test_size-p.validation_size)); end_val = int(n*(1-p.test_size))
    blocks = [(0,end_train),(end_train,end_val),(end_val,n)]
    windows = [make_windows(a[start:end],p.input_width,p.horizon,p.target_columns,start) for start,end in blocks]
    X = np.concatenate([w[0] for w in windows]); y = np.concatenate([w[1] for w in windows])
    sizes = [len(w[0]) for w in windows]; cut = np.cumsum([0]+sizes)
    train,val,test = [np.arange(cut[i],cut[i+1]) for i in range(3)]
    result = _fit(X,y,train,val,test,p,'time_series',False,windows[2][2])
    result.update(train_indices=np.arange(end_train),validation_indices=np.arange(end_train,end_val),test_indices=np.arange(end_val,n))
    return result
