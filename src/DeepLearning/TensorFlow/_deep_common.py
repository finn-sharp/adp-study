"""
================================================================================
_deep_common.py - 새 CNN·SimpleRNN·LSTM·GRU의 분할·학습·검증 계약.
================================================================================
설명:
    - 새 알고리즘 파일의 공통 검증·학습·자료 로딩 계약을 제공합니다.
분석 가정:
    - 입력 정의역·학습 전용 전처리·거래 분모와 알고리즘별 한도를 유지합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    별도 알고리즘의 공통 계약 추가
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
from .common import PredictionAdapter, numeric_X, _tensorflow, _dataset


def validate_training(profile):
    """새 모델의 공통 학습·분할 설정을 검증합니다.

    Args:
        profile: 새 모델 전용 Profile.
    Returns:
        None: 설정이 유효함.
    Raises:
        ValueError/TypeError: 설정 범위·타입 오류.
    """
    for name in ('random_state','patience'):
        positive_int(getattr(profile,name),name,0)
    for name in ('epochs','batch_size'):
        positive_int(getattr(profile,name),name)
    finite_number(profile.learning_rate,'learning_rate',True)
    for name in ('test_size','validation_size'):
        finite_number(getattr(profile,name),name,True)
        if getattr(profile,name)>=1:raise ValueError(name+'는 1 미만입니다.')
    if profile.task not in ('classification','regression'):raise ValueError('task 오류')
    if profile.scaling not in ('none','standard'):raise ValueError('scaling 오류')
    if isinstance(profile.verbose,bool) or profile.verbose not in (0,1,2):raise ValueError('verbose 오류')
    finite_number(profile.dropout,'dropout')
    if not 0<=profile.dropout<1:raise ValueError('dropout은 [0,1)입니다.')


def target_array(y, n, task):
    """원 라벨 또는 회귀 단위를 보존한 타깃을 확인합니다.

    Args:
        y: 분류 1차원 라벨 또는 회귀 1/2차원 수치.
        n: X의 표본 수.
        task: classification/regression.
    Returns:
        np.ndarray: 원 라벨 또는 float32 타깃.
    Raises:
        ValueError/TypeError: 타입·shape·결측 오류.
    """
    a=np.asarray(y)
    if np.iscomplexobj(a):raise ValueError('타깃은 실수 또는 문자열이어야 합니다.')
    if a.ndim==0 or len(a)!=n:raise ValueError('X/y 표본 수 불일치')
    if task=='classification':
        if a.ndim!=1 or a.dtype.kind not in 'biufUS':raise ValueError('1차원 수치/문자 분류 라벨이 필요합니다.')
        if a.dtype.kind in 'f' and not np.isfinite(a).all():raise ValueError('타깃 결측')
    else:
        a=np.asarray(y,dtype=np.float32)
        if a.ndim not in (1,2) or (a.ndim==2 and a.shape[1]==0) or not np.isfinite(a).all():raise ValueError('유한한 1/2차원 회귀 타깃이 필요합니다.')
    return a


def fit_supervised(X,y,profile,builder,test_data=None):
    """학습 자료만 정규화·라벨 적합하고 validation으로만 조기 종료합니다.

    Args:
        X: 검증한 CNN/순환 입력.
        y: 검증한 타깃.
        profile: 새 모델 Profile.
        builder: tf·입력 shape·출력 shape·분류 여부로 미적합 모델을 만드는 함수.
        test_data: 고정 보류 (X_test,y_test) 또는 None. 제공 시 validation_size는 X 내 비율.
    Returns:
        dict: 기존 TF DTO와 같은 필드 및 split_mode.
    Raises:
        ValueError/TypeError/ImportError: 입력·분할·클래스·학습 또는 TF 미설치.
    """
    classification=profile.task=='classification'
    indices=np.arange(len(X));split_mode='random_holdout'
    if test_data is None:
        if profile.test_size+profile.validation_size>=1:raise ValueError('학습 비율이 양수여야 합니다.')
        remaining,test=train_test_split(indices,test_size=profile.test_size,random_state=profile.random_state,stratify=y if classification else None)
        train,val=train_test_split(remaining,test_size=profile.validation_size/(1-profile.test_size),random_state=profile.random_state,stratify=y[remaining] if classification else None)
    else:
        if not isinstance(test_data,(tuple,list)) or len(test_data)!=2:raise ValueError('test_data는 (X_test,y_test)입니다.')
        Xt=numeric_X(test_data[0],X.ndim);yt=target_array(test_data[1],len(Xt),profile.task)
        if Xt.shape[1:]!=X.shape[1:] or yt.shape[1:]!=y.shape[1:]:raise ValueError('고정 test shape 불일치')
        train,val=train_test_split(indices,test_size=profile.validation_size,random_state=profile.random_state,stratify=y if classification else None)
        test=np.arange(len(X),len(X)+len(Xt));X=np.concatenate([X,Xt]);y=np.concatenate([y,yt]);split_mode='provided_test'
    scaler=StandardScaler().fit(X[train].reshape(-1,X.shape[-1])) if profile.scaling=='standard' else None
    scaled=scaler.transform(X.reshape(-1,X.shape[-1])).reshape(X.shape).astype(np.float32) if scaler is not None else X
    encoder=None;scalar=not classification and y.ndim==1
    if classification:
        encoder=LabelEncoder().fit(y[train])
        if len(encoder.classes_)<2:raise ValueError('학습 클래스가 2개 이상 필요합니다.')
        target=encoder.transform(y);output_shape=(len(encoder.classes_),)
    else:
        target=y[:,None] if scalar else y;output_shape=target.shape[1:]
    tf=_tensorflow(profile.random_state)
    model=builder(tf,X.shape[1:],output_shape,classification)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=profile.learning_rate),loss='sparse_categorical_crossentropy' if classification else 'mse')
    callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=profile.patience,restore_best_weights=True)
    history=model.fit(_dataset(tf,scaled[train],target[train],profile.batch_size),validation_data=_dataset(tf,scaled[val],target[val],profile.batch_size),epochs=profile.epochs,callbacks=[callback],verbose=profile.verbose,shuffle=False).history
    if callback.best_weights is not None:model.set_weights(callback.best_weights)
    if not all(np.isfinite(v).all() for v in history.values()):raise ValueError('비유한 학습 loss')
    adapter=PredictionAdapter(model,scaler,X.shape[1:],encoder,scalar)
    predictions=adapter.predict(X[test]);raw=model(scaled[test],training=False).numpy()
    if not np.isfinite(raw).all():raise ValueError('비유한 예측')
    proba=adapter.predict_proba(X[test]) if classification else None
    if classification:metrics={'accuracy':float(accuracy_score(y[test],predictions)),'log_loss':float(log_loss(y[test],proba,labels=encoder.classes_))}
    else:
        mse=float(mean_squared_error(y[test].reshape(len(test),-1),predictions.reshape(len(test),-1)))
        metrics={'mse':mse,'mae':float(mean_absolute_error(y[test].reshape(len(test),-1),predictions.reshape(len(test),-1))),'rmse':float(np.sqrt(mse))}
    parameters=asdict(profile)
    parameters.update(effective_test_fraction=len(test)/len(X),effective_validation_fraction=len(val)/len(X))
    return dict(task=profile.task,parameters=parameters,model=adapter,history={k:[float(v) for v in values] for k,values in history.items()},test_metrics=metrics,predictions=predictions,probabilities=proba,classes=None if encoder is None else encoder.classes_,train_indices=train,validation_indices=val,test_indices=test,target_indices=None,y_test=y[test],input_shape=tuple(X.shape[1:]),epochs_trained=len(history['loss']),best_epoch=int(np.argmin(history['val_loss']))+1,split_mode=split_mode)
