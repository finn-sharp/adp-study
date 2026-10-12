"""TF 선택 의존성, 실제 학습·새 입력 예측·누수 없는 분할과 원 시점 검증."""
import importlib.util
import json
import unittest
import numpy as np
from src.Recipes import load_profile
from src.DeepLearning.TensorFlow.classification import run_tf_classification
from src.DeepLearning.TensorFlow.regression import run_tf_regression
from src.DeepLearning.TensorFlow.rnn import run_tf_rnn
from src.DeepLearning.TensorFlow.time_series import run_tf_time_series
from src.DeepLearning.TensorFlow.common import make_windows

class TensorFlowContractTest(unittest.TestCase):
    def test_windows_target_alignment(self):
        series=np.arange(80).reshape(40,2)
        X,y,times=make_windows(series,4,3,(1,),offset=100)
        self.assertEqual(X.shape,(34,4,2));self.assertEqual(y.shape,(34,3,1))
        np.testing.assert_array_equal(X[0],series[:4])
        np.testing.assert_array_equal(y[0,:,0],series[4:7,1])
        np.testing.assert_array_equal(times[0],[104,105,106])
        with self.assertRaises(ValueError):make_windows(series,40,2)

    def test_validation_before_training(self):
        X=np.ones((40,3));y=np.arange(40.)
        for config in ({'epochs':True},{'test_size':0.9,'validation_size':0.2},{'learning_rate':0},{'patience':-1}):
            with self.assertRaises(ValueError):run_tf_regression(X,y,overrides=config)
        with self.assertRaises(ValueError):run_tf_rnn(X,y)
        with self.assertRaises(ValueError):run_tf_time_series(np.ones((20,2)))
        p=load_profile('tf-regression')
        with self.assertRaises(ValueError):run_tf_regression(X,y,p,overrides={'epochs':1})
        self.assertIsInstance(load_profile('tf-time-series',overrides={'target_columns':[0]}).target_columns,tuple)

@unittest.skipUnless(importlib.util.find_spec('tensorflow'), '별도 TensorFlow CPU 환경에서 실제 학습 검증')
class TensorFlowTrainingTest(unittest.TestCase):
    def setUp(self):
        self.X=np.random.RandomState(7).normal(size=(80,3)).astype(np.float32)
        self.options={'epochs':2,'batch_size':16,'patience':1}

    def check_result(self,r):
        self.assertEqual(r.epochs_trained,len(r.history['loss']))
        self.assertEqual(r.best_epoch,np.argmin(r.history['val_loss'])+1)
        json.loads(r.to_json());self.assertNotIn('model',r.to_dict())
        self.assertFalse(set(r.train_indices)&set(r.test_indices))
        self.assertFalse(set(r.validation_indices)&set(r.test_indices))

    def test_dense_classification_labels_and_training_scaler(self):
        y=np.where(self.X[:,0]>0,'positive','negative')
        r=run_tf_classification(self.X,y,overrides=self.options)
        self.check_result(r)
        np.testing.assert_allclose(r.model.scaler.mean_,self.X[r.train_indices].mean(axis=0),atol=1e-7)
        np.testing.assert_array_equal(r.model.predict(self.X[r.test_indices]),r.predictions)
        np.testing.assert_allclose(r.probabilities.sum(axis=1),1,atol=1e-6)
        np.testing.assert_array_equal(r.classes,['negative','positive'])
        # Full budget completion must also restore the minimum-validation-loss weights.
        import tensorflow as tf
        values=r.model._raw(self.X[r.validation_indices])
        truth=r.model.encoder.transform(y[r.validation_indices])
        loss=float(tf.keras.losses.sparse_categorical_crossentropy(truth,values).numpy().mean())
        self.assertAlmostEqual(loss,min(r.history['val_loss']),places=5)

    def test_dense_regression_shapes_and_metric(self):
        y=self.X[:,0]*2
        for target in (y,np.stack([y,y*3],axis=1)):
            r=run_tf_regression(self.X,target,overrides=self.options)
            self.check_result(r)
            self.assertEqual(r.predictions.shape,r.y_test.shape)
            self.assertAlmostEqual(r.test_metrics['mse'],np.mean((r.y_test-r.predictions)**2),places=5)
            np.testing.assert_allclose(r.model.predict(self.X[r.test_indices]),r.predictions)

    def test_all_rnn_cells_and_classification(self):
        X=np.repeat(self.X[:,None,:],4,axis=1)
        for cell in ('simple_rnn','lstm','gru'):
            r=run_tf_rnn(X,self.X[:,0],overrides=dict(self.options,cell=cell))
            self.check_result(r)
            self.assertEqual(r.input_shape,(4,3))
            np.testing.assert_allclose(r.model.predict(X[r.test_indices]),r.predictions)
        r=run_tf_rnn(X,(self.X[:,0]>0).astype(int),task='classification',overrides=self.options)
        self.assertEqual(r.probabilities.shape[1],2)

    def test_chronological_forecast_and_original_units(self):
        series=np.stack([np.arange(160)*0.1,np.sin(np.arange(160)/4)],axis=1)
        r=run_tf_time_series(series,overrides=dict(self.options,input_width=4,horizon=3,target_columns=[0]))
        self.check_result(r)
        self.assertLess(max(r.train_indices),min(r.validation_indices))
        self.assertLess(max(r.validation_indices),min(r.test_indices))
        self.assertEqual(r.predictions.shape[1:],(3,1))
        np.testing.assert_allclose(r.y_test[:,:,0],series[r.target_indices,0])
        X,y,t=make_windows(series[r.test_indices],4,3,(0,),offset=r.test_indices[0])
        np.testing.assert_allclose(r.model.predict(X),r.predictions)
        # Scaler uses training input windows only (not raw validation/test).
        trainX,_,_=make_windows(series[r.train_indices],4,3,(0,))
        np.testing.assert_allclose(r.model.scaler.mean_,trainX.reshape(-1,2).mean(axis=0),atol=1e-6)
        self.assertGreaterEqual(r.target_indices.min(),min(r.test_indices)+4)

if __name__=='__main__':unittest.main()
