"""별도 CNN·순환 모델의 아키텍처·학습 누수·공식 test 분리·명세 검증."""
import ast
from dataclasses import fields
import importlib,importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
from src.Recipes import load_profile
from src.DeepLearning.TensorFlow.cnn import run_tf_cnn
from src.DeepLearning.TensorFlow.simple_rnn import run_tf_simple_rnn
from src.DeepLearning.TensorFlow.lstm import run_tf_lstm
from src.DeepLearning.TensorFlow.gru import run_tf_gru

CASES=[('tf-cnn','cnn','TFCNN'),('tf-simple-rnn','simple_rnn','TFSimpleRNN'),('tf-lstm','lstm','TFLSTM'),('tf-gru','gru','TFGRU')]


class DeepExpansionContractTest(unittest.TestCase):
    def test_separate_full_profile_dto_and_function_contracts(self):
        for key,file,name in CASES:
            module=importlib.import_module('src.DeepLearning.TensorFlow.'+file)
            P=getattr(module,name+'Profile');O=getattr(module,name+'OutputDTO')
            for task in ('classification','regression'):self.assertIsInstance(load_profile(key,task=task),P)
            source=Path(module.__file__).read_text();labels=['1. Imports','2. Input Profile','3. Output DTO','4. Input Validation','5. Individual Model Runners','6. Public Runner','7. Example Execution']
            self.assertEqual([source.index(s) for s in labels],sorted(source.index(s) for s in labels))
            for cls in (P,O):
                for field in fields(cls):self.assertIn(field.name+' (',cls.__doc__)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,ast.FunctionDef):
                    for label in ('Args:','Returns:','Raises:'):self.assertIn(label,ast.get_docstring(node))

    def test_validation_before_optional_tensorflow_import(self):
        X=np.ones((60,8,8,1));y=np.arange(60)%3
        for options in ({'dimension':3},{'conv_filters':[]},{'pool_size':10},{'dropout':1},{'epochs':True}):
            with self.assertRaises(ValueError):run_tf_cnn(X,y,overrides=options)
        with self.assertRaises(ValueError):run_tf_cnn(X.astype(complex),y)
        for runner in (run_tf_simple_rnn,run_tf_lstm,run_tf_gru):
            for options in ({'num_layers':0},{'bidirectional':1},{'recurrent_dropout':1}):
                with self.assertRaises(ValueError):runner(np.ones((60,8,2)),y,overrides=options)
            with self.assertRaises(ValueError):runner(X,y)
        with self.assertRaises(ValueError):run_tf_cnn(X,y,load_profile('tf-cnn'),overrides={})


@unittest.skipUnless(importlib.util.find_spec('tensorflow'),'별도 TensorFlow 환경에서 실제 학습 검증')
class DeepExpansionTrainingTest(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.RandomState(8);self.options={'epochs':2,'batch_size':16,'patience':1}

    def check_result(self,r):
        self.assertFalse(set(r.train_indices)&set(r.validation_indices))
        self.assertFalse(set(r.train_indices)&set(r.test_indices))
        self.assertFalse(set(r.validation_indices)&set(r.test_indices))
        self.assertEqual(r.predictions.shape,r.y_test.shape)
        self.assertEqual(r.best_epoch,int(np.argmin(r.history['val_loss']))+1)
        json.loads(r.to_json());self.assertNotIn('model',r.to_dict())

    def test_cnn2d_fixed_test_preserves_split_scaler_and_actual_conv_layers(self):
        X=self.rng.normal(size=(60,8,8,1));y=np.arange(60)%3
        Xt=self.rng.normal(loc=100,size=(15,8,8,1));yt=np.arange(15)%3
        r=run_tf_cnn(X,y,test_data=(Xt,yt),overrides=dict(self.options,conv_filters=[4,8],hidden_layer_sizes=[8]))
        self.check_result(r);self.assertEqual(r.split_mode,'provided_test')
        np.testing.assert_array_equal(r.test_indices,np.arange(60,75));np.testing.assert_array_equal(r.y_test,yt)
        np.testing.assert_allclose(r.model.scaler.mean_,X[r.train_indices].reshape(-1,1).mean(axis=0),rtol=1e-6)
        from tensorflow.keras.layers import Conv2D
        self.assertEqual(sum(isinstance(layer,Conv2D) for layer in r.model.keras_model.layers),2)
        np.testing.assert_allclose(r.model.predict_proba(Xt[:3]).sum(axis=1),1,atol=1e-6)
        with self.assertRaises(ValueError):r.model.predict(np.ones((1,4,4,1)))

    def test_cnn1d_multitarget_regression_and_unseen_fixed_test_labels(self):
        X=self.rng.normal(size=(60,12,2));y=np.column_stack([X[:,:,0].mean(axis=1),X[:,:,1].mean(axis=1)])
        r=run_tf_cnn(X,y,task='regression',overrides=dict(self.options,dimension=1,conv_filters=[4],hidden_layer_sizes=[]))
        self.check_result(r);self.assertEqual(r.model.predict(X[:2]).shape,(2,2))
        from tensorflow.keras.layers import Conv1D
        self.assertTrue(any(isinstance(layer,Conv1D) for layer in r.model.keras_model.layers))
        with self.assertRaises(ValueError):run_tf_cnn(X,np.arange(60)%2,overrides={'dimension':1},test_data=(X[:2],np.array([8,8])))

    def test_stacked_bidirectional_simple_rnn_lstm_gru_are_distinct(self):
        X=self.rng.normal(size=(60,6,2));y=np.where(X[:,:,0].sum(axis=1)>0,'positive','negative')
        from tensorflow.keras.layers import Bidirectional,SimpleRNN,LSTM,GRU
        for runner,kind in [(run_tf_simple_rnn,SimpleRNN),(run_tf_lstm,LSTM),(run_tf_gru,GRU)]:
            r=runner(X,y,overrides=dict(self.options,num_layers=2,bidirectional=True,units=4,dropout=.1,recurrent_dropout=.1))
            self.check_result(r)
            layers=[l for l in r.model.keras_model.layers if isinstance(l,Bidirectional)]
            self.assertEqual(len(layers),2);self.assertIsInstance(layers[0].forward_layer,kind)
            self.assertTrue(layers[0].forward_layer.return_sequences);self.assertFalse(layers[1].forward_layer.return_sequences)
            np.testing.assert_array_equal(r.model.predict(X[r.test_indices]),r.predictions)
            reg=runner(X,X.sum(axis=(1,2)),task='regression',overrides=dict(self.options,units=4))
            self.check_result(reg);self.assertEqual(reg.model.predict(X[:2]).shape,(2,))


if __name__=='__main__':unittest.main()
