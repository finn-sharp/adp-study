"""분포별 NB native/PMF 참조·평활화·미지/결측·보류/CV 누수·정수 정의역 검증."""
import unittest,importlib,json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB,ComplementNB
from src.MachineLearning.NaiveBayes.poisson import PoissonNB,run_poisson
from src.MachineLearning.NaiveBayes.binomial import BinomialNB,run_binomial
from src.MachineLearning.NaiveBayes.categorical import CategoricalNB,run_categorical

class DistributionNaiveBayesTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.RandomState(17);self.X=pd.DataFrame(rng.binomial(5,.35,size=(120,4)),columns=list('abcd'));self.y=np.array(['ham','spam']*60)

    def test_four_native_families_match_identical_training_rows(self):
        for name,native in [('gaussian',GaussianNB()),('bernoulli',BernoulliNB(binarize=None)),('multinomial',MultinomialNB()),('complement',ComplementNB())]:
            data=(self.X>0).astype(int) if name=='bernoulli' else self.X
            module=importlib.import_module('src.MachineLearning.NaiveBayes.'+name);result=getattr(module,'run_'+name)(data,self.y)
            reference=clone(native).fit(data.iloc[result.train_indices],self.y[result.train_indices]);np.testing.assert_allclose(result.probabilities,reference.predict_proba(data.iloc[result.test_indices]));np.testing.assert_array_equal(result.predictions,reference.predict(data.iloc[result.test_indices]));json.loads(result.to_json())
            np.testing.assert_array_equal(result.class_counts,np.array([(self.y[result.train_indices]==label).sum() for label in result.classes]))

    def test_poisson_gamma_rate_and_log_pmf_posterior(self):
        model=PoissonNB(rate_alpha=2.,rate_beta=3.).fit(self.X,self.y);new=self.X.iloc[:6]
        expected=np.vstack([(self.X.loc[self.y==label].sum().to_numpy()+2)/(np.sum(self.y==label)+3) for label in model.classes_]);np.testing.assert_allclose(model.rates_,expected)
        scores=np.column_stack([stats.poisson.logpmf(new.to_numpy(),rate).sum(axis=1)+np.log(count/len(self.y)) for rate,count in zip(expected,model.class_count_)])
        np.testing.assert_allclose(model.predict_log_proba(new),scores-logsumexp(scores,axis=1,keepdims=True));np.testing.assert_allclose(model.predict_proba(new).sum(axis=1),1.)
        clone(model).fit(self.X,self.y);self.assertEqual(model.predict(new).dtype,self.y.dtype)

    def test_binomial_beta_probability_pmf_and_bernoulli_equivalence(self):
        model=BinomialNB(n_trials=5,alpha=2.,beta=3.).fit(self.X,self.y);new=self.X.iloc[:6]
        expected=np.vstack([(self.X.loc[self.y==label].sum().to_numpy()+2)/(np.sum(self.y==label)*5+5) for label in model.classes_]);np.testing.assert_allclose(model.success_prob_,expected)
        scores=np.column_stack([stats.binom.logpmf(new.to_numpy(),5,p).sum(axis=1)+np.log(count/len(self.y)) for p,count in zip(expected,model.class_count_)])
        np.testing.assert_allclose(model.predict_log_proba(new),scores-logsumexp(scores,axis=1,keepdims=True))
        binary=(self.X>0).astype(int);binomial=BinomialNB().fit(binary,self.y);bernoulli=BernoulliNB(alpha=1.,binarize=None).fit(binary,self.y);np.testing.assert_allclose(binomial.predict_proba(binary),bernoulli.predict_proba(binary))
        clone(model).fit(self.X,self.y)

    def test_categorical_counts_unknown_and_missing_are_distinct_and_train_only(self):
        X=pd.DataFrame({'city':['Seoul','Busan',None,'(missing)']*30,'grade':['low','high']*60});test=np.arange(80,120);X.loc[test,'city']='NewCity'
        result=run_categorical(X,self.y,test_indices=test);model=result.model.named_steps['model'];self.assertNotIn('NewCity',model.category_maps_[0]);self.assertIn('(missing)',model.category_maps_[0]);self.assertTrue((model.category_count_[0][:,0]==0).all());self.assertGreater(model.category_count_[0][:,1].sum(),0)
        np.testing.assert_allclose(result.probabilities.sum(axis=1),1.)
        new=pd.DataFrame({'city':[None,'NeverSeen','(missing)'],'grade':['low','high','low']});prediction=model.predict_proba(new);self.assertTrue(np.isfinite(prediction).all())
        mapping=model.category_maps_[0];self.assertNotEqual(mapping['(missing)'],1);clone(model).fit(X.iloc[:80],self.y[:80])
        # Known category posterior is sum of independently normalized Feature counts.
        score=model.class_log_prior_.copy()
        for j,value in enumerate(['Seoul','low']):score+=model.feature_log_prob_[j][:,model.category_maps_[j][value]]
        expected=np.exp(score-logsumexp(score));np.testing.assert_allclose(model.predict_proba(pd.DataFrame({'city':['Seoul'],'grade':['low']}))[0],expected)

    def test_fixed_split_cv_chronological_and_unseen_classes(self):
        result=run_poisson(self.X,self.y,test_indices=np.arange(90,120),overrides={'param_grid':{'rate_alpha':[.5,1.,2.]},'cv':3});self.assertEqual(len(result.train_indices),90);self.assertFalse(set(result.train_indices)&set(result.test_indices));self.assertTrue(result.best_params)
        fitted=result.model.named_steps['model'];expected=np.vstack([(self.X.iloc[result.train_indices].loc[self.y[result.train_indices]==label].sum().to_numpy()+fitted.rate_alpha)/(np.sum(self.y[result.train_indices]==label)+fitted.rate_beta) for label in fitted.classes_]);np.testing.assert_allclose(fitted.rates_,expected)
        chronological=run_binomial(self.X,self.y,overrides={'n_trials':5,'split_strategy':'chronological'});self.assertLess(chronological.train_indices.max(),chronological.test_indices.min())
        with self.assertRaises(ValueError):run_poisson(self.X,self.y,test_indices=[1,1])
        unseen=self.y.copy();unseen[90:]='new'
        with self.assertRaises(ValueError):run_poisson(self.X,unseen,test_indices=np.arange(90,120))

    def test_distribution_domains_and_prediction_schema(self):
        with self.assertRaises(ValueError):run_poisson(self.X+.1,self.y)
        with self.assertRaises(ValueError):run_poisson(self.X-10,self.y)
        with self.assertRaises(ValueError):run_poisson(self.X,self.y,overrides={'scaling':'standard'})
        with self.assertRaises(ValueError):run_binomial(self.X,self.y,overrides={'n_trials':1})
        with self.assertRaises(ValueError):run_binomial(self.X,self.y,overrides={'n_trials':0})
        from src.MachineLearning.NaiveBayes.bernoulli import run_bernoulli
        with self.assertRaises(ValueError):run_bernoulli(self.X,self.y)
        self.assertEqual(len(run_bernoulli(self.X,self.y,overrides={'binarize':2.}).predictions),36)
        with self.assertRaises(ValueError):run_categorical(pd.DataFrame({'a':[1,'1',1,'1']}),[0,1,0,1])
        model=CategoricalNB().fit(pd.DataFrame({'a':['x','y','x','y']}),[0,1,0,1])
        with self.assertRaises(ValueError):model.predict(pd.DataFrame({'b':['x']}))
        large=PoissonNB().fit(np.array([[0,1000],[1,900],[900,1],[1000,0]]),['a','a','b','b']);np.testing.assert_allclose(large.predict_proba([[10000,0]]).sum(),1.)


class SmallPriorStabilityTest(unittest.TestCase):
    def test_binomial_tiny_positive_prior_keeps_finite_log_probabilities(self):
        from src.MachineLearning.NaiveBayes.binomial import BinomialNB
        model=BinomialNB(n_trials=1,alpha=1e-30,beta=1e-30).fit([[0],[0],[1],[1]],[0,0,1,1])
        self.assertTrue(np.isfinite(model.predict_log_proba([[0],[1]])).all())
        np.testing.assert_allclose(model.predict_proba([[0],[1]]).sum(axis=1),1.)

if __name__=='__main__':unittest.main()
