"""GA의 제약/정수/재현성/엘리트 및 생산 전수 해를 독립 검증."""
import ast,importlib,json,sys
from dataclasses import fields
from pathlib import Path
from itertools import product
import unittest
import numpy as np
from src.Recipes import load_profile
from src.Optimization.genetic_algorithm import run_genetic_algorithm
from src.Optimization.production_allocation import run_production_allocation

class GeneticOptimizationTest(unittest.TestCase):
    def test_integer_optimum_reproducibility_and_elitism(self):
        objective=lambda x:float((x[0]-3)**2+(x[1]-2)**2)
        kwargs=dict(integer_mask=[True,True],constraints=lambda x:([4-x.sum()],[]),overrides={'population_size':30,'generations':40,'random_state':7})
        a=run_genetic_algorithm(objective,[[0,5],[0,5]],**kwargs);b=run_genetic_algorithm(objective,[[0,5],[0,5]],**kwargs)
        self.assertEqual(a.best_objective,0);np.testing.assert_array_equal(a.best_solution,[3,2]);np.testing.assert_array_equal(a.population,b.population)
        self.assertTrue((np.diff(a.history.best_objective.dropna())<=0).all());self.assertTrue((a.population==np.rint(a.population)).all());json.loads(a.to_json())
        self.assertEqual(a.n_evaluations,(a.n_generations+1)*30)

    def test_mixed_maximize_and_feasibility_priority(self):
        r=run_genetic_algorithm(lambda x:float(x.sum()),[[0,4],[0,1]],integer_mask=[True,False],constraints=lambda x:([x.sum()-3],[]),initial_population=[[2,1]],overrides={'maximize':True,'population_size':20,'generations':10})
        self.assertAlmostEqual(r.best_objective,3);self.assertTrue(r.feasible);self.assertEqual(r.best_solution[0],round(r.best_solution[0]))
        no=run_genetic_algorithm(lambda x:-float(x[0]),[[0,1]],constraints=lambda x:([2-x[0]],[]),overrides={'generations':3})
        self.assertFalse(no.feasible);self.assertIsNone(no.best_solution);self.assertIsNone(json.loads(no.to_json())['best_objective']);self.assertGreater(no.best_violation,0)
        eq=run_genetic_algorithm(lambda x:float(x[0]**2),[[0,2]],integer_mask=[True],constraints=lambda x:([],[x[0]-1]),initial_population=[[1]])
        self.assertTrue(eq.feasible);self.assertEqual(eq.best_objective,1)

    def test_bad_domains_fail(self):
        for bounds in ([],[[2,1]],[[0,np.inf]],[[0,1,2]]):
            with self.assertRaises(ValueError):run_genetic_algorithm(lambda x:0.,bounds)
        for objective in (lambda x:np.nan,lambda x:[1]):
            with self.assertRaises(ValueError):run_genetic_algorithm(objective,[[0,1]])
        with self.assertRaises(ValueError):run_genetic_algorithm(lambda x:0.,[[.1,.9]],integer_mask=[True])
        with self.assertRaises(ValueError):run_genetic_algorithm(lambda x:0.,[[0,1]],initial_population=[[2]])
        with self.assertRaises(ValueError):run_genetic_algorithm(lambda x:0.,[[0,1]],overrides={'mutation_rate':2.})
        with self.assertRaises(ValueError):run_genetic_algorithm(lambda x:0.,[[0,1]],integer_mask=[1])

    def test_production_reference_enumeration_and_impossible_budget(self):
        r=run_production_allocation([3,2,4],[5000,7000,4000],[1000,1200,800]);np.testing.assert_array_equal(r.exact_solution,[2,1,2]);self.assertEqual(r.total_cost,29800);self.assertEqual(r.annual_production,60000);self.assertEqual(r.objective_gap,0)
        candidates=[(6000*a+8200*b+4800*c,(a,b,c)) for a,b,c in product(range(1,11),repeat=3) if min(a*40000,b*60000,c*30000)>=50000]
        self.assertEqual(min(candidates)[0],r.exact_cost)
        no=run_production_allocation([3,2,4],[5000,7000,4000],[1000,1200,800],overrides={'budget':29000.,'generations':4});self.assertEqual(no.exact_status,'infeasible');self.assertIsNone(no.total_cost)
        with self.assertRaises(ValueError):run_production_allocation([3]*8,[1]*8,[1]*8)
        json.loads(r.to_json())

    def test_ax_and_contract_and_skeleton(self):
        import matplotlib;matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from src.Visualization.optimization.convergence import plot_convergence
        from src.Optimization.skeleton.production_allocation import main
        r=main();fig,axes=plt.subplots(1,2);before=tuple(axes[1].get_children());out=plot_convergence(r.ga_result,ax=axes[0]);self.assertIs(out.axes,axes[0]);self.assertEqual(tuple(axes[1].get_children()),before);self.assertEqual(len(fig.axes),2);json.loads(out.to_json());plt.close(fig)
        for key in ('genetic-algorithm','production-allocation','plot-optimization-convergence'):
            from src.Recipes.loader import _REGISTRY
            module=importlib.import_module(_REGISTRY[key][0]);profile=load_profile(key)
            for cls in (type(profile),getattr(module,type(profile).__name__.replace('Profile','OutputDTO'))):
                for f in fields(cls):self.assertIn(f.name+' (',cls.__doc__)
            source=Path(module.__file__).read_text()
            if sys.version_info>=(3,8):ast.parse(source,feature_version=(3,7))
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,ast.FunctionDef):
                    for label in ('Args:','Returns:','Raises:'):self.assertIn(label,ast.get_docstring(node))

if __name__=='__main__':unittest.main()
