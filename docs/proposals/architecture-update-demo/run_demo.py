"""저장소 루트에서 실행하는 검토 전용 harness. src/등록부/기본 JSON은 수정하지 않습니다."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.Recipes import loader


def main():
    """후보 모듈을 임시 등록하고 독립 수계산·계약을 검증한 뒤 등록을 복원합니다."""
    name = "architecture_review_km"
    key = "survival-kaplan-meier"
    previous_module = sys.modules.get(name)
    previous_entry = loader._REGISTRY.get(key)
    spec = importlib.util.spec_from_file_location(name, HERE / "kaplan_meier.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        loader._REGISTRY[key] = (name, "KaplanMeierProfile", None, (None,))
        recipe = str(HERE / "survival-kaplan-meier.json")
        result = module.run_kaplan_meier(
            [1, 2, 2, 3, 4], [1, 1, 0, 0, 1], recipe_path=recipe,
            overrides={"label": "공급사 A", "evaluation_times": [0, 1, 2, 3, 4]})
        # t=2 직전 위험집합은 사건 1명과 절단 1명을 모두 포함한 4명.
        np.testing.assert_array_equal(result.at_risk, [5, 4, 2, 1])
        np.testing.assert_allclose(result.survival, [4/5, (4/5)*(3/4), 3/5, 0])
        np.testing.assert_allclose(result.survival_at_times, [1, .8, .6, .6, 0])
        assert result.median_survival_time == 4
        json.loads(result.to_json())
        (HERE / "demo_result.json").write_text(result.to_json(indent=2) + "\n", encoding="utf-8")
        checks = {"ties_hand_calculation": True, "right_continuous_queries": True,
                  "median_crossing": True, "existing_loader_and_DTO": True}
        censored = module.run_kaplan_meier([1, 2], [0, 0], recipe_path=recipe)
        assert censored.median_survival_time is None
        np.testing.assert_array_equal(censored.survival, [1, 1])
        zero = module.run_kaplan_meier([0, 1], [1, 0], recipe_path=recipe,
                                      overrides={"evaluation_times": [0, 1]})
        np.testing.assert_allclose(zero.survival_at_times, [.5, .5])
        checks["all_censored_and_time_zero"] = True
        permuted = module.run_kaplan_meier([4, 2, 1, 3, 2], [1, 0, 1, 0, 1], recipe_path=recipe)
        np.testing.assert_allclose(permuted.survival, result.survival)
        checks["row_order_invariance"] = True
        explicit = module.run_kaplan_meier([1, 2], [1, 0], module.KaplanMeierProfile())
        assert explicit.sample_size == 2
        invalid = [lambda: module.run_kaplan_meier([1], [2], recipe_path=recipe),
                   lambda: module.run_kaplan_meier([float('nan')], [1], recipe_path=recipe),
                   lambda: module.run_kaplan_meier([1], [1], recipe_path=recipe, overrides={"typo": 1}),
                   lambda: module.run_kaplan_meier([1], [1], recipe_path=recipe, overrides={"evaluation_times": [2]}),
                   lambda: module.run_kaplan_meier([1], [1], module.KaplanMeierProfile(), recipe_path=recipe)]
        for call in invalid:
            try:
                call()
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError("잘못된 입력이 거부되지 않았습니다.")
        checks["invalid_input_and_configuration_rejected"] = True
        ast.parse((HERE / "kaplan_meier.py").read_text(), feature_version=(3, 7))
        checks["python37_grammar_only"] = True
        reference = "not installed; independent hand calculation used"
        try:
            from lifelines import KaplanMeierFitter
        except ImportError:
            pass
        else:
            fitted = KaplanMeierFitter().fit([1, 2, 2, 3, 4], [1, 1, 0, 0, 1])
            np.testing.assert_allclose(fitted.predict(result.times), result.survival)
            assert fitted.median_survival_time_ == result.median_survival_time
            reference = "lifelines KaplanMeierFitter matched"
        report = {"status": "passed", "python": sys.version.split()[0], "checks": checks,
                  "reference_comparison": reference,
                  "scope": "review candidate only; no production registration; no Python 3.7 runtime claim"}
        (HERE / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if previous_entry is None:
            loader._REGISTRY.pop(key, None)
        else:
            loader._REGISTRY[key] = previous_entry
        if previous_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module


if __name__ == "__main__":
    main()
