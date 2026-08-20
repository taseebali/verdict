"""Manual end-to-end smoke test of the core ML pipeline, persistence, ensembles, and explainability."""
import sys, traceback
import pandas as pd
sys.path.insert(0, ".")

def step(name, fn):
    try:
        result = fn()
        print(f"[OK] {name}")
        return result
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()
        sys.exit(1)

df = step("load demo csv", lambda: pd.read_csv("data/verdict_demo.csv"))
print(f"  shape={df.shape}")

from src.core.pipeline import MLPipeline
pipeline = MLPipeline(df, target_col="churn")

step("validate", lambda: pipeline.validate())
step("preprocess", lambda: pipeline.preprocess())
train_results = step("train (random_forest)", lambda: pipeline.train(["random_forest"]))
print(f"  train_results keys={list(train_results.keys())}")

eval_results = step("evaluate", lambda: pipeline.evaluate())
print(f"  eval_results={eval_results}")

# Model persistence
from src.artifacts.model_serializer import ModelSerializer
model = pipeline.model_manager.get_models()["random_forest"]

def do_save():
    ser = ModelSerializer()
    result = ser.save_model(model, model_name="random_forest_smoketest", metadata={"target": "churn"}, overwrite=True)
    return ser, result

ser, save_result = step("save model", do_save)
print(f"  save result: {save_result}")

def do_load():
    return ser.load_model("random_forest_smoketest")

loaded = step("load model back", do_load)

def do_predict_with_loaded():
    X_test, y_test = pipeline.get_test_data()
    return loaded.predict(X_test[:5])

preds = step("predict with reloaded model", do_predict_with_loaded)
print(f"  preds={preds}")

# SHAP explainability
from src.explain.explainability import ExplainabilityAnalyzer

def do_explain():
    X_train, X_test = pipeline.X_train, pipeline.X_test
    y_test = pipeline.y_test
    analyzer = ExplainabilityAnalyzer(model, X_train, X_test, pipeline.preprocessor.get_feature_names())
    return analyzer.get_feature_importance(use_cache=False, y_test=y_test)

importance = step("SHAP feature importance", do_explain)
print(f"  importance: {importance}")

# Ensembles (Voting + Stacking, sklearn-native, no xgboost/lightgbm required)
from src.core.ml_operations import EnsembleManager

def do_voting():
    X_train, X_test = pipeline.X_train, pipeline.X_test
    y_train, y_test = pipeline.y_train, pipeline.y_test
    mgr = EnsembleManager()
    return mgr.train_voting(X_train, X_test, y_train, y_test, voting="soft")

voting_result = step("train VotingClassifier ensemble", do_voting)
print(f"  voting test accuracy={getattr(voting_result, 'test_score', voting_result)}")

def do_stacking():
    X_train, X_test = pipeline.X_train, pipeline.X_test
    y_train, y_test = pipeline.y_train, pipeline.y_test
    mgr = EnsembleManager()
    return mgr.train_stacking(X_train, X_test, y_train, y_test)

stacking_result = step("train StackingClassifier ensemble", do_stacking)
print(f"  stacking test accuracy={getattr(stacking_result, 'test_score', stacking_result)}")

print("\nALL SMOKE TESTS PASSED")
