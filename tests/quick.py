import pandas as pd

from src.pipeline import MLPipeline
from src.whatif import WhatIfAnalyzer  # <-- change if your file is elsewhere


def main():
    # 1) Load demo data
    df = pd.read_csv("data/demo_business_dataset.csv")
    target_col = "churn"

    # 2) Build and run pipeline
    pipeline = MLPipeline(df, target_col)
    result = pipeline.run_full_pipeline()

    if result["status"] != "success":
        raise RuntimeError(result)

    # 3) Grab what we need from pipeline/preprocessor
    feature_names = result["feature_names"]

    # IMPORTANT:
    # These should come from your Preprocessor so they match training.
    # If you don't have these getters yet, I’ll show fallback below.
    numeric_cols = pipeline.preprocessor.numeric_cols
    categorical_cols = pipeline.preprocessor.categorical_cols

    # If you store encoders in the preprocessor, use that:
    label_encoders = getattr(pipeline.preprocessor, "label_encoders", {})

    # 4) Create WhatIf analyzer
    whatif = WhatIfAnalyzer(
        pipeline=pipeline,
        feature_names=feature_names,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        label_encoders=label_encoders,
    )

    # 5) Print diagnostics
    print("onehot?", whatif._is_onehot)
    print("\nFeature ranges (numeric):")
    print(whatif.get_feature_ranges())

    print("\nCategorical options:")
    print(whatif.get_categorical_options())


if __name__ == "__main__":
    main()
