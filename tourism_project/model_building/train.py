
import pandas as pd
import joblib
import mlflow

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

XTRAIN_PATH = DATA_DIR / "Xtrain.csv"
XTEST_PATH = DATA_DIR / "Xtest.csv"
YTRAIN_PATH = DATA_DIR / "ytrain.csv"
YTEST_PATH = DATA_DIR / "ytest.csv"

DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = DEPLOYMENT_DIR / "tourism_model.joblib"


# ============================================================
# CHECK INPUT FILES
# ============================================================

required_files = [
    XTRAIN_PATH,
    XTEST_PATH,
    YTRAIN_PATH,
    YTEST_PATH
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

print("Loading train and test data...")

X_train = pd.read_csv(XTRAIN_PATH)
X_test = pd.read_csv(XTEST_PATH)

y_train = pd.read_csv(YTRAIN_PATH).squeeze("columns")
y_test = pd.read_csv(YTEST_PATH).squeeze("columns")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# IDENTIFY FEATURES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# MODEL
# ============================================================

rf_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf_model)
    ]
)


# ============================================================
# HYPERPARAMETER GRID
# ============================================================

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5]
}


# ============================================================
# MLFLOW
# ============================================================

mlflow.set_experiment(
    "Tourism_Package_Prediction"
)


# ============================================================
# GRID SEARCH
# ============================================================

print("\nStarting hyperparameter tuning...")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

with mlflow.start_run() as run:

    grid_search.fit(
        X_train,
        y_train
    )

    best_model = grid_search.best_estimator_

    best_params = grid_search.best_params_

    best_cv_score = grid_search.best_score_

    print("\nBest parameters:")
    print(best_params)

    print(
        f"\nBest CV accuracy: {best_cv_score:.4f}"
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    y_pred = best_model.predict(
        X_test
    )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # ========================================================
    # LOG PARAMETERS
    # ========================================================

    mlflow.log_params({
        "model": "RandomForestClassifier",
        "cv": 3,
        "scoring": "accuracy",
        **best_params
    })

    # ========================================================
    # LOG METRICS
    # ========================================================

    mlflow.log_metrics({
        "cv_accuracy": best_cv_score,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1
    })

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n==============================")
    print("MODEL EVALUATION")
    print("==============================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    print("\nBest model saved to:")
    print(MODEL_PATH)

    print(
        "Model file exists:",
        MODEL_PATH.exists()
    )

    print(
        "Model size:",
        MODEL_PATH.stat().st_size,
        "bytes"
    )

    print("\nMLflow parameters and metrics logged.")

    print(
        "MLflow run ID:",
        run.info.run_id
    )


# ============================================================
# FINAL CHECK
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model was not created: {MODEL_PATH}"
    )

print("\nTraining completed successfully!")
