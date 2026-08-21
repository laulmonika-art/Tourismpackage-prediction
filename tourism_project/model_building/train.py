
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

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


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

XTRAIN_PATH = DATA_DIR / "Xtrain.csv"
XTEST_PATH = DATA_DIR / "Xtest.csv"
YTRAIN_PATH = DATA_DIR / "ytrain.csv"
YTEST_PATH = DATA_DIR / "ytest.csv"

DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = DEPLOYMENT_DIR / "tourism_model.joblib"


# =========================================================
# Load train and test data
# =========================================================

print("Loading training and testing data...")

for path in [
    XTRAIN_PATH,
    XTEST_PATH,
    YTRAIN_PATH,
    YTEST_PATH
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

X_train = pd.read_csv(XTRAIN_PATH)
X_test = pd.read_csv(XTEST_PATH)

y_train = pd.read_csv(YTRAIN_PATH).squeeze()
y_test = pd.read_csv(YTEST_PATH).squeeze()

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# =========================================================
# Identify numerical and categorical columns
# =========================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical columns:")
print(numeric_features)

print("\nCategorical columns:")
print(categorical_features)


# =========================================================
# Preprocessing
# =========================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
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
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# =========================================================
# Define model
# =========================================================

model = RandomForestClassifier(
    random_state=42
)


# =========================================================
# Create pipeline
# =========================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


# =========================================================
# Hyperparameter grid
# =========================================================

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5]
}


# =========================================================
# MLflow experiment
# =========================================================

mlflow.set_experiment(
    "Tourism_Package_Prediction"
)


# =========================================================
# Hyperparameter tuning
# =========================================================

print("\nStarting GridSearchCV...")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)


with mlflow.start_run():

    grid_search.fit(
        X_train,
        y_train
    )

    # Best model
    best_model = grid_search.best_estimator_

    # Best parameters
    best_params = grid_search.best_params_

    print("\nBest parameters:")
    print(best_params)

    print(
        "\nBest cross-validation accuracy:",
        grid_search.best_score_
    )


    # =====================================================
    # Log parameters
    # =====================================================

    mlflow.log_params(best_params)

    mlflow.log_metric(
        "cv_accuracy",
        grid_search.best_score_
    )


    # =====================================================
    # Test prediction
    # =====================================================

    y_pred = best_model.predict(X_test)


    # =====================================================
    # Evaluation metrics
    # =====================================================

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


    # =====================================================
    # Log metrics to MLflow
    # =====================================================

    mlflow.log_metric(
        "test_accuracy",
        accuracy
    )

    mlflow.log_metric(
        "test_precision",
        precision
    )

    mlflow.log_metric(
        "test_recall",
        recall
    )

    mlflow.log_metric(
        "test_f1",
        f1
    )


    # =====================================================
    # Display results
    # =====================================================

    print("\nModel Evaluation")
    print("==============================")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


    # =====================================================
    # Save best model
    # =====================================================

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    print(
        "\nBest model saved to:"
    )

    print(MODEL_PATH)


    # =====================================================
    # Log model to MLflow
    # =====================================================

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="tourism_model"
    )


print("\nTraining completed successfully.")
