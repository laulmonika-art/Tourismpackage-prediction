import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = DEPLOYMENT_DIR / "tourism_model.joblib"


# ---------------------------------------------------------
# Load train and test data
# ---------------------------------------------------------

print("Loading training and testing data...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

TARGET = "ProdTaken"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# ---------------------------------------------------------
# Identify numerical and categorical columns
# ---------------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
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
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ---------------------------------------------------------
# Define model
# ---------------------------------------------------------

model = RandomForestClassifier(
    random_state=42
)


# ---------------------------------------------------------
# Create pipeline
# ---------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)


# ---------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5]
}


# ---------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------

mlflow.set_experiment("Tourism_Product_Prediction")


# ---------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------

print("\nStarting hyperparameter tuning...")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)


with mlflow.start_run():

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print("\nBest CV score:")
    print(grid_search.best_score_)


    # -----------------------------------------------------
    # Log best parameters
    # -----------------------------------------------------

    mlflow.log_params(grid_search.best_params_)

    mlflow.log_metric(
        "best_cv_accuracy",
        grid_search.best_score_
    )


    # -----------------------------------------------------
    # Evaluate best model
    # -----------------------------------------------------

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

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


    # -----------------------------------------------------
    # Log evaluation metrics
    # -----------------------------------------------------

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)


    # -----------------------------------------------------
    # Print evaluation results
    # -----------------------------------------------------

    print("\nTest Results")
    print("-------------------------")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)


    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    joblib.dump(best_model, MODEL_PATH)

    print("\nBest model saved to:")
    print(MODEL_PATH)


    # -----------------------------------------------------
    # Log model to MLflow
    # -----------------------------------------------------

    mlflow.sklearn.log_model(
        best_model,
        "tourism_model"
    )

print("\nTraining completed successfully.")