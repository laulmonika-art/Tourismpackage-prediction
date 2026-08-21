
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ---------------------------------------------------------
# Remove unnecessary columns
# ---------------------------------------------------------

# CustomerID is an identifier and should not be used for prediction.
# Unnamed: 0 is usually a saved pandas index and is also unnecessary.

columns_to_remove = [
    "CustomerID",
    "Unnamed: 0"
]

columns_to_remove = [
    column for column in columns_to_remove
    if column in df.columns
]

df = df.drop(columns=columns_to_remove)

print("Removed columns:", columns_to_remove)
print("Dataset shape after cleaning:", df.shape)


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

TARGET = "ProdTaken"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ---------------------------------------------------------
# Train-test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# Save split files locally
# ---------------------------------------------------------

X_train_path = OUTPUT_DIR / "Xtrain.csv"
X_test_path = OUTPUT_DIR / "Xtest.csv"
y_train_path = OUTPUT_DIR / "ytrain.csv"
y_test_path = OUTPUT_DIR / "ytest.csv"

X_train.to_csv(X_train_path, index=False)
X_test.to_csv(X_test_path, index=False)
y_train.to_csv(y_train_path, index=False)
y_test.to_csv(y_test_path, index=False)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\nData preparation completed successfully.")

print("Training features:", X_train.shape)
print("Testing features :", X_test.shape)
print("Training target  :", y_train.shape)
print("Testing target   :", y_test.shape)

print("\nFiles created:")

print(X_train_path)
print(X_test_path)
print(y_train_path)
print(y_test_path)
