import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input dataset
DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data():

    # Load dataset
    print(f"Loading dataset from: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Original dataset shape: {df.shape}")

    # Remove unnecessary columns
    columns_to_remove = ["CustomerID", "Unnamed: 0"]

    existing_columns = [
        col for col in columns_to_remove
        if col in df.columns
    ]

    df = df.drop(columns=existing_columns)

    print(f"Dataset shape after removing columns: {df.shape}")

    # Split data into training and testing sets
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["ProdTaken"]
    )

    # Save locally
    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nData preparation completed.")
    print(f"Training data: {train_path}")
    print(f"Testing data:  {test_path}")

    print(f"\nTraining shape: {train_df.shape}")
    print(f"Testing shape:  {test_df.shape}")


if __name__ == "__main__":
    prepare_data()