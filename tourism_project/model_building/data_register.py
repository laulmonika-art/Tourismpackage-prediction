
import pandas as pd
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"


# Expected columns
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]


def register_dataset():

    # Check dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Read CSV
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Dataset path : {DATA_PATH}")
    print(f"Rows         : {df.shape[0]}")
    print(f"Columns      : {df.shape[1]}")

    # Check expected columns
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    print("\nAll expected columns are present.")

    # Short summary
    print("\nDataset summary:")
    print(df.info())

    print("\nTarget distribution:")
    if "ProdTaken" in df.columns:
        print(df["ProdTaken"].value_counts())

    print("\nFirst 5 records:")
    print(df.head())


if __name__ == "__main__":
    register_dataset()
