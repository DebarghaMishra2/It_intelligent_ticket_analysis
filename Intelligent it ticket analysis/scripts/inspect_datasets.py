
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"


def inspect_file(file_path):

    print("\n" + "=" * 70)
    print("FILE:", file_path.name)
    print("=" * 70)

    df = pd.read_csv(file_path, low_memory=False)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(
        df.isnull().sum().sort_values(ascending=False)
    )

    print("\nDuplicate records:")
    print(df.duplicated().sum())

    print("\nFirst three records:")
    print(df.head(3).to_string(index=False))

    if "label_name" in df.columns:
        print("\nCategory distribution:")
        print(df["label_name"].value_counts())

    if "issue_priority" in df.columns:
        print("\nPriority distribution:")
        print(df["issue_priority"].value_counts())


if __name__ == "__main__":

    files = list(DATA_DIR.rglob("*.csv"))

    if not files:
        print("No CSV files found. Download datasets first.")

    for file in files:
        inspect_file(file)
