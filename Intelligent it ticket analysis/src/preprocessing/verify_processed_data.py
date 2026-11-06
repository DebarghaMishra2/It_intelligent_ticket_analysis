from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

for file in DATA_DIR.glob("*.csv"):

    print("\n" + "=" * 50)
    print("FILE:", file.name)
    print("=" * 50)

    df = pd.read_csv(file)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSample records:")
    print(df[[
        "ticket_id",
        "source_dataset",
        "description",
        "category"
    ]].head(3).to_string(index=False))

    assert df["ticket_id"].is_unique
    assert df["description"].notna().all()
    assert df["category"].notna().all()

print("\nVerification completed.")