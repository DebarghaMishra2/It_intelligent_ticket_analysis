
from pathlib import Path
import pandas as pd


# -------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "classification"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# UNIFIED TICKET SCHEMA
# -------------------------------------------------

SCHEMA_COLUMNS = [
    "ticket_id",
    "source_record_id",
    "source_dataset",
    "record_origin",
    "source_split",
    "title",
    "description",
    "symptoms",
    "application",
    "service",
    "category",
    "subcategory",
    "priority",
    "intent",
    "impact",
    "urgency",
    "status",
    "assignment_group",
    "environment",
    "created_at",
    "resolution_text"
]


def normalize_classification(split_name):

    file_path = RAW_DIR / f"{split_name}.csv"

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)

    print(f"\nProcessing {split_name}")
    print("Original shape:", df.shape)

    # Validate required text field
    if "text" not in df.columns:
        raise ValueError(
            f"'text' column missing. Available: {df.columns.tolist()}"
        )

    # Preserve actual category names
    if "label_name" in df.columns:
        categories = df["label_name"]

    elif "label" in df.columns:
        if pd.api.types.is_numeric_dtype(df["label"]):
            raise ValueError(
                "Numeric labels detected without label_name. "
                "Re-download with the label mapping preserved."
            )
        categories = df["label"]

    else:
        raise ValueError("No classification label found.")

    # Build normalized dataset
    normalized = pd.DataFrame(
        index=df.index,
        columns=SCHEMA_COLUMNS
    )

    # Internal generated IDs; these are NOT original source IDs
    normalized["ticket_id"] = [
        f"ITS-{split_name.upper()}-{i:06d}"
        for i in range(len(df))
    ]

    normalized["source_record_id"] = pd.NA

    normalized["source_dataset"] = (
        "tasksource/it-support-tickets"
    )

    normalized["record_origin"] = "public_deidentified"

    normalized["source_split"] = split_name

    normalized["description"] = df["text"].astype("string").str.strip()

    normalized["category"] = categories.astype("string")

    # Remove records without usable text or category
    normalized = normalized.dropna(
        subset=["description", "category"]
    )

    normalized = normalized[
        (normalized["description"] != "") &
        (normalized["category"] != "")
    ].copy()

    # Save processed dataset
    output_path = (
        OUTPUT_DIR / f"classification_{split_name}.csv"
    )

    normalized.to_csv(output_path, index=False)

    print("Normalized shape:", normalized.shape)
    print("Category distribution:")
    print(normalized["category"].value_counts())

    print("\nSaved:", output_path)


if __name__ == "__main__":

    normalize_classification("train")
    normalize_classification("test")

    print("\nDataset normalization completed!")
