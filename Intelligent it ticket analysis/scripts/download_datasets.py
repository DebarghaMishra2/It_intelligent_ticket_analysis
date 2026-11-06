
from pathlib import Path
from datasets import load_dataset


# Project root directory
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"


def download_and_save(dataset_name, output_dir, config=None):

    print("\n" + "=" * 60)
    print(f"Downloading: {dataset_name}")
    print("=" * 60)

    # Download dataset
    if config:
        dataset = load_dataset(dataset_name, config)
    else:
        dataset = load_dataset(dataset_name)

    output_path = RAW_DIR / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all available splits
    for split_name, split_data in dataset.items():

        df = split_data.to_pandas()

        # Decode classification labels into readable names
        if "label" in df.columns:
            label_feature = split_data.features.get("label")

            if hasattr(label_feature, "int2str"):
                df["label_name"] = df["label"].apply(
                    lambda x: label_feature.int2str(int(x))
                    if x is not None else None
                )

        file_path = output_path / f"{split_name}.csv"

        df.to_csv(file_path, index=False)

        print(f"\nSaved: {file_path}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")

    print("\nDownload completed successfully!")


if __name__ == "__main__":

    # Dataset A: IT Support Ticket Classification
    download_and_save(
        dataset_name="tasksource/it-support-tickets",
        output_dir="classification"
    )

    # Dataset B1: Reporter messages from Mendeley
    download_and_save(
        dataset_name="tasksource/help-desk-tickets",
        config="default",
        output_dir="helpdesk/messages"
    )

    # Dataset B2: Mendeley operational metadata
    download_and_save(
        dataset_name="tasksource/help-desk-tickets",
        config="issues",
        output_dir="helpdesk/issues"
    )

    print("\nALL DATASETS DOWNLOADED!")
