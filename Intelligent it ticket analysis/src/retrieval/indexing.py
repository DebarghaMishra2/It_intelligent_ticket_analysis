
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.preprocessing.text_cleaner import build_search_text


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "classification_train.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"

EMBEDDING_PATH = MODELS_DIR / "ticket_embeddings.npy"

METADATA_PATH = MODELS_DIR / "ticket_embedding_metadata.csv"

CONFIG_PATH = MODELS_DIR / "embedding_config.json"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

BATCH_SIZE = 32

# Tickets containing only "server" are too generic
# to provide meaningful historical retrieval evidence.
GENERIC_ONLY_TERMS = {"server"}


# --------------------------------------------------
# 2. TEXT TOKENIZATION
# --------------------------------------------------

def tokenize(text):
    """Convert normalized text into searchable tokens."""

    if not isinstance(text, str):
        return []

    return re.findall(r"[a-z0-9_]+", text.lower())


# --------------------------------------------------
# 3. LOAD AND FILTER RETRIEVAL CORPUS
# --------------------------------------------------

def load_retrieval_corpus():

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_DATA_PATH}"
        )

    df = pd.read_csv(TRAIN_DATA_PATH)

    required_columns = {
        "ticket_id",
        "description",
        "category"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    original_count = len(df)

    # Generate cleaned searchable text.
    df["search_text"] = df.apply(
        build_search_text,
        axis=1
    )

    # Tokenize searchable text.
    df["_tokens"] = df["search_text"].apply(tokenize)

    # Remove empty records.
    empty_mask = df["_tokens"].apply(
        lambda tokens: len(tokens) == 0
    )

    # Remove records containing only generic terms.
    generic_mask = df["_tokens"].apply(
        lambda tokens:
        bool(tokens)
        and set(tokens).issubset(GENERIC_ONLY_TERMS)
    )

    excluded_mask = empty_mask | generic_mask

    excluded_records = df.loc[excluded_mask].copy()

    df = df.loc[~excluded_mask].copy()

    # Remove temporary token column from metadata.
    df.drop(columns=["_tokens"], inplace=True)

    df.reset_index(drop=True, inplace=True)

    # Validate ticket IDs.
    if df["ticket_id"].duplicated().any():
        raise ValueError(
            "Duplicate ticket IDs detected in retrieval corpus."
        )

    if df["search_text"].isna().any():
        raise ValueError(
            "Missing search_text values detected."
        )

    if df["search_text"].str.strip().eq("").any():
        raise ValueError(
            "Empty searchable records remain after filtering."
        )

    print("\nRETRIEVAL CORPUS PREPARATION")

    print("Original training records:", original_count)
    print("Empty records excluded:", int(empty_mask.sum()))
    print("Generic-only records excluded:", int(generic_mask.sum()))
    print("Total excluded:", int(excluded_mask.sum()))
    print("Records to index:", len(df))

    if not excluded_records.empty:
        print("\nExcluded ticket IDs:")
        print(
            excluded_records["ticket_id"].tolist()
        )

    return df


# --------------------------------------------------
# 4. GENERATE SEMANTIC EMBEDDINGS
# --------------------------------------------------

def create_embeddings(df):

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    texts = df["search_text"].tolist()

    print("\nGenerating semantic embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.shape[0] != len(df):
        raise ValueError(
            "Embedding count does not match metadata count."
        )

    print("\nEmbedding shape:", embeddings.shape)

    return embeddings


# --------------------------------------------------
# 5. SAVE INDEX AND METADATA
# --------------------------------------------------

def save_index(df, embeddings):

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Preserve all original ticket fields and add search_text.
    df.to_csv(
        METADATA_PATH,
        index=False
    )

    np.save(
        EMBEDDING_PATH,
        embeddings
    )

    config = {
        "model_name": MODEL_NAME,
        "embedding_dimensions": int(embeddings.shape[1]),
        "normalize_embeddings": True,
        "num_records": int(len(df)),
        "batch_size": BATCH_SIZE,
        "corpus_file": str(TRAIN_DATA_PATH.name),
        "embedding_file": EMBEDDING_PATH.name,
        "metadata_file": METADATA_PATH.name
    }

    with open(CONFIG_PATH, "w") as file:
        json.dump(
            config,
            file,
            indent=4
        )

    print("\nSEMANTIC INDEX CREATED")

    print("Tickets indexed:", len(df))

    print("Embedding dimensions:", embeddings.shape)

    print("Original descriptions preserved.")

    print("Metadata saved:", METADATA_PATH)

    print("Embeddings saved:", EMBEDDING_PATH)

    print("Configuration saved:", CONFIG_PATH)


# --------------------------------------------------
# 6. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\nLoading historical tickets...")

    df = load_retrieval_corpus()

    print("\nLoading and generating embeddings...")

    embeddings = create_embeddings(df)

    save_index(
        df,
        embeddings
    )

    print("\nSUCCESS: Semantic indexing completed.")


if __name__ == "__main__":
    main()
