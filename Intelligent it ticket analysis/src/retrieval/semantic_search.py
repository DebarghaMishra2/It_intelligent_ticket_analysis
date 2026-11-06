
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.retrieval.indexing import (
    EMBEDDING_PATH,
    METADATA_PATH,
    MODEL_NAME
)

from src.preprocessing.text_cleaner import normalize_ticket_text


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

TOP_K = 5


# --------------------------------------------------
# 2. LOAD SEMANTIC INDEX
# --------------------------------------------------

def load_semantic_index():

    print("\nLoading semantic index...")

    if not EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {EMBEDDING_PATH}. "
            "Run python -m src.retrieval.indexing first."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    embeddings = np.load(EMBEDDING_PATH)

    metadata = pd.read_csv(METADATA_PATH)

    # Verify the index structure.
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            "Embedding count does not match metadata count."
        )

    if embeddings.shape[1] != 384:
        raise ValueError(
            f"Unexpected embedding dimension: {embeddings.shape[1]}"
        )

    if metadata["ticket_id"].duplicated().any():
        raise ValueError(
            "Duplicate ticket IDs found in semantic metadata."
        )

    # Verify normalization.
    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            "Stored embeddings are not normalized."
        )

    print("Semantic index loaded successfully.")

    print("Historical tickets:", len(metadata))

    print("Embedding dimensions:", embeddings.shape)

    return embeddings, metadata


# --------------------------------------------------
# 3. LOAD SENTENCE TRANSFORMER
# --------------------------------------------------

def load_embedding_model():

    print("\nLoading Sentence Transformer...")

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded:", MODEL_NAME)

    return model


# --------------------------------------------------
# 4. SEMANTIC SEARCH
# --------------------------------------------------

def search_semantic(
    query,
    model,
    embeddings,
    metadata,
    top_k=TOP_K
):

    cleaned_query = normalize_ticket_text(query)

    if not cleaned_query.strip():
        print("\nNo meaningful search terms found.")
        return []

    # Generate embedding for the user query.
    query_embedding = model.encode(
        cleaned_query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    # Cosine similarity.
    # Both document and query vectors are normalized,
    # so their dot product equals cosine similarity.
    scores = embeddings @ query_embedding

    # Get highest-scoring results.
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in ranked_indices:

        row = metadata.iloc[index]

        results.append({
            "ticket_id": row["ticket_id"],
            "category": row["category"],
            "description": row["description"],
            "search_text": row["search_text"],
            "semantic_score": float(scores[index])
        })

    return results


# --------------------------------------------------
# 5. DISPLAY RESULTS
# --------------------------------------------------

def display_results(results):

    print("\nTOP", TOP_K, "SEMANTIC RESULTS")

    if not results:
        print("No historical tickets found.")
        return

    for rank, result in enumerate(results, start=1):

        print("\nResult:", rank)

        print("Ticket ID:", result["ticket_id"])

        print("Category:", result["category"])

        print(
            "Cosine similarity:",
            round(result["semantic_score"], 4)
        )

        print(
            "Description:",
            result["description"]
        )

        print(
            "Cleaned:",
            result["search_text"]
        )


# --------------------------------------------------
# 6. MULTILINE USER INPUT
# --------------------------------------------------

def get_user_query():

    print("\nEnter IT support ticket.")

    print("Press ENTER on an empty line when finished.\n")

    lines = []

    while True:

        line = input()

        if line == "":
            break

        lines.append(line)

    return "\n".join(lines).strip()


# --------------------------------------------------
# 7. MAIN FUNCTION
# --------------------------------------------------

def main():

    embeddings, metadata = load_semantic_index()

    model = load_embedding_model()

    query = get_user_query()

    if not query:
        print("Empty query. Exiting.")
        return

    results = search_semantic(
        query,
        model,
        embeddings,
        metadata,
        TOP_K
    )

    display_results(results)


if __name__ == "__main__":
    main()
