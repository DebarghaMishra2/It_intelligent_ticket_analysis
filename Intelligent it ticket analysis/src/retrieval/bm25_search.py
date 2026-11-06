
from pathlib import Path
import re

import pandas as pd
from rank_bm25 import BM25Okapi

from src.preprocessing.text_cleaner import (
    build_search_text,
    normalize_ticket_text
)


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "classification_train.csv"
)

GENERIC_ONLY_TERMS = {"server"}

TOP_K = 5


# --------------------------------------------------
# 2. TOKENIZATION
# --------------------------------------------------

def tokenize(text):

    if not isinstance(text, str):
        return []

    return re.findall(
        r"[a-z0-9_]+",
        text.lower()
    )


# --------------------------------------------------
# 3. LOAD AND FILTER THE RETRIEVAL CORPUS
# --------------------------------------------------

def load_retrieval_corpus():

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_DATA_PATH}"
        )

    df = pd.read_csv(TRAIN_DATA_PATH)

    original_count = len(df)

    # Create normalized searchable text.
    df["search_text"] = df.apply(
        build_search_text,
        axis=1
    )

    # Tokenize descriptions.
    df["_tokens"] = df["search_text"].apply(tokenize)

    # Find empty records.
    empty_mask = df["_tokens"].apply(
        lambda tokens: len(tokens) == 0
    )

    # Find records containing only generic terms.
    generic_mask = df["_tokens"].apply(
        lambda tokens:
        bool(tokens)
        and set(tokens).issubset(GENERIC_ONLY_TERMS)
    )

    excluded_mask = empty_mask | generic_mask

    excluded_count = int(excluded_mask.sum())

    df = df.loc[~excluded_mask].copy()

    df.reset_index(drop=True, inplace=True)

    # Validate corpus.
    if df["ticket_id"].duplicated().any():
        raise ValueError(
            "Duplicate ticket IDs detected."
        )

    if df["search_text"].str.strip().eq("").any():
        raise ValueError(
            "Empty searchable records remain."
        )

    print("\nBM25 CORPUS PREPARATION")

    print("Original training records:", original_count)

    print("Excluded records:", excluded_count)

    print("Historical tickets:", len(df))

    return df


# --------------------------------------------------
# 4. BUILD BM25 INDEX
# --------------------------------------------------

def build_bm25_index():

    df = load_retrieval_corpus()

    print("\nBuilding BM25 index...")

    tokenized_corpus = df["search_text"].apply(
        tokenize
    ).tolist()

    bm25 = BM25Okapi(tokenized_corpus)

    print("\nBM25 index built successfully!")

    print("Indexed historical tickets:", len(df))

    return bm25, df


# --------------------------------------------------
# 5. SEARCH HISTORICAL TICKETS
# --------------------------------------------------

def search_bm25(
    query,
    bm25,
    df,
    top_k=TOP_K
):

    # Clean the user's query.
    cleaned_query = normalize_ticket_text(query)

    query_tokens = tokenize(cleaned_query)

    if not query_tokens:
        print("\nNo meaningful search terms found.")
        return []

    # Calculate BM25 relevance scores.
    scores = bm25.get_scores(query_tokens)

    # Rank results by BM25 score.
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    results = []

    for index in ranked_indices:

        if len(results) >= top_k:
            break

        score = float(scores[index])

        # Ignore tickets with no query-term contribution.
        if score == 0:
            continue

        row = df.iloc[index]

        result = {
            "ticket_id": row["ticket_id"],
            "category": row["category"],
            "description": row["description"],
            "search_text": row["search_text"],
            "bm25_score": score
        }

        results.append(result)

    return results


# --------------------------------------------------
# 6. DISPLAY RESULTS
# --------------------------------------------------

def display_results(results):

    print("\nTOP", TOP_K, "BM25 RESULTS")

    if not results:
        print("No matching historical tickets found.")
        return

    for rank, result in enumerate(results, start=1):

        print("\nResult:", rank)

        print("Ticket ID:", result["ticket_id"])

        print("Category:", result["category"])

        print(
            "BM25 score:",
            round(result["bm25_score"], 4)
        )

        print(
            "Original:",
            result["description"]
        )

        print(
            "Cleaned:",
            result["search_text"]
        )


# --------------------------------------------------
# 7. MULTILINE USER INPUT
# --------------------------------------------------

def get_user_query():

    print("\nEnter IT ticket description.")

    print("Press ENTER on an empty line when finished.\n")

    lines = []

    while True:

        line = input()

        if line == "":
            break

        lines.append(line)

    return "\n".join(lines).strip()


# --------------------------------------------------
# 8. MAIN FUNCTION
# --------------------------------------------------

def main():

    bm25, df = build_bm25_index()

    query = get_user_query()

    if not query:
        print("Empty query. Exiting.")
        return

    results = search_bm25(
        query,
        bm25,
        df,
        TOP_K
    )

    display_results(results)


if __name__ == "__main__":
    main()
