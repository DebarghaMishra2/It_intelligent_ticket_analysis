
from pathlib import Path
import sys
import re
import json
from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.text_cleaner import build_search_text


TRAIN_PATH = (
    ROOT / "data" / "processed" / "classification_train.csv"
)

TEST_PATH = (
    ROOT / "data" / "processed" / "classification_test.csv"
)

SUMMARY_PATH = (
    ROOT / "reports" / "train_test_duplicate_audit.csv"
)

DETAIL_PATH = (
    ROOT / "reports" / "train_test_duplicate_audit_top_matches.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

GENERIC_ONLY_TERMS = {"server"}

# Similarity thresholds are review flags, not proof of leakage.
VERY_HIGH_THRESHOLD = 0.95
HIGH_THRESHOLD = 0.85
POSSIBLE_THRESHOLD = 0.75

TOP_MATCHES = 3


# ============================================================
# 3. TEXT UTILITIES
# ============================================================

def tokenize(text):

    if not isinstance(text, str):
        return []

    return re.findall(
        r"[a-z0-9_]+",
        text.lower()
    )


def is_generic_only(text):

    tokens = tokenize(text)

    return (
        bool(tokens)
        and set(tokens).issubset(GENERIC_ONLY_TERMS)
    )


# ============================================================
# 4. LOAD DATASETS
# ============================================================

def load_datasets():

    print("\nLoading training and test datasets...")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)

    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    train = pd.read_csv(TRAIN_PATH)

    test = pd.read_csv(TEST_PATH)

    required = {
        "ticket_id",
        "category",
        "description"
    }

    for name, df in [
        ("Training", train),
        ("Test", test)
    ]:

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} dataset missing columns: {missing}"
            )

    # Use the exact same text-cleaning function.
    train["search_text"] = train.apply(
        build_search_text,
        axis=1
    )

    test["search_text"] = test.apply(
        build_search_text,
        axis=1
    )

    print("Training records:", len(train))
    print("Test records:", len(test))

    return train, test


# ============================================================
# 5. MATCH THE ACTUAL RETRIEVAL CORPUS
# ============================================================

def prepare_training_corpus(train):

    empty_mask = (
        train["search_text"]
        .fillna("")
        .str.strip()
        .eq("")
    )

    generic_mask = train["search_text"].apply(
        is_generic_only
    )

    excluded_mask = empty_mask | generic_mask

    excluded = train.loc[excluded_mask].copy()

    eligible = train.loc[~excluded_mask].copy()

    eligible = eligible.reset_index(drop=True)

    if eligible["ticket_id"].duplicated().any():
        raise ValueError(
            "Duplicate ticket IDs in training dataset."
        )

    print("\nTRAINING CORPUS")

    print("Original training records:", len(train))
    print("Empty records excluded:", int(empty_mask.sum()))
    print("Generic-only records excluded:", int(generic_mask.sum()))
    print("Eligible retrieval records:", len(eligible))

    print("\nExcluded ticket IDs:")

    print(
        excluded["ticket_id"].astype(str).tolist()
    )

    return eligible


# ============================================================
# 6. PREPARE TEST QUERIES
# ============================================================

def prepare_test_queries(test):

    test = test.copy()

    test["_tokens"] = test["search_text"].apply(tokenize)

    test["_skip"] = test.apply(
        lambda row:
        len(row["_tokens"]) == 0
        or set(row["_tokens"]).issubset(GENERIC_ONLY_TERMS),
        axis=1
    )

    print("\nTEST QUERY QUALITY")

    print("Total test records:", len(test))
    print("Empty/generic queries:", int(test["_skip"].sum()))
    print("Queries to audit:", int((~test["_skip"]).sum()))

    return test


# ============================================================
# 7. EXACT DUPLICATE LOOKUP
# ============================================================

def build_exact_lookup(train):

    lookup = defaultdict(list)

    for _, row in train.iterrows():

        text = str(row["search_text"]).strip()

        if text:
            lookup[text].append(row)

    return lookup


# ============================================================
# 8. AUDIT TRAIN-TEST SIMILARITY
# ============================================================

def run_duplicate_audit(train, test):

    print("\nBuilding character-level TF-IDF index...")

    train_texts = train["search_text"].fillna("").tolist()

    test_texts = test["search_text"].fillna("").tolist()

    # Character n-grams work well for:
    # exact repetitions, minor edits and formatting changes.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        sublinear_tf=True,
        min_df=1
    )

    combined_texts = train_texts + test_texts

    matrix = vectorizer.fit_transform(combined_texts)

    train_matrix = matrix[:len(train_texts)]

    test_matrix = matrix[len(train_texts):]

    print("Training TF-IDF shape:", train_matrix.shape)
    print("Test TF-IDF shape:", test_matrix.shape)

    print("\nCalculating cross-split similarity...")

    similarities = cosine_similarity(
        test_matrix,
        train_matrix,
        dense_output=True
    )

    exact_lookup = build_exact_lookup(train)

    summary_rows = []

    detail_rows = []

    for query_index, (_, query_row) in enumerate(
        test.iterrows()
    ):

        query_text = str(query_row["search_text"]).strip()

        query_id = str(query_row["ticket_id"])

        query_category = query_row["category"]

        if bool(query_row["_skip"]):

            summary_rows.append({
                "query_ticket_id": query_id,
                "query_category": query_category,
                "query_description": query_row["description"],
                "query_search_text": query_text,
                "audit_status": "SKIPPED_EMPTY_OR_GENERIC",
                "exact_duplicate_count": 0,
                "best_match_ticket_id": "",
                "best_match_category": "",
                "best_match_similarity": np.nan,
                "audit_flag": "NOT_AUDITED"
            })

            continue

        scores = similarities[query_index]

        ranked_indices = np.argsort(scores)[::-1]

        # Exact matches based on normalized search text.
        exact_matches = exact_lookup.get(
            query_text,
            []
        )

        top_matches = []

        for train_index in ranked_indices[:TOP_MATCHES]:

            train_row = train.iloc[int(train_index)]

            similarity = float(scores[train_index])

            match = {
                "train_ticket_id": str(train_row["ticket_id"]),
                "train_category": train_row["category"],
                "similarity": similarity,
                "train_description": train_row["description"],
                "train_search_text": train_row["search_text"]
            }

            top_matches.append(match)

            detail_rows.append({

                "query_ticket_id": query_id,

                "query_category": query_category,

                "query_search_text": query_text,

                "train_ticket_id": str(train_row["ticket_id"]),

                "train_category": train_row["category"],

                "similarity": similarity,

                "is_exact_match": (
                    query_text == str(train_row["search_text"]).strip()
                ),

                "train_description": train_row["description"],

                "train_search_text": train_row["search_text"]

            })

        best_match = top_matches[0] if top_matches else None

        best_similarity = (
            best_match["similarity"]
            if best_match else 0.0
        )

        # Flag the record for human review.
        if exact_matches:

            audit_flag = "EXACT_TEXT_DUPLICATE"

        elif best_similarity >= VERY_HIGH_THRESHOLD:

            audit_flag = "VERY_HIGH_SIMILARITY_REVIEW"

        elif best_similarity >= HIGH_THRESHOLD:

            audit_flag = "HIGH_SIMILARITY_REVIEW"

        elif best_similarity >= POSSIBLE_THRESHOLD:

            audit_flag = "POSSIBLE_SIMILARITY_REVIEW"

        else:

            audit_flag = "NO_STRONG_SIMILARITY_FLAG"

        summary_rows.append({

            "query_ticket_id": query_id,

            "query_category": query_category,

            "query_description": query_row["description"],

            "query_search_text": query_text,

            "audit_status": "AUDITED",

            "exact_duplicate_count": len(exact_matches),

            "best_match_ticket_id": (
                best_match["train_ticket_id"]
                if best_match else ""
            ),

            "best_match_category": (
                best_match["train_category"]
                if best_match else ""
            ),

            "best_match_similarity": best_similarity,

            "audit_flag": audit_flag,

            "top_matches_json": json.dumps(
                top_matches,
                ensure_ascii=False
            )

        })

        if (query_index + 1) % 100 == 0:

            print(
                f"Audited {query_index + 1}/{len(test)} test records"
            )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(detail_rows)
    )


# ============================================================
# 9. SAVE REPORTS
# ============================================================

def save_reports(summary, details):

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    details.to_csv(
        DETAIL_PATH,
        index=False
    )

    print("\n" + "=" * 65)

    print("TRAIN-TEST DUPLICATE AUDIT COMPLETED")

    print("=" * 65)

    print("\nAudit flag distribution:")

    print(
        summary["audit_flag"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nExact duplicate test tickets:")

    exact = summary[
        summary["audit_flag"] == "EXACT_TEXT_DUPLICATE"
    ]

    print(len(exact))

    print("\nTop similarity review candidates:")

    flagged = summary[
        summary["audit_flag"].isin([
            "VERY_HIGH_SIMILARITY_REVIEW",
            "HIGH_SIMILARITY_REVIEW",
            "POSSIBLE_SIMILARITY_REVIEW"
        ])
    ]

    print(
        flagged[
            [
                "query_ticket_id",
                "query_category",
                "best_match_ticket_id",
                "best_match_category",
                "best_match_similarity",
                "audit_flag"
            ]
        ].head(20).to_string(index=False)
    )

    print("\nSummary report:")
    print(SUMMARY_PATH)

    print("\nTop-match report:")
    print(DETAIL_PATH)

    print("\nIMPORTANT:")
    print(
        "Similarity thresholds are review flags, "
        "not proof of data leakage."
    )

    print(
        "Manually inspect flagged pairs before deciding "
        "whether to exclude training examples."
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("\nIT-SENTINEL: TRAIN-TEST DUPLICATE AUDIT")

    print("Project root:", ROOT)

    train, test = load_datasets()

    eligible_train = prepare_training_corpus(train)

    test = prepare_test_queries(test)

    summary, details = run_duplicate_audit(
        eligible_train,
        test
    )

    save_reports(
        summary,
        details
    )


if __name__ == "__main__":

    main()
