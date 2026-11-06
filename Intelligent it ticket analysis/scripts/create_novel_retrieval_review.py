
from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.text_cleaner import build_search_text

from src.retrieval.bm25_search import (
    build_bm25_index,
    search_bm25
)

from src.retrieval.semantic_search import (
    load_semantic_index,
    load_embedding_model,
    search_semantic
)

from src.retrieval.hybrid_search import (
    reciprocal_rank_fusion,
    validate_shared_corpus
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TRAIN_PATH = (
    ROOT / "data" / "processed" / "classification_train.csv"
)

TEST_PATH = (
    ROOT / "data" / "processed" / "classification_test.csv"
)

REPORT_DIR = ROOT / "reports"

AUDIT_PATH = (
    REPORT_DIR / "novel_query_eligibility_audit.csv"
)

CANDIDATE_PATH = (
    REPORT_DIR / "novel_retrieval_review_candidates.csv"
)

RANDOM_SEED = 42

MAX_QUERIES_PER_CATEGORY = 3

NEAR_DUPLICATE_THRESHOLD = 0.95

RETRIEVAL_K = 50

REVIEW_K = 5

RRF_K = 60

GENERIC_ONLY_TERMS = {"server"}


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


def safe_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# 4. LOAD TRAINING AND TEST DATA
# ============================================================

def load_datasets():

    print("\nLoading datasets...")

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
                f"{name} data is missing columns: {missing}"
            )

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
# 5. PREPARE THE ELIGIBLE TRAINING CORPUS
# ============================================================

def prepare_training_corpus(train):

    token_lists = train["search_text"].apply(tokenize)

    empty_mask = token_lists.apply(
        lambda tokens: len(tokens) == 0
    )

    generic_mask = token_lists.apply(
        lambda tokens:
        bool(tokens)
        and set(tokens).issubset(GENERIC_ONLY_TERMS)
    )

    excluded_mask = empty_mask | generic_mask

    eligible = train.loc[~excluded_mask].copy()

    eligible = eligible.reset_index(drop=True)

    if eligible["ticket_id"].duplicated().any():
        raise ValueError(
            "Duplicate training ticket IDs detected."
        )

    if eligible["search_text"].str.strip().eq("").any():
        raise ValueError(
            "Empty searchable descriptions remain."
        )

    print("\nTRAINING CORPUS")

    print("Original records:", len(train))
    print("Excluded records:", int(excluded_mask.sum()))
    print("Eligible records:", len(eligible))

    return eligible


# ============================================================
# 6. AUDIT EXACT AND NEAR DUPLICATES
# ============================================================

def audit_test_queries(train, test):

    print("\nStarting cross-split duplicate audit...")

    eligible_train = prepare_training_corpus(train)

    # Remove empty and generic-only test descriptions.
    test_tokens = test["search_text"].apply(tokenize)

    empty_mask = test_tokens.apply(
        lambda tokens: len(tokens) == 0
    )

    generic_mask = test_tokens.apply(
        lambda tokens:
        bool(tokens)
        and set(tokens).issubset(GENERIC_ONLY_TERMS)
    )

    valid_mask = ~(empty_mask | generic_mask)

    valid_test = test.loc[valid_mask].copy()

    print("Valid test queries:", len(valid_test))

    if valid_test.empty:
        raise ValueError("No valid test queries available.")

    # Character n-gram TF-IDF for duplicate screening.
    train_texts = eligible_train["search_text"].tolist()

    test_texts = valid_test["search_text"].tolist()

    print("\nBuilding character-level TF-IDF features...")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        sublinear_tf=True,
        min_df=1
    )

    train_matrix = vectorizer.fit_transform(train_texts)

    test_matrix = vectorizer.transform(test_texts)

    print("\nCalculating similarities...")

    similarities = cosine_similarity(
        test_matrix,
        train_matrix,
        dense_output=True
    )

    # Exact normalized-text lookup.
    exact_lookup = {}

    for index, text in enumerate(train_texts):

        exact_lookup.setdefault(
            text.strip(),
            []
        ).append(index)

    audit_rows = []

    for query_index, (_, query) in enumerate(
        valid_test.iterrows()
    ):

        query_text = str(query["search_text"]).strip()

        scores = similarities[query_index]

        ranked = np.argsort(scores)[::-1]

        exact_indices = exact_lookup.get(
            query_text,
            []
        )

        best_index = int(ranked[0])

        best_score = float(scores[best_index])

        best_row = eligible_train.iloc[best_index]

        top_matches = []

        for train_index in ranked[:3]:

            train_row = eligible_train.iloc[
                int(train_index)
            ]

            top_matches.append({
                "ticket_id": str(train_row["ticket_id"]),
                "category": train_row["category"],
                "similarity": float(scores[train_index]),
                "description": safe_text(
                    train_row["description"]
                )
            })

        # Conservative eligibility decisions.
        if exact_indices:

            status = "EXCLUDE_EXACT_DUPLICATE"

        elif best_score >= NEAR_DUPLICATE_THRESHOLD:

            status = "HOLD_FOR_MANUAL_REVIEW"

        else:

            status = "ELIGIBLE_NOVEL"

        audit_rows.append({

            "query_source_ticket_id": query["ticket_id"],

            "query_category": query["category"],

            "query_text": safe_text(query["description"]),

            "query_search_text": query_text,

            "exact_match_count": len(exact_indices),

            "best_match_ticket_id": best_row["ticket_id"],

            "best_match_category": best_row["category"],

            "best_match_similarity": best_score,

            "eligibility_status": status,

            "top_matches": str(top_matches)

        })

        if (query_index + 1) % 100 == 0:

            print(
                f"Audited {query_index + 1}/{len(valid_test)} queries"
            )

    # Include skipped records in the audit report.
    for _, query in test.loc[~valid_mask].iterrows():

        audit_rows.append({

            "query_source_ticket_id": query["ticket_id"],

            "query_category": query["category"],

            "query_text": safe_text(query["description"]),

            "query_search_text": query["search_text"],

            "exact_match_count": 0,

            "best_match_ticket_id": "",

            "best_match_category": "",

            "best_match_similarity": np.nan,

            "eligibility_status": "SKIPPED_EMPTY_OR_GENERIC",

            "top_matches": ""

        })

    audit_df = pd.DataFrame(audit_rows)

    return eligible_train, audit_df


# ============================================================
# 7. SELECT NOVEL QUERIES
# ============================================================

def select_novel_queries(audit_df, test):

    print("\nSELECTING NOVEL EVALUATION QUERIES")

    eligible_ids = set(
        audit_df.loc[
            audit_df["eligibility_status"] == "ELIGIBLE_NOVEL",
            "query_source_ticket_id"
        ].astype(str)
    )

    eligible_test = test[
        test["ticket_id"].astype(str).isin(eligible_ids)
    ].copy()

    # Avoid selecting duplicate test descriptions within a category.
    eligible_test = eligible_test.drop_duplicates(
        subset=["category", "search_text"]
    )

    rng = np.random.default_rng(RANDOM_SEED)

    selected_rows = []

    category_summary = []

    for category, group in eligible_test.groupby(
        "category",
        sort=True
    ):

        group = group.sort_values("ticket_id")

        n = min(
            MAX_QUERIES_PER_CATEGORY,
            len(group)
        )

        chosen_positions = rng.choice(
            len(group),
            size=n,
            replace=False
        )

        chosen = group.iloc[
            sorted(chosen_positions)
        ]

        selected_rows.extend(
            chosen.to_dict("records")
        )

        category_summary.append({

            "category": category,

            "eligible_unique_queries": len(group),

            "selected_queries": n

        })

    selected = pd.DataFrame(selected_rows)

    if selected.empty:
        raise ValueError(
            "No novel queries are eligible for evaluation."
        )

    selected = selected.reset_index(drop=True)

    category_summary = pd.DataFrame(category_summary)

    print("\nCategory selection summary:")

    print(category_summary.to_string(index=False))

    print("\nTotal selected novel queries:", len(selected))

    return selected


# ============================================================
# 8. INITIALIZE RETRIEVAL SYSTEMS
# ============================================================

def prepare_retrieval_systems():

    print("\nBuilding BM25...")

    bm25, bm25_df = build_bm25_index()

    print("\nLoading semantic index...")

    embeddings, metadata = load_semantic_index()

    validate_shared_corpus(
        bm25_df,
        metadata
    )

    model = load_embedding_model()

    return bm25, bm25_df, embeddings, metadata, model


# ============================================================
# 9. GENERATE CANDIDATES
# ============================================================

def generate_query_candidates(
    query,
    bm25,
    bm25_df,
    embeddings,
    metadata,
    model
):

    query_text = safe_text(query["description"])

    bm25_results = search_bm25(
        query_text,
        bm25,
        bm25_df,
        top_k=RETRIEVAL_K
    )

    semantic_results = search_semantic(
        query_text,
        model,
        embeddings,
        metadata,
        top_k=RETRIEVAL_K
    )

    hybrid_results = reciprocal_rank_fusion(
        bm25_results,
        semantic_results,
        rrf_k=RRF_K,
        top_k=RETRIEVAL_K
    )

    bm25_map = {
        str(item["ticket_id"]): item
        for item in bm25_results
    }

    semantic_map = {
        str(item["ticket_id"]): item
        for item in semantic_results
    }

    hybrid_map = {
        str(item["ticket_id"]): item
        for item in hybrid_results
    }

    bm25_ranks = {
        ticket_id: rank
        for rank, ticket_id in enumerate(
            bm25_map,
            start=1
        )
    }

    semantic_ranks = {
        ticket_id: rank
        for rank, ticket_id in enumerate(
            semantic_map,
            start=1
        )
    }

    hybrid_ranks = {
        ticket_id: rank
        for rank, ticket_id in enumerate(
            hybrid_map,
            start=1
        )
    }

    candidate_ids = set(
        list(bm25_map)[:REVIEW_K]
        + list(semantic_map)[:REVIEW_K]
        + list(hybrid_map)[:REVIEW_K]
    )

    candidate_ids = sorted(
        candidate_ids,
        key=lambda ticket_id: (
            hybrid_ranks.get(ticket_id, 999),
            ticket_id
        )
    )

    rows = []

    for ticket_id in candidate_ids:

        b = bm25_map.get(ticket_id)

        s = semantic_map.get(ticket_id)

        h = hybrid_map.get(ticket_id)

        base = b or s or h

        rows.append({

            "ticket_id": ticket_id,

            "category": base["category"],

            "description": base["description"],

            "search_text": base["search_text"],

            "bm25_rank": bm25_ranks.get(ticket_id),

            "bm25_score": b["bm25_score"] if b else None,

            "semantic_rank": semantic_ranks.get(ticket_id),

            "semantic_score": s["semantic_score"] if s else None,

            "hybrid_rank": hybrid_ranks.get(ticket_id),

            "rrf_score": h["rrf_score"] if h else None,

            "is_relevant": "",

            "review_notes": ""

        })

    return rows


# ============================================================
# 10. MAIN CANDIDATE GENERATION
# ============================================================

def main():

    print("\nIT-SENTINEL NOVEL-TICKET RETRIEVAL BENCHMARK")

    print("Project root:", ROOT)

    train, test = load_datasets()

    eligible_train, audit_df = audit_test_queries(
        train,
        test
    )

    # Save audit before candidate generation.
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    audit_df.to_csv(
        AUDIT_PATH,
        index=False
    )

    print("\nNOVEL QUERY AUDIT")

    print(
        audit_df["eligibility_status"]
        .value_counts()
        .to_string()
    )

    selected_queries = select_novel_queries(
        audit_df,
        test
    )

    # Initialize retrieval systems.
    (
        bm25,
        bm25_df,
        embeddings,
        metadata,
        model
    ) = prepare_retrieval_systems()

    # Ensure the same corpus was audited and indexed.
    audited_ids = set(
        eligible_train["ticket_id"].astype(str)
    )

    indexed_ids = set(
        metadata["ticket_id"].astype(str)
    )

    if audited_ids != indexed_ids:
        raise ValueError(
            "Audited training corpus and retrieval index differ."
        )

    all_rows = []

    # Generate candidates for each novel query.
    for index, (_, query) in enumerate(
        selected_queries.iterrows(),
        start=1
    ):

        query_id = f"NOV-{index:03d}"

        print("\n" + "=" * 60)

        print("Processing:", query_id)

        print("Category:", query["category"])

        print("Ticket:", query["ticket_id"])

        candidates = generate_query_candidates(
            query,
            bm25,
            bm25_df,
            embeddings,
            metadata,
            model
        )

        print("Candidates:", len(candidates))

        for candidate in candidates:

            candidate["query_id"] = query_id

            candidate["query_category"] = query["category"]

            candidate["query_source_ticket_id"] = query["ticket_id"]

            candidate["query_text"] = safe_text(
                query["description"]
            )

            all_rows.append(candidate)

    # Save review file.
    candidate_df = pd.DataFrame(all_rows)

    columns = [
        "query_id",
        "query_category",
        "query_source_ticket_id",
        "query_text",
        "ticket_id",
        "category",
        "description",
        "search_text",
        "bm25_rank",
        "bm25_score",
        "semantic_rank",
        "semantic_score",
        "hybrid_rank",
        "rrf_score",
        "is_relevant",
        "review_notes"
    ]

    candidate_df = candidate_df[columns]

    candidate_df.to_csv(
        CANDIDATE_PATH,
        index=False
    )

    print("\n" + "=" * 60)

    print("NOVEL RETRIEVAL CANDIDATES CREATED")

    print("Selected queries:", len(selected_queries))

    print("Candidate rows:", len(candidate_df))

    print("\nEligibility audit:", AUDIT_PATH)

    print("Candidate review file:", CANDIDATE_PATH)

    print("\nRelevance labels are intentionally blank.")

    print("Review them before calculating retrieval metrics.")

    print("\nSUCCESS!")


if __name__ == "__main__":

    main()
