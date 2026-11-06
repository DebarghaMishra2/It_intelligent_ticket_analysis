
from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# File location:
# project_root/scripts/create_retrieval_review.py
#
# parent     = project_root/scripts
# parent.parent = project_root

ROOT = Path(__file__).resolve().parent.parent

# Ensure project root is available for imports from src.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# 2. IMPORT PROJECT MODULES
# ============================================================

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
# 3. CONFIGURATION
# ============================================================

TEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "classification_test.csv"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "retrieval_review_candidates_v2.csv"
)

RANDOM_SEED = 42

# Retrieve more candidates internally.
RETRIEVAL_K = 50

# Show the union of the top five from each method for review.
REVIEW_K = 5

RRF_K = 60

EXPECTED_CATEGORIES = {
    "Active Directory",
    "Computer-Services",
    "EOL",
    "Fileservice",
    "O365",
    "Software",
    "Support general"
}


# ============================================================
# 4. TEXT UTILITIES
# ============================================================

def tokenize(text):
    """Tokenize text for basic quality checks."""

    if not isinstance(text, str):
        return []

    return re.findall(
        r"[a-z0-9_]+",
        text.lower()
    )


def safe_text(value):
    """Convert missing values to empty strings."""

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# 5. SELECT EVALUATION QUERIES FROM TEST DATA
# ============================================================

def select_evaluation_queries():

    print("\nLoading classification test dataset...")

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}\n"
            "Check that classification_test.csv exists inside "
            "data/processed."
        )

    df = pd.read_csv(TEST_PATH)

    required_columns = {
        "ticket_id",
        "category",
        "description"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in test dataset: "
            f"{missing_columns}"
        )

    original_count = len(df)

    # Create cleaned text for query quality checks.
    df["query_search_text"] = df.apply(
        build_search_text,
        axis=1
    )

    df["_tokens"] = df["query_search_text"].apply(
        tokenize
    )

    # Remove empty descriptions.
    empty_mask = df["_tokens"].apply(
        lambda tokens: len(tokens) == 0
    )

    # Remove descriptions containing only generic terms.
    generic_only = {"server"}

    generic_mask = df["_tokens"].apply(
        lambda tokens:
        bool(tokens)
        and set(tokens).issubset(generic_only)
    )

    valid_mask = ~(empty_mask | generic_mask)

    df = df.loc[valid_mask].copy()

    if df.empty:
        raise ValueError(
            "No suitable evaluation queries found in the test dataset."
        )

    # Select one representative query from every category.
    rng = np.random.default_rng(RANDOM_SEED)

    selected_rows = []

    for category, group in df.groupby(
        "category",
        sort=True
    ):

        group = group.sort_values("ticket_id")

        selected_position = int(
            rng.integers(0, len(group))
        )

        selected_rows.append(
            group.iloc[selected_position]
        )

    queries = pd.DataFrame(
        selected_rows
    ).reset_index(drop=True)

    queries = queries.drop(
        columns=["_tokens"],
        errors="ignore"
    )

    selected_categories = set(
        queries["category"].astype(str)
    )

    missing_categories = (
        EXPECTED_CATEGORIES - selected_categories
    )

    print("\nEVALUATION QUERY SELECTION")

    print("Original test records:", original_count)

    print("Empty records excluded:", int(empty_mask.sum()))

    print("Generic-only records excluded:", int(generic_mask.sum()))

    print("Valid test records:", len(df))

    print("Categories represented:", len(selected_categories))

    if missing_categories:
        print(
            "\nWARNING: These expected categories were not represented:"
        )
        print(sorted(missing_categories))

    print("\nSelected evaluation queries:")

    for index, row in queries.iterrows():

        print(
            f"Q-{index + 1:03d}",
            "| Category:",
            row["category"],
            "| Source ticket:",
            row["ticket_id"]
        )

        print(
            "Description:",
            safe_text(row["description"])
        )

    return queries


# ============================================================
# 6. PREPARE RETRIEVAL SYSTEMS
# ============================================================

def prepare_retrieval_systems():

    print("\n" + "=" * 65)

    print("PREPARING RETRIEVAL SYSTEMS")

    print("=" * 65)

    # -------------------------
    # BM25
    # -------------------------

    print("\nBuilding BM25 index...")

    bm25, bm25_df = build_bm25_index()

    # -------------------------
    # Semantic
    # -------------------------

    print("\nLoading semantic index...")

    embeddings, semantic_metadata = load_semantic_index()

    # -------------------------
    # Shared corpus validation
    # -------------------------

    validate_shared_corpus(
        bm25_df,
        semantic_metadata
    )

    # -------------------------
    # Sentence Transformer
    # -------------------------

    print("\nLoading Sentence Transformer...")

    model = load_embedding_model()

    return (
        bm25,
        bm25_df,
        embeddings,
        semantic_metadata,
        model
    )


# ============================================================
# 7. GENERATE CANDIDATES FOR ONE QUERY
# ============================================================

def generate_candidates_for_query(
    query_text,
    bm25,
    bm25_df,
    embeddings,
    semantic_metadata,
    model
):

    # -------------------------
    # BM25 retrieval
    # -------------------------

    bm25_results = search_bm25(
        query_text,
        bm25,
        bm25_df,
        top_k=RETRIEVAL_K
    )

    # -------------------------
    # Semantic retrieval
    # -------------------------

    semantic_results = search_semantic(
        query_text,
        model,
        embeddings,
        semantic_metadata,
        top_k=RETRIEVAL_K
    )

    # -------------------------
    # Hybrid retrieval
    # -------------------------

    hybrid_results = reciprocal_rank_fusion(
        bm25_results,
        semantic_results,
        rrf_k=RRF_K,
        top_k=RETRIEVAL_K
    )

    # -------------------------
    # Create ticket lookup maps
    # -------------------------

    bm25_map = {
        str(result["ticket_id"]): result
        for result in bm25_results
    }

    semantic_map = {
        str(result["ticket_id"]): result
        for result in semantic_results
    }

    hybrid_map = {
        str(result["ticket_id"]): result
        for result in hybrid_results
    }

    # -------------------------
    # Create ranking maps
    # -------------------------

    bm25_rank_map = {
        ticket_id: rank
        for rank, ticket_id in enumerate(
            bm25_map.keys(),
            start=1
        )
    }

    semantic_rank_map = {
        ticket_id: rank
        for rank, ticket_id in enumerate(
            semantic_map.keys(),
            start=1
        )
    }

    hybrid_rank_map = {
        ticket_id: rank
        for rank, ticket_id in enumerate(
            hybrid_map.keys(),
            start=1
        )
    }

    # -------------------------
    # Union of top candidates
    # -------------------------

    candidate_ids = set(
        list(bm25_map.keys())[:REVIEW_K]
        + list(semantic_map.keys())[:REVIEW_K]
        + list(hybrid_map.keys())[:REVIEW_K]
    )

    # Rank candidate rows for convenient manual review.
    def candidate_sort_key(ticket_id):

        return (
            hybrid_rank_map.get(ticket_id, 999),

            min(
                bm25_rank_map.get(ticket_id, 999),
                semantic_rank_map.get(ticket_id, 999)
            ),

            ticket_id
        )

    candidate_ids = sorted(
        candidate_ids,
        key=candidate_sort_key
    )

    rows = []

    # -------------------------
    # Create review records
    # -------------------------

    for ticket_id in candidate_ids:

        bm25_result = bm25_map.get(ticket_id)

        semantic_result = semantic_map.get(ticket_id)

        hybrid_result = hybrid_map.get(ticket_id)

        # Every candidate must have appeared in at least one method.
        base = (
            bm25_result
            or semantic_result
            or hybrid_result
        )

        rows.append({

            "ticket_id": ticket_id,

            "category": base["category"],

            "description": base["description"],

            "search_text": base["search_text"],

            "bm25_rank": bm25_rank_map.get(ticket_id),

            "bm25_score": (
                bm25_result["bm25_score"]
                if bm25_result is not None else None
            ),

            "semantic_rank": semantic_rank_map.get(ticket_id),

            "semantic_score": (
                semantic_result["semantic_score"]
                if semantic_result is not None else None
            ),

            "hybrid_rank": hybrid_rank_map.get(ticket_id),

            "rrf_score": (
                hybrid_result["rrf_score"]
                if hybrid_result is not None else None
            ),

            # These fields are deliberately left blank.
            "is_relevant": "",

            "review_notes": ""

        })

    return rows


# ============================================================
# 8. GENERATE ALL EVALUATION CANDIDATES
# ============================================================

def generate_candidates():

    print("\nIT-SENTINEL RETRIEVAL EVALUATION")

    print("Project root:", ROOT)

    print("Test dataset:", TEST_PATH)

    print("Output file:", OUTPUT_PATH)

    # -------------------------
    # Validate paths
    # -------------------------

    if not ROOT.exists():
        raise FileNotFoundError(
            f"Project root does not exist: {ROOT}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Classification test dataset does not exist: {TEST_PATH}"
        )

    # -------------------------
    # Select test queries
    # -------------------------

    queries = select_evaluation_queries()

    # -------------------------
    # Initialize retrieval systems
    # -------------------------

    (
        bm25,
        bm25_df,
        embeddings,
        semantic_metadata,
        model
    ) = prepare_retrieval_systems()

    all_rows = []

    # -------------------------
    # Process every query
    # -------------------------

    for query_number, (_, query_row) in enumerate(
        queries.iterrows(),
        start=1
    ):

        query_id = f"Q-{query_number:03d}"

        query_text = safe_text(
            query_row["description"]
        )

        query_category = query_row["category"]

        source_ticket_id = query_row["ticket_id"]

        print("\n" + "=" * 65)

        print("Processing query:", query_id)

        print("Category:", query_category)

        print("Source ticket:", source_ticket_id)

        print("Query:", query_text)

        candidate_rows = generate_candidates_for_query(
            query_text=query_text,
            bm25=bm25,
            bm25_df=bm25_df,
            embeddings=embeddings,
            semantic_metadata=semantic_metadata,
            model=model
        )

        print(
            "Unique candidates generated:",
            len(candidate_rows)
        )

        # Add query-level metadata to every candidate.
        for candidate in candidate_rows:

            candidate["query_id"] = query_id

            candidate["query_category"] = query_category

            candidate["query_source_ticket_id"] = (
                source_ticket_id
            )

            candidate["query_text"] = query_text

            all_rows.append(candidate)

    # ========================================================
    # 9. SAVE CANDIDATE REVIEW CSV
    # ========================================================

    output_df = pd.DataFrame(all_rows)

    # Explicit column order.
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

    output_df = output_df[columns]

    # Create reports folder if it does not exist.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # 10. FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 65)

    print("RETRIEVAL REVIEW CANDIDATES CREATED")

    print("=" * 65)

    print(
        "Evaluation queries:",
        output_df["query_id"].nunique()
    )

    print(
        "Candidate rows:",
        len(output_df)
    )

    print(
        "Unique historical tickets:",
        output_df["ticket_id"].nunique()
    )

    print("\nCandidates per query:")

    print(
        output_df.groupby(
            ["query_id", "query_category"]
        ).size().to_string()
    )

    print("\nOutput saved to:")

    print(OUTPUT_PATH)

    print("\nIMPORTANT REVIEW INSTRUCTIONS")

    print("1. is_relevant is intentionally blank.")

    print("2. Mark relevant tickets as 1.")

    print("3. Mark irrelevant tickets as 0.")

    print("4. Add a short reason in review_notes when useful.")

    print(
        "5. Review relevance based on problem similarity "
        "and usefulness, not category alone."
    )

    print(
        "\nDo not treat automatically generated relevance "
        "labels as ground truth."
    )

    print("\nSUCCESS: Candidate generation completed.")


# ============================================================
# 11. MAIN
# ============================================================

if __name__ == "__main__":

    generate_candidates()
