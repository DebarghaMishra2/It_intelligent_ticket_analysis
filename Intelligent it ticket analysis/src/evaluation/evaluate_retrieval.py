
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "reports" /
    "retrieval_review_candidates_filled.csv"
)

OUTPUT_DIR = ROOT / "reports"

TOP_K = 5

METHODS = {
    "BM25": "bm25_rank",
    "Semantic": "semantic_rank",
    "Hybrid": "hybrid_rank"
}


# ==========================================
# 2. LOAD AND VALIDATE DATA
# ==========================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    required = [
        "query_id",
        "ticket_id",
        "is_relevant",
        *METHODS.values()
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Missing column: {column}"
            )

    df["is_relevant"] = pd.to_numeric(
        df["is_relevant"],
        errors="coerce"
    )

    if df["is_relevant"].isna().any():
        raise ValueError(
            "Some relevance labels are missing."
        )

    if not df["is_relevant"].isin([0, 1]).all():
        raise ValueError(
            "Relevance labels must be 0 or 1."
        )

    for column in METHODS.values():
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    if df.duplicated(
        subset=["query_id", "ticket_id"]
    ).any():
        raise ValueError(
            "Duplicate query-ticket pairs found."
        )

    return df


# ==========================================
# 3. CALCULATE METRICS FOR ONE QUERY
# ==========================================

def calculate_query_metrics(group, rank_column):

    ranked = group[
        group[rank_column].notna()
    ].sort_values(rank_column)

    top_results = ranked[
        ranked[rank_column] <= TOP_K
    ].head(TOP_K)

    relevant_found = int(
        top_results["is_relevant"].sum()
    )

    relevant_in_pool = int(
        group["is_relevant"].sum()
    )

    # Precision@5
    precision = relevant_found / TOP_K

    # Recall@5
    recall = (
        relevant_found / relevant_in_pool
        if relevant_in_pool > 0
        else np.nan
    )

    # MRR@5
    relevant_rankings = top_results[
        top_results["is_relevant"] == 1
    ]

    if not relevant_rankings.empty:

        first_rank = float(
            relevant_rankings.iloc[0][rank_column]
        )

        reciprocal_rank = 1 / first_rank

    else:

        first_rank = np.nan
        reciprocal_rank = 0.0

    return {
        "retrieved_at_5": len(top_results),
        "relevant_in_pool": relevant_in_pool,
        "relevant_found_at_5": relevant_found,
        "precision_at_5": precision,
        "recall_at_5": recall,
        "first_relevant_rank": first_rank,
        "mrr_at_5": reciprocal_rank,
        "has_relevant_in_pool": relevant_in_pool > 0
    }


# ==========================================
# 4. EVALUATE ALL METHODS
# ==========================================

def evaluate_all_methods(df):

    details = []

    for method, rank_column in METHODS.items():

        for query_id, group in df.groupby("query_id"):

            metrics = calculate_query_metrics(
                group,
                rank_column
            )

            details.append({
                "method": method,
                "query_id": query_id,
                **metrics
            })

    return pd.DataFrame(details)


# ==========================================
# 5. GENERATE SUMMARY
# ==========================================

def generate_summary(details):

    summary = []

    for method in METHODS:

        data = details[
            details["method"] == method
        ].copy()

        total_hits = int(
            data["relevant_found_at_5"].sum()
        )

        total_relevant = int(
            data["relevant_in_pool"].sum()
        )

        # Macro recall over eligible queries
        valid_recall = data[
            data["recall_at_5"].notna()
        ]

        macro_recall = (
            valid_recall["recall_at_5"].mean()
            if not valid_recall.empty
            else np.nan
        )

        # True micro pooled recall
        micro_recall = (
            total_hits / total_relevant
            if total_relevant > 0
            else np.nan
        )

        # MRR including zero-hit queries
        mrr_all = data["mrr_at_5"].mean()

        # MRR among queries with judged positives
        positive_queries = data[
            data["has_relevant_in_pool"]
        ]

        mrr_positive = (
            positive_queries["mrr_at_5"].mean()
            if not positive_queries.empty
            else np.nan
        )

        summary.append({

            "method": method,

            "precision_at_5": data[
                "precision_at_5"
            ].mean(),

            "macro_recall_at_5": macro_recall,

            "micro_pooled_recall_at_5": micro_recall,

            "mrr_at_5_all_queries": mrr_all,

            "mrr_at_5_positive_pool_queries": mrr_positive,

            "relevant_found": total_hits,

            "relevant_in_pool": total_relevant,

            "queries_evaluated": len(data),

            "queries_with_relevant_candidates":
                len(positive_queries)

        })

    return pd.DataFrame(summary)


# ==========================================
# 6. SAVE RESULTS
# ==========================================

def save_results(details, summary):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    details.to_csv(
        OUTPUT_DIR / "retrieval_evaluation_by_query.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_DIR / "retrieval_metrics_summary.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / "retrieval_metrics_summary.json",
        "w"
    ) as file:

        json.dump(
            summary.to_dict(orient="records"),
            file,
            indent=4
        )


# ==========================================
# 7. MAIN FUNCTION
# ==========================================

def main():

    print("Loading retrieval review dataset...")

    df = load_data()

    print("Candidate pairs:", len(df))
    print("Queries:", df["query_id"].nunique())
    print(
        "Relevant candidates:",
        int(df["is_relevant"].sum())
    )

    print("\nEvaluating retrieval methods...")

    details = evaluate_all_methods(df)

    summary = generate_summary(details)

    print("\n" + "=" * 80)
    print("CORRECTED RETRIEVAL EVALUATION")
    print("=" * 80)

    print(
        summary.round(4).to_string(index=False)
    )

    print("\nQueries without relevant candidates:")

    invalid_queries = (
        details[
            details["relevant_in_pool"] == 0
        ]
        ["query_id"]
        .unique()
    )

    print(list(invalid_queries))

    save_results(details, summary)

    print("\nEvaluation reports saved successfully!")


if __name__ == "__main__":
    main()
