"""Preliminary retrieval evaluation using AI-provisional labels.

This script intentionally does NOT treat blank / NEEDS_HUMAN_REVIEW rows as
irrelevant. It reports conditional precision, precision bounds, label coverage,
known-label pooled recall, and confirmed-MRR lower bounds.

Run from the project root:
    python src/evaluation/evaluate_provisional_retrieval.py

Default input:
    reports/novel_retrieval_review_candidates_provisional_filled.csv

These results are for development diagnostics only. AI-provisional suggestions
are not human-verified ground truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "reports" / "novel_retrieval_review_candidates_provisional_filled.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "retrieval_provisional_evaluation"
METHODS = {
    "BM25": "bm25_rank",
    "Semantic": "semantic_rank",
    "Hybrid": "hybrid_rank",
}
TOP_K = 5
REQUIRED_COLUMNS = {
    "query_id",
    "query_category",
    "ticket_id",
    "is_relevant",
    "label_status",
    *METHODS.values(),
}


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Expected reports/novel_retrieval_review_candidates_provisional_filled.csv"
        )

    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if df.empty:
        raise ValueError("Input CSV is empty.")

    duplicate_mask = df.duplicated(["query_id", "ticket_id"], keep=False)
    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, ["query_id", "ticket_id"]].head(10)
        raise ValueError("Duplicate query/ticket pairs found:\n" + examples.to_string(index=False))

    labels = pd.to_numeric(df["is_relevant"], errors="coerce")
    invalid = df["is_relevant"].notna() & (labels.isna() | ~labels.isin([0, 1]))
    if invalid.any():
        raise ValueError("is_relevant may contain only 0, 1, or blank values.")
    df["is_relevant"] = labels

    # Do not silently treat unresolved human-review rows as irrelevant.
    review_status = df["label_status"].fillna("").astype(str)
    if (df.loc[review_status.eq("NEEDS_HUMAN_REVIEW"), "is_relevant"].notna()).any():
        raise ValueError(
            "Rows marked NEEDS_HUMAN_REVIEW must remain blank in is_relevant "
            "for this provisional evaluation."
        )

    unknown_statuses = set(review_status.unique()) - {"AI_PROVISIONAL", "NEEDS_HUMAN_REVIEW"}
    if unknown_statuses:
        raise ValueError(f"Unexpected label_status values: {sorted(unknown_statuses)}")

    provisional_mask = review_status.eq("AI_PROVISIONAL")
    if df.loc[provisional_mask, "is_relevant"].isna().any():
        raise ValueError("AI_PROVISIONAL rows must have 0/1 labels in is_relevant.")

    for method, col in METHODS.items():
        ranks = pd.to_numeric(df[col], errors="coerce")
        if (df[col].notna() & ranks.isna()).any():
            raise ValueError(f"{col} contains non-numeric rank values.")
        if (ranks.dropna() < 1).any():
            raise ValueError(f"{col} ranks must start at 1.")
        df[col] = ranks

    for query_id, group in df.groupby("query_id"):
        for method, col in METHODS.items():
            top_count = int(group[col].le(TOP_K).sum())
            if top_count != TOP_K:
                raise ValueError(
                    f"{query_id}: expected exactly {TOP_K} {method} top results; found {top_count}."
                )
            top_ranks = group.loc[group[col].le(TOP_K), col]
            if top_ranks.duplicated().any():
                raise ValueError(f"{query_id}: duplicate {method} ranks among top {TOP_K}.")

    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for query_id, group in df.groupby("query_id", sort=True):
        categories = group["query_category"].dropna().astype(str).unique()
        if len(categories) != 1:
            raise ValueError(f"{query_id} must have exactly one query_category.")

        reviewed_pool = group[group["is_relevant"].notna()]
        unreviewed_pool_count = int(group["is_relevant"].isna().sum())
        known_relevant_pool = int(reviewed_pool["is_relevant"].eq(1).sum())
        known_irrelevant_pool = int(reviewed_pool["is_relevant"].eq(0).sum())

        for method, rank_col in METHODS.items():
            top = group[group[rank_col].le(TOP_K)].sort_values(rank_col)
            judged_top = top[top["is_relevant"].notna()]
            unjudged_top_count = int(top["is_relevant"].isna().sum())
            known_relevant_top = int(judged_top["is_relevant"].eq(1).sum())
            judged_top_count = len(judged_top)

            # Conditional precision uses only labelled items in top 5.
            conditional_precision = (
                known_relevant_top / judged_top_count if judged_top_count else np.nan
            )
            # Bounds account for unjudged top-5 candidates: assume all are 0 / all are 1.
            precision_lower = known_relevant_top / TOP_K
            precision_upper = (known_relevant_top + unjudged_top_count) / TOP_K
            # This is recall against only AI-labelled relevant items in the judged pool.
            known_label_recall = (
                known_relevant_top / known_relevant_pool if known_relevant_pool else np.nan
            )
            known_positive_ranks = judged_top.loc[
                judged_top["is_relevant"].eq(1), rank_col
            ]
            confirmed_mrr_lower_bound = (
                1.0 / float(known_positive_ranks.min())
                if not known_positive_ranks.empty
                else 0.0
            )

            rows.append(
                {
                    "query_id": query_id,
                    "query_category": categories[0],
                    "method": method,
                    "candidate_pool_size": int(len(group)),
                    "reviewed_pool_count": int(len(reviewed_pool)),
                    "unreviewed_pool_count": unreviewed_pool_count,
                    "known_relevant_in_reviewed_pool": known_relevant_pool,
                    "known_irrelevant_in_reviewed_pool": known_irrelevant_pool,
                    "top_5_count": int(len(top)),
                    "reviewed_top_5_count": judged_top_count,
                    "unreviewed_top_5_count": unjudged_top_count,
                    "top_5_label_coverage": judged_top_count / TOP_K,
                    "known_relevant_in_top_5": known_relevant_top,
                    "conditional_precision_at_5_on_reviewed": conditional_precision,
                    "precision_at_5_lower_bound": precision_lower,
                    "precision_at_5_upper_bound": precision_upper,
                    "known_label_pooled_recall_at_5": known_label_recall,
                    "confirmed_mrr_at_5_lower_bound": confirmed_mrr_lower_bound,
                }
            )

    return pd.DataFrame(rows)


def aggregate(query_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "top_5_label_coverage",
        "conditional_precision_at_5_on_reviewed",
        "precision_at_5_lower_bound",
        "precision_at_5_upper_bound",
        "known_label_pooled_recall_at_5",
        "confirmed_mrr_at_5_lower_bound",
    ]

    def agg_frame(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        grouped = frame.groupby(group_cols, as_index=False).agg(
            queries_evaluated=("query_id", "nunique"),
            queries_with_known_relevant_pool=(
                "known_relevant_in_reviewed_pool", lambda values: int((values > 0).sum())
            ),
            mean_unreviewed_candidates_in_pool=("unreviewed_pool_count", "mean"),
            **{metric: (metric, "mean") for metric in metric_cols},
        )
        return grouped

    overall = agg_frame(query_metrics, ["method"])
    by_category = agg_frame(query_metrics, ["query_category", "method"])
    return overall, by_category


def main() -> int:
    input_path = DEFAULT_INPUT
    output_dir = DEFAULT_OUTPUT_DIR

    try:
        df = load_data(input_path)
        query_metrics = compute_metrics(df)
        overall, by_category = aggregate(query_metrics)

        output_dir.mkdir(parents=True, exist_ok=True)
        query_path = output_dir / "provisional_metrics_by_query.csv"
        category_path = output_dir / "provisional_metrics_by_category.csv"
        overall_path = output_dir / "provisional_metrics_overall.csv"
        summary_path = output_dir / "provisional_evaluation_summary.json"

        query_metrics.to_csv(query_path, index=False)
        by_category.to_csv(category_path, index=False)
        overall.to_csv(overall_path, index=False)

        status_counts = df["label_status"].value_counts(dropna=False).to_dict()
        summary = {
            "evaluation_type": "AI-assisted provisional development diagnostic",
            "not_ground_truth": True,
            "input_file": str(input_path),
            "candidate_rows": int(len(df)),
            "queries_evaluated": int(df["query_id"].nunique()),
            "unique_candidate_tickets": int(df["ticket_id"].nunique()),
            "label_status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "label_counts": {
                "provisional_relevant_1": int(df["is_relevant"].eq(1).sum()),
                "provisional_irrelevant_0": int(df["is_relevant"].eq(0).sum()),
                "unreviewed_blank": int(df["is_relevant"].isna().sum()),
            },
            "metric_definitions": {
                "top_5_label_coverage": "Fraction of each method's top five candidates with a provisional 0/1 label.",
                "conditional_precision_at_5_on_reviewed": "Relevant labelled candidates divided by labelled candidates in top five; excludes unreviewed candidates and must be read with coverage.",
                "precision_at_5_lower_bound": "Confirmed relevant candidates divided by five; treats unreviewed top-five candidates as not yet confirmed relevant.",
                "precision_at_5_upper_bound": "Confirmed relevant plus unreviewed top-five candidates divided by five; assumes all unreviewed top-five candidates are relevant.",
                "known_label_pooled_recall_at_5": "Confirmed relevant top-five results divided by confirmed relevant candidates in the reviewed portion of the pooled candidate set. Not full-corpus recall.",
                "confirmed_mrr_at_5_lower_bound": "Reciprocal rank of first confirmed relevant candidate in top five, or zero if none is confirmed; unreviewed candidates may hide an earlier relevant result.",
            },
            "caution": (
                "All labels currently originate from AI suggestions; 39 rows marked NEEDS_HUMAN_REVIEW "
                "remain unjudged. These metrics are preliminary diagnostics only, not human-verified "
                "benchmark results and not suitable as claims of final retrieval performance."
            ),
            "scope_note": (
                "Recall is measured only against the AI-labelled subset of the judged union candidate pool, "
                "not the complete 1,570-ticket historical corpus."
            ),
            "overall_results": overall.replace({np.nan: None}).to_dict(orient="records"),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("\nPROVISIONAL RETRIEVAL DIAGNOSTICS COMPLETE")
        print("=" * 72)
        print(f"Candidate rows: {len(df)}")
        print(f"Queries: {df['query_id'].nunique()}")
        print(f"AI-provisional 1 labels: {int(df['is_relevant'].eq(1).sum())}")
        print(f"AI-provisional 0 labels: {int(df['is_relevant'].eq(0).sum())}")
        print(f"Unreviewed rows: {int(df['is_relevant'].isna().sum())}")
        print("\nOVERALL PRELIMINARY METRICS")
        print(overall.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        print("\nReports written to:")
        print(f"- {overall_path}")
        print(f"- {category_path}")
        print(f"- {query_path}")
        print(f"- {summary_path}")
        print("\nWARNING: AI-provisional labels are not human-verified ground truth.")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
