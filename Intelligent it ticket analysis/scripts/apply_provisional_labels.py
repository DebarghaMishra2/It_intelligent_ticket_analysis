"""Copy AI-provisional relevance suggestions into the working relevance column.

This script is for preliminary development only. Rows requiring human review
are deliberately left blank and must not be treated as ground-truth labels.

Run from the project root:
    python scripts/apply_provisional_labels.py
"""

from pathlib import Path
import sys

import pandas as pd


# scripts/apply_provisional_labels.py -> project root is its parent directory's parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "reports" / "novel_retrieval_review_candidates_provisional.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "novel_retrieval_review_candidates_provisional_filled.csv"

REQUIRED_COLUMNS = {
    "is_relevant",
    "is_relevant_provisional",
    "label_status",
}


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_PATH}\n"
            "Place novel_retrieval_review_candidates_provisional.csv in the project's reports/ folder."
        )

    df = pd.read_csv(INPUT_PATH)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "The input CSV is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    # Work with numeric suggestions so values such as '1' and '0' are handled safely.
    suggestions = pd.to_numeric(df["is_relevant_provisional"], errors="coerce")
    provisional_rows = df["label_status"].eq("AI_PROVISIONAL")
    human_review_rows = df["label_status"].eq("NEEDS_HUMAN_REVIEW")

    # Validate that every AI_PROVISIONAL row has a binary suggestion.
    invalid_suggestions = provisional_rows & ~suggestions.isin([0, 1])
    if invalid_suggestions.any():
        bad_rows = (df.index[invalid_suggestions] + 2).tolist()  # +2 accounts for header and 0-based index
        raise ValueError(
            "AI_PROVISIONAL rows must have 0 or 1 in is_relevant_provisional. "
            f"Check CSV row(s): {bad_rows[:20]}"
        )

    # Start clean: only copy suggestions for rows explicitly marked AI_PROVISIONAL.
    df["is_relevant"] = pd.NA
    df.loc[provisional_rows, "is_relevant"] = suggestions.loc[provisional_rows].astype("Int64")

    # Keep NEEDS_HUMAN_REVIEW rows blank, regardless of any accidental suggestion.
    df.loc[human_review_rows, "is_relevant"] = pd.NA

    # Do not silently process unexpected statuses.
    known_statuses = {"AI_PROVISIONAL", "NEEDS_HUMAN_REVIEW"}
    unexpected_statuses = set(df["label_status"].dropna().unique()) - known_statuses
    if unexpected_statuses:
        raise ValueError(f"Unexpected label_status value(s): {sorted(unexpected_statuses)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    relevant_count = int((df["is_relevant"] == 1).sum())
    irrelevant_count = int((df["is_relevant"] == 0).sum())
    blank_count = int(df["is_relevant"].isna().sum())

    print("\nSUCCESS: provisional labels copied.")
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Total candidate rows       : {len(df)}")
    print(f"Provisional relevant (1)   : {relevant_count}")
    print(f"Provisional irrelevant (0) : {irrelevant_count}")
    print(f"Left blank for human review: {blank_count}")
    print("\nNOTE: This is a provisional development dataset, not final evaluation ground truth.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)