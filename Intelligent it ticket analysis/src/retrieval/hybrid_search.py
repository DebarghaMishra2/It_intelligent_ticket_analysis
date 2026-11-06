
from src.retrieval.bm25_search import (
    build_bm25_index,
    search_bm25,
    get_user_query
)

from src.retrieval.semantic_search import (
    load_semantic_index,
    load_embedding_model,
    search_semantic
)


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

TOP_K = 5

CANDIDATE_K = 50

RRF_K = 60


# --------------------------------------------------
# 2. VERIFY BOTH RETRIEVAL CORPORA
# --------------------------------------------------

def validate_shared_corpus(bm25_df, semantic_metadata):

    bm25_ids = set(bm25_df["ticket_id"].astype(str))

    semantic_ids = set(
        semantic_metadata["ticket_id"].astype(str)
    )

    if bm25_df["ticket_id"].duplicated().any():
        raise ValueError(
            "BM25 corpus contains duplicate ticket IDs."
        )

    if semantic_metadata["ticket_id"].duplicated().any():
        raise ValueError(
            "Semantic corpus contains duplicate ticket IDs."
        )

    if bm25_ids != semantic_ids:

        bm25_only = bm25_ids - semantic_ids

        semantic_only = semantic_ids - bm25_ids

        raise ValueError(
            "BM25 and semantic indexes do not contain "
            "the same ticket IDs.\n"
            f"BM25 only: {sorted(bm25_only)[:10]}\n"
            f"Semantic only: {sorted(semantic_only)[:10]}"
        )

    print("\nSHARED CORPUS VALIDATION PASSED")

    print("BM25 tickets:", len(bm25_ids))

    print("Semantic tickets:", len(semantic_ids))

    print("Common tickets:", len(bm25_ids & semantic_ids))


# --------------------------------------------------
# 3. RECIPROCAL RANK FUSION
# --------------------------------------------------

def reciprocal_rank_fusion(
    bm25_results,
    semantic_results,
    rrf_k=RRF_K,
    top_k=TOP_K
):

    combined = {}

    # Add BM25 results.
    for rank, result in enumerate(bm25_results, start=1):

        ticket_id = str(result["ticket_id"])

        combined[ticket_id] = {
            "ticket_id": ticket_id,
            "category": result["category"],
            "description": result["description"],
            "search_text": result["search_text"],
            "bm25_rank": rank,
            "bm25_score": result["bm25_score"],
            "semantic_rank": None,
            "semantic_score": None,
            "rrf_score": 1.0 / (rrf_k + rank)
        }

    # Add semantic results and combine rankings.
    for rank, result in enumerate(semantic_results, start=1):

        ticket_id = str(result["ticket_id"])

        if ticket_id not in combined:

            combined[ticket_id] = {
                "ticket_id": ticket_id,
                "category": result["category"],
                "description": result["description"],
                "search_text": result["search_text"],
                "bm25_rank": None,
                "bm25_score": None,
                "semantic_rank": rank,
                "semantic_score": result["semantic_score"],
                "rrf_score": 0.0
            }

        else:

            combined[ticket_id]["semantic_rank"] = rank

            combined[ticket_id]["semantic_score"] = (
                result["semantic_score"]
            )

        combined[ticket_id]["rrf_score"] += (
            1.0 / (rrf_k + rank)
        )

    # Sort by fused ranking score.
    ranked_results = sorted(
        combined.values(),
        key=lambda item: (
            -item["rrf_score"],
            item["ticket_id"]
        )
    )

    return ranked_results[:top_k]


# --------------------------------------------------
# 4. DISPLAY HYBRID RESULTS
# --------------------------------------------------

def display_hybrid_results(results):

    print("\nTOP", TOP_K, "HYBRID RESULTS")

    if not results:

        print("No matching historical tickets found.")

        return

    for rank, result in enumerate(results, start=1):

        print("\nResult:", rank)

        print("Ticket ID:", result["ticket_id"])

        print("Category:", result["category"])

        print("Description:", result["description"])

        print("Cleaned:", result["search_text"])

        print("BM25 rank:", result["bm25_rank"])

        print("BM25 score:", result["bm25_score"])

        print("Semantic rank:", result["semantic_rank"])

        print("Semantic score:", result["semantic_score"])

        print("RRF score:", round(result["rrf_score"], 6))


# --------------------------------------------------
# 5. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\nLoading BM25...")

    bm25, bm25_df = build_bm25_index()

    print("\nLoading semantic index...")

    embeddings, semantic_metadata = load_semantic_index()

    # Ensure both systems use exactly the same corpus.
    validate_shared_corpus(
        bm25_df,
        semantic_metadata
    )

    print("\nLoading semantic model...")

    model = load_embedding_model()

    print("\nHybrid retrieval ready.")

    # Get one query for both retrieval methods.
    query = get_user_query()

    if not query:

        print("Empty query. Exiting.")

        return

    # Retrieve more than five candidates from each method.
    print("\nRunning BM25 retrieval...")

    bm25_results = search_bm25(
        query,
        bm25,
        bm25_df,
        top_k=CANDIDATE_K
    )

    print("BM25 candidates:", len(bm25_results))

    print("\nRunning semantic retrieval...")

    semantic_results = search_semantic(
        query,
        model,
        embeddings,
        semantic_metadata,
        top_k=CANDIDATE_K
    )

    print("Semantic candidates:", len(semantic_results))

    # Combine rankings.
    print("\nCombining rankings with RRF...")

    hybrid_results = reciprocal_rank_fusion(
        bm25_results,
        semantic_results,
        rrf_k=RRF_K,
        top_k=TOP_K
    )

    display_hybrid_results(hybrid_results)


if __name__ == "__main__":

    main()
