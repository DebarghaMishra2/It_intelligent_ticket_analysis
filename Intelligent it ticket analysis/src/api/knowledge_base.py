"""Load and search explicitly approved knowledge articles.

Expected JSON format: a list of articles containing article_id, title, content,
resolution_steps, approval_status, source_name, source_reference, approved_by,
and approved_at. Only articles with approval_status == "approved" are indexed.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_PATH = ROOT / "data" / "knowledge_base" / "approved_articles.json"
REQUIRED_FIELDS = {
    "article_id",
    "title",
    "content",
    "resolution_steps",
    "approval_status",
    "source_name",
    "source_reference",
    "approved_by",
    "approved_at",
}


def load_approved_knowledge_base() -> Dict[str, Any]:
    if not KNOWLEDGE_PATH.exists():
        return {"articles": [], "vectorizer": None, "matrix": None}

    with KNOWLEDGE_PATH.open("r", encoding="utf-8") as file:
        raw_articles = json.load(file)

    if not isinstance(raw_articles, list):
        raise ValueError("approved_articles.json must contain a JSON list.")

    articles: List[Dict[str, Any]] = []
    texts: List[str] = []
    for index, article in enumerate(raw_articles):
        if not isinstance(article, dict):
            raise ValueError(f"Knowledge article at index {index} must be an object.")
        if not REQUIRED_FIELDS.issubset(article):
            missing = sorted(REQUIRED_FIELDS - set(article))
            raise ValueError(f"Knowledge article at index {index} is missing: {missing}")
        if article["approval_status"] != "approved":
            continue
        steps = article["resolution_steps"]
        if not isinstance(steps, list) or not steps or not all(isinstance(x, str) for x in steps):
            raise ValueError(
                f"Approved article {article['article_id']} must have a non-empty list of resolution_steps."
            )
        if not article["source_name"] or not article["source_reference"]:
            raise ValueError(
                f"Approved article {article['article_id']} must include source attribution."
            )
        article_text = " ".join(
            [str(article["title"]), str(article["content"]), *steps]
        ).strip()
        if article_text:
            articles.append(article)
            texts.append(article_text)

    if not texts:
        return {"articles": [], "vectorizer": None, "matrix": None}

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return {"articles": articles, "vectorizer": vectorizer, "matrix": matrix}


def search_approved_knowledge(
    query: str, knowledge_resources: Dict[str, Any], top_k: int = 3
) -> List[Dict[str, Any]]:
    vectorizer = knowledge_resources.get("vectorizer")
    matrix = knowledge_resources.get("matrix")
    articles = knowledge_resources.get("articles", [])
    if vectorizer is None or matrix is None or not articles:
        return []

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked_indices = scores.argsort()[::-1]

    results: List[Dict[str, Any]] = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        article = articles[int(index)]
        results.append(
            {
                "article_id": article["article_id"],
                "title": article["title"],
                "resolution_steps": article["resolution_steps"],
                "source_name": article["source_name"],
                "source_reference": article["source_reference"],
                "approved_by": article["approved_by"],
                "approved_at": article["approved_at"],
                "lexical_similarity": round(score, 4),
                "human_validation_required": True,
            }
        )
        if len(results) >= top_k:
            break
    return results
