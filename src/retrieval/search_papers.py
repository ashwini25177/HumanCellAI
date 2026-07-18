"""
HumanCellAI - Scientific Paper Retriever with Cross-Encoder Reranking

Pipeline:
    Question
        -> embedding search over FAISS
        -> retrieve broad candidate set
        -> cross-encoder relevance reranking
        -> section-aware scoring
        -> paper diversity filtering
        -> citation-ready evidence

Inputs:
    data/vectorstore/paper_index.faiss
    data/vectorstore/paper_chunks.pkl
    data/vectorstore/index_config.json
"""

from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


VECTORSTORE_DIR = Path("data/vectorstore")

INDEX_FILE = VECTORSTORE_DIR / "paper_index.faiss"
RECORDS_FILE = VECTORSTORE_DIR / "paper_chunks.pkl"
CONFIG_FILE = VECTORSTORE_DIR / "index_config.json"

# Retrieve many candidates first, then rerank them.
INITIAL_CANDIDATES = 30
FINAL_TOP_K = 5

# Prevent a single paper from dominating all evidence.
MAX_RESULTS_PER_PAPER = 2

# Minimum FAISS semantic similarity.
MIN_SEMANTIC_SCORE = 0.30

# Lightweight cross-encoder suitable for a Windows CPU.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Sections preferred for explanatory scientific questions.
SECTION_BONUSES = {
    "abstract": 0.25,
    "intro": 0.20,
    "introduction": 0.20,
    "background": 0.15,
    "results": 0.12,
    "discussion": 0.12,
    "conclusion": 0.10,
    "conclusions": 0.10,
    "methods": 0.02,
    "materials and methods": 0.02,
}

BAD_SECTION_NAMES = {
    "ref",
    "refs",
    "reference",
    "references",
    "bibliography",
}


def load_resources() -> tuple[
    faiss.Index,
    list[dict[str, Any]],
    SentenceTransformer,
    CrossEncoder,
    dict[str, Any],
]:
    """Load index, records, embedding model and reranker."""

    for path in (INDEX_FILE, RECORDS_FILE, CONFIG_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file does not exist: {path}\n"
                "Run the corpus and index builders first."
            )

    index = faiss.read_index(str(INDEX_FILE))

    with open(RECORDS_FILE, "rb") as handle:
        records = pickle.load(handle)

    with open(CONFIG_FILE, "r", encoding="utf-8") as handle:
        configuration = json.load(handle)

    if index.ntotal != len(records):
        raise RuntimeError(
            "FAISS vector count and citation-record count do not match."
        )

    embedding_model = SentenceTransformer(
        configuration["model_name"]
    )

    reranker = CrossEncoder(RERANKER_MODEL)

    return (
        index,
        records,
        embedding_model,
        reranker,
        configuration,
    )


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def normalize_section(section: str) -> str:
    return normalize_text(section).strip(" .:_-")


def section_bonus(section: str) -> float:
    """
    Give a small preference to explanatory sections such as
    abstract, introduction and discussion.
    """
    normalized = normalize_section(section)

    for name, bonus in SECTION_BONUSES.items():
        if normalized == name or normalized.startswith(name):
            return bonus

    return 0.0


def is_bad_section(section: str) -> bool:
    normalized = normalize_section(section)

    return (
        normalized in BAD_SECTION_NAMES
        or normalized.startswith("ref ")
        or normalized.startswith("reference ")
    )


def looks_like_low_quality_equation_fragment(text: str) -> bool:
    """
    Detect passages whose mathematical content was damaged during
    conversion from XML to plain text.

    Example:
        'if cell i does not belong to lineage l, i.e., , then set .'
    """
    text = str(text).strip()

    empty_equation_patterns = [
        r",\s*,",
        r"=\s*\.",
        r"\bi\.e\.,\s*,",
        r"\bdenoted by\s*;",
        r"\bdenoted by\s*\.",
    ]

    pattern_matches = sum(
        bool(re.search(pattern, text, flags=re.IGNORECASE))
        for pattern in empty_equation_patterns
    )

    return pattern_matches >= 2


def retrieve_candidates(
    question: str,
    index: faiss.Index,
    records: list[dict[str, Any]],
    embedding_model: SentenceTransformer,
) -> list[dict[str, Any]]:
    """
    Retrieve a broad candidate set using FAISS semantic similarity.
    """
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    candidate_count = min(
        INITIAL_CANDIDATES,
        len(records),
    )

    scores, indices = index.search(
        query_embedding,
        candidate_count,
    )

    candidates: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    seen_texts: set[str] = set()

    for semantic_score, record_index in zip(
        scores[0],
        indices[0],
    ):
        if record_index < 0:
            continue

        semantic_score = float(semantic_score)

        if semantic_score < MIN_SEMANTIC_SCORE:
            continue

        record = dict(records[int(record_index)])

        chunk_id = str(record.get("chunk_id", ""))
        text = str(record.get("text", "")).strip()
        section = str(record.get("section", ""))

        if not text:
            continue

        if is_bad_section(section):
            continue

        if looks_like_low_quality_equation_fragment(text):
            continue

        normalized_text = normalize_text(text)

        if chunk_id in seen_chunk_ids:
            continue

        if normalized_text in seen_texts:
            continue

        record["semantic_score"] = semantic_score

        candidates.append(record)
        seen_chunk_ids.add(chunk_id)
        seen_texts.add(normalized_text)

    return candidates


def rerank_candidates(
    question: str,
    candidates: list[dict[str, Any]],
    reranker: CrossEncoder,
) -> list[dict[str, Any]]:
    """
    Use a cross-encoder to estimate whether each passage directly
    answers the question.
    """
    if not candidates:
        return []

    question_passage_pairs = [
        [question, str(candidate["text"])]
        for candidate in candidates
    ]

    reranker_scores = reranker.predict(
        question_passage_pairs,
        show_progress_bar=False,
    )

    reranked: list[dict[str, Any]] = []

    for candidate, raw_reranker_score in zip(
        candidates,
        reranker_scores,
    ):
        result = dict(candidate)

        raw_reranker_score = float(raw_reranker_score)

        # Cross-encoder output is a relevance logit, not a probability.
        result["reranker_score"] = raw_reranker_score

        # Combined score uses:
        # - direct question-passage relevance;
        # - semantic retrieval score;
        # - small section preference.
        result["combined_score"] = (
            raw_reranker_score
            + 1.5 * float(result["semantic_score"])
            + section_bonus(str(result.get("section", "")))
        )

        reranked.append(result)

    reranked.sort(
        key=lambda result: result["combined_score"],
        reverse=True,
    )

    return reranked


def apply_paper_diversity(
    reranked: list[dict[str, Any]],
    top_k: int = FINAL_TOP_K,
) -> list[dict[str, Any]]:
    """
    Limit the number of passages selected from any single paper.
    """
    final_results: list[dict[str, Any]] = []
    paper_counts: defaultdict[str, int] = defaultdict(int)

    for result in reranked:
        pmcid = str(result.get("pmcid", "Unknown"))

        if paper_counts[pmcid] >= MAX_RESULTS_PER_PAPER:
            continue

        final_results.append(result)
        paper_counts[pmcid] += 1

        if len(final_results) >= top_k:
            break

    return final_results


def retrieve_evidence(
    question: str,
    index: faiss.Index,
    records: list[dict[str, Any]],
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder,
) -> list[dict[str, Any]]:
    candidates = retrieve_candidates(
        question=question,
        index=index,
        records=records,
        embedding_model=embedding_model,
    )

    reranked = rerank_candidates(
        question=question,
        candidates=candidates,
        reranker=reranker,
    )

    return apply_paper_diversity(
        reranked=reranked,
        top_k=FINAL_TOP_K,
    )


def print_results(
    question: str,
    results: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 80)
    print("HumanCellAI - Reranked Scientific Evidence")
    print("=" * 80)

    print("\nQuestion:")
    print(question)

    if not results:
        print(
            "\nNo sufficiently relevant evidence was found in the "
            "currently indexed research papers."
        )
        return

    print("\nFinal evidence passages:", len(results))

    for rank, result in enumerate(results, start=1):
        print("\n" + "-" * 80)
        print(f"RESULT {rank}")
        print("-" * 80)

        print(
            "Combined score:",
            f"{float(result['combined_score']):.4f}",
        )
        print(
            "Reranker score:",
            f"{float(result['reranker_score']):.4f}",
        )
        print(
            "Semantic score:",
            f"{float(result['semantic_score']):.4f}",
        )

        print("Paper:", result.get("title", "Unknown"))
        print("PMCID:", result.get("pmcid", "Unknown"))
        print("Category:", result.get("category", "Unknown"))
        print("Section:", result.get("section", "Unknown"))
        print(
            "Passage number:",
            result.get("passage_number", "Unknown"),
        )
        print("Chunk ID:", result.get("chunk_id", "Unknown"))

        print("\nSupporting passage:\n")
        print(result.get("text", ""))

        pmcid = str(result.get("pmcid", "")).strip()

        if pmcid.startswith("PMC"):
            print(
                "\nArticle URL:",
                f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
            )


def main() -> None:
    print("=" * 80)
    print("HumanCellAI - Reranked Scientific Paper Search")
    print("=" * 80)
    print("Loading scientific retrieval resources...")

    (
        index,
        records,
        embedding_model,
        reranker,
        configuration,
    ) = load_resources()

    print("Vectors loaded:", index.ntotal)
    print(
        "Embedding model:",
        configuration["model_name"],
    )
    print("Reranker model:", RERANKER_MODEL)

    question = input(
        "\nEnter a biological or single-cell question: "
    ).strip()

    if not question:
        print("No question was entered.")
        return

    results = retrieve_evidence(
        question=question,
        index=index,
        records=records,
        embedding_model=embedding_model,
        reranker=reranker,
    )

    print_results(
        question=question,
        results=results,
    )


if __name__ == "__main__":
    main()