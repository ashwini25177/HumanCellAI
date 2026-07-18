"""
HumanCellAI - Scientific Literature Vector Index Builder

Input:
    data/processed/paper_chunks.jsonl

Outputs:
    data/vectorstore/paper_index.faiss
    data/vectorstore/paper_chunks.pkl
    data/vectorstore/index_config.json

The script:
    1. Loads citation-aware scientific chunks.
    2. Converts them into dense embeddings.
    3. Normalizes embeddings for cosine-similarity retrieval.
    4. Stores vectors in a FAISS inner-product index.
    5. Preserves all citation metadata.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed/paper_chunks.jsonl")
VECTORSTORE_DIR = Path("data/vectorstore")

INDEX_FILE = VECTORSTORE_DIR / "paper_index.faiss"
RECORDS_FILE = VECTORSTORE_DIR / "paper_chunks.pkl"
CONFIG_FILE = VECTORSTORE_DIR / "index_config.json"

# Use this model for the first reliable version.
# It is lightweight enough to run on a normal Windows computer.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32


def load_chunk_records(path: Path) -> list[dict[str, Any]]:
    """
    Read citation-aware chunk records from a JSONL file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file was not found: {path}\n"
            "Run src/ingestion/build_corpus.py first."
        )

    records: list[dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

            text = str(record.get("text", "")).strip()

            if not text:
                continue

            required_fields = {
                "chunk_id",
                "pmcid",
                "title",
                "category",
                "section",
                "text",
            }

            missing_fields = required_fields.difference(record)

            if missing_fields:
                raise ValueError(
                    f"Record at line {line_number} is missing fields: "
                    f"{sorted(missing_fields)}"
                )

            records.append(record)

    if not records:
        raise RuntimeError(
            "No valid scientific chunks were found in the corpus."
        )

    return records


def prepare_embedding_text(record: dict[str, Any]) -> str:
    """
    Combine metadata with passage text before embedding.

    Including the title and section can improve retrieval because a query
    may refer to a paper topic even when the exact phrase is absent from
    the paragraph itself.
    """
    title = str(record.get("title", "")).strip()
    section = str(record.get("section", "")).strip()
    text = str(record.get("text", "")).strip()

    return (
        f"Paper title: {title}\n"
        f"Section: {section}\n"
        f"Scientific passage: {text}"
    )


def main() -> None:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("HumanCellAI - Scientific Literature Index Builder")
    print("=" * 72)

    records = load_chunk_records(CHUNKS_FILE)

    print("Scientific chunks loaded:", len(records))
    print("Embedding model:", MODEL_NAME)

    embedding_texts = [
        prepare_embedding_text(record)
        for record in records
    ]

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        embedding_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Expected a 2D embedding matrix, got shape {embeddings.shape}."
        )

    if embeddings.shape[0] != len(records):
        raise RuntimeError(
            "The number of embeddings does not match the number of records."
        )

    dimension = embeddings.shape[1]

    # Because vectors are normalized, inner product corresponds to
    # cosine-similarity ranking.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    if index.ntotal != len(records):
        raise RuntimeError(
            "FAISS did not store the expected number of vectors."
        )

    faiss.write_index(
        index,
        str(INDEX_FILE),
    )

    with open(RECORDS_FILE, "wb") as handle:
        pickle.dump(
            records,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    configuration = {
        "model_name": MODEL_NAME,
        "embedding_dimension": int(dimension),
        "number_of_vectors": int(index.ntotal),
        "similarity": "cosine_via_normalized_inner_product",
        "chunks_file": str(CHUNKS_FILE),
        "index_file": str(INDEX_FILE),
        "records_file": str(RECORDS_FILE),
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            configuration,
            handle,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("Vector index created successfully")
    print("=" * 72)
    print("Vectors stored:", index.ntotal)
    print("Embedding dimension:", dimension)
    print("FAISS index:", INDEX_FILE)
    print("Citation records:", RECORDS_FILE)
    print("Index configuration:", CONFIG_FILE)


if __name__ == "__main__":
    main()