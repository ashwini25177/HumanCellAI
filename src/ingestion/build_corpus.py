"""
HumanCellAI - BioC Corpus Builder

Reads PMC BioC JSON articles from:
    documents/papers/**/*.json

Creates:
    data/processed/paper_chunks.jsonl
    data/metadata/paper_metadata.csv

Each chunk preserves:
    - PMCID
    - paper title
    - category
    - section
    - passage number
    - source file
    - chunk text
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


PAPERS_DIR = Path("documents/papers")
PROCESSED_DIR = Path("data/processed")
METADATA_DIR = Path("data/metadata")

CHUNKS_FILE = PROCESSED_DIR / "paper_chunks.jsonl"
METADATA_FILE = METADATA_DIR / "paper_metadata.csv"

MAX_CHUNK_WORDS = 220
OVERLAP_WORDS = 40
MIN_PASSAGE_WORDS = 20

EXCLUDED_SECTION_TERMS = {
    "ref",
    "refs",
    "reference",
    "references",
    "bibliography",
    "ack",
    "acknowledgment",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",
    "funding",
    "author contribution",
    "author contributions",
    "competing interest",
    "competing interests",
    "conflict of interest",
    "conflicts of interest",
    "supplement",
    "supplementary",
    "supplementary material",
    "supplementary information",
}

STANDALONE_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "results",
    "discussion",
    "methods",
    "materials and methods",
    "conclusion",
    "conclusions",
    "references",
}


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_bioc_documents(data: Any) -> list[dict[str, Any]]:
    """
    Handles BioC structure:

    list
      -> collection
         -> documents
            -> passages
    """
    documents: list[dict[str, Any]] = []

    if isinstance(data, list):
        for item in data:
            documents.extend(extract_bioc_documents(item))
        return documents

    if not isinstance(data, dict):
        return documents

    if isinstance(data.get("passages"), list):
        documents.append(data)
        return documents

    if isinstance(data.get("documents"), list):
        for document in data["documents"]:
            documents.extend(extract_bioc_documents(document))
        return documents

    for value in data.values():
        if isinstance(value, (list, dict)):
            documents.extend(extract_bioc_documents(value))

    return documents


def normalize_pmcid(value: Any) -> str:
    value = clean_text(str(value))

    if not value:
        return ""

    value_upper = value.upper()

    if value_upper.startswith("PMC"):
        return value_upper

    if value.isdigit():
        return f"PMC{value}"

    return value


def extract_pmcid(
    document: dict[str, Any],
    passages: list[dict[str, Any]],
    fallback: str,
) -> str:
    document_id = normalize_pmcid(document.get("id", ""))

    if document_id.startswith("PMC"):
        return document_id

    for passage in passages:
        infons = passage.get("infons", {}) or {}

        for key in (
            "pmcid",
            "article-id_pmc",
            "article_id",
            "article-id",
        ):
            candidate = normalize_pmcid(infons.get(key, ""))

            if candidate.startswith("PMC"):
                return candidate

    if document_id:
        return document_id

    return normalize_pmcid(fallback)


def get_section_name(passage: dict[str, Any]) -> str:
    infons = passage.get("infons", {}) or {}

    for key in (
        "section",
        "section_title",
        "section_type",
        "type",
    ):
        value = infons.get(key)

        if value:
            return clean_text(str(value))

    return "Unknown"


def find_title(
    passages: list[dict[str, Any]],
    fallback: str,
) -> str:
    for passage in passages:
        infons = passage.get("infons", {}) or {}

        metadata_text = " ".join(
            str(infons.get(key, "")).lower()
            for key in (
                "type",
                "section",
                "section_title",
                "section_type",
            )
        )

        if "title" in metadata_text:
            title = clean_text(str(passage.get("text", "")))

            if title:
                return title

    for passage in passages:
        text = clean_text(str(passage.get("text", "")))
        word_count = len(text.split())

        if 4 <= word_count <= 60:
            return text

    return fallback


def should_exclude_section(section: str) -> bool:
    """
    Exclude references, acknowledgements, supplementary material,
    funding statements, and other sections that should not be used
    as biological evidence.
    """
    normalized = clean_text(section).lower().strip(" .:-_")

    return any(
        normalized == term
        or normalized.startswith(term + " ")
        or normalized.startswith(term + "_")
        for term in EXCLUDED_SECTION_TERMS
    )


def is_low_information_text(text: str) -> bool:
    normalized = text.lower().strip(" .:")

    if normalized in STANDALONE_HEADINGS:
        return True

    if len(text.split()) < MIN_PASSAGE_WORDS:
        return True

    return False

def looks_like_reference_entry(text: str) -> bool:
    """
    Detect bibliography-style entries that may not be correctly labelled
    as REF in the BioC metadata.
    """
    normalized = clean_text(text)

    reference_patterns = [
        r"^\w[\w\-']+ [A-Z](?:,|\.)(?:\s+[A-Z](?:,|\.))*",
        r"\bdoi:\s*10\.",
        r"\bPMID:\s*\d+",
        r"\bbioRxiv\b",
        r"\bet al\.\s+\(\d{4}\)",
        r"\b\d{4};\d+\(\d+\):\d+",
    ]

    matches = sum(
        bool(re.search(pattern, normalized, flags=re.IGNORECASE))
        for pattern in reference_patterns
    )

    return matches >= 2


def split_into_chunks(
    text: str,
    max_words: int = MAX_CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[str]:
    words = text.split()

    if not words:
        return []

    if len(words) <= max_words:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(0, end - overlap_words)

    return chunks


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(PAPERS_DIR.rglob("*.json"))

    print("=" * 72)
    print("HumanCellAI - BioC Corpus Builder")
    print("=" * 72)
    print("BioC JSON files found:", len(json_files))

    if not json_files:
        raise FileNotFoundError(
            "No JSON files found under documents/papers."
        )

    all_chunks: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    for json_file in json_files:
        print("\n" + "-" * 72)
        print("Processing:", json_file)

        try:
            with open(json_file, "r", encoding="utf-8") as handle:
                article_data = json.load(handle)

        except OSError as error:
            print("Skipped: file could not be opened:", error)
            continue

        except json.JSONDecodeError as error:
            print("Skipped: invalid JSON:", error)
            continue

        documents = extract_bioc_documents(article_data)

        print("BioC documents detected:", len(documents))

        if not documents:
            print("Skipped: no BioC documents detected.")
            continue

        category = json_file.parent.name

        for document_number, document in enumerate(
            documents,
            start=1,
        ):
            passages = document.get("passages", []) or []

            pmcid = extract_pmcid(
                document=document,
                passages=passages,
                fallback=json_file.stem,
            )

            title = find_title(
                passages=passages,
                fallback=json_file.stem.replace("_", " "),
            )

            indexed_passages = 0
            paper_chunk_count = 0

            for passage_number, passage in enumerate(
                passages,
                start=1,
            ):
                text = clean_text(str(passage.get("text", "")))

                if not text:
                    continue

                if is_low_information_text(text):
                    continue
                
                if looks_like_reference_entry(text):
                    continue
                
                section = get_section_name(passage)

                if should_exclude_section(section):
                    continue

                passage_chunks = split_into_chunks(text)

                if not passage_chunks:
                    continue

                indexed_passages += 1

                for local_chunk_number, chunk_text in enumerate(
                    passage_chunks,
                    start=1,
                ):
                    paper_chunk_count += 1

                    chunk_id = (
                        f"{pmcid}_"
                        f"d{document_number}_"
                        f"p{passage_number}_"
                        f"c{local_chunk_number}"
                    )

                    record = {
                        "chunk_id": chunk_id,
                        "pmcid": pmcid,
                        "title": title,
                        "category": category,
                        "section": section,
                        "passage_number": passage_number,
                        "chunk_number": local_chunk_number,
                        "source_file": str(json_file),
                        "text": chunk_text,
                        "word_count": len(chunk_text.split()),
                    }

                    all_chunks.append(record)

            metadata_rows.append(
                {
                    "pmcid": pmcid,
                    "title": title,
                    "category": category,
                    "source_file": str(json_file),
                    "passages_indexed": indexed_passages,
                    "chunks_created": paper_chunk_count,
                }
            )

            print("PMCID:", pmcid)
            print("Title:", title)
            print("Passages indexed:", indexed_passages)
            print("Chunks created:", paper_chunk_count)

    if not all_chunks:
        raise RuntimeError(
            "No chunks were created from the available papers."
        )

    with open(CHUNKS_FILE, "w", encoding="utf-8") as handle:
        for record in all_chunks:
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    metadata_fields = [
        "pmcid",
        "title",
        "category",
        "source_file",
        "passages_indexed",
        "chunks_created",
    ]

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=metadata_fields,
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\n" + "=" * 72)
    print("Corpus creation completed successfully")
    print("=" * 72)
    print("Papers processed:", len(metadata_rows))
    print("Total chunks:", len(all_chunks))
    print("Chunks file:", CHUNKS_FILE)
    print("Metadata file:", METADATA_FILE)


if __name__ == "__main__":
    main()