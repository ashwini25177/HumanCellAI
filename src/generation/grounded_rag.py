"""
HumanCellAI - Fail-Safe Grounded Scientific RAG

Pipeline:
    Question
        -> FAISS semantic retrieval
        -> Cross-encoder reranking
        -> Evidence filtering
        -> Local Ollama generation
        -> Citation validation
        -> Safe extractive fallback when validation fails
"""

from __future__ import annotations

import re
import sys
from typing import Any

import ollama

from src.retrieval.search_papers import (
    load_resources,
    retrieve_evidence,
)


MODEL_NAME = "qwen2:1.5b"

MIN_RERANKER_SCORE = 0.0
MAX_EVIDENCE_PASSAGES = 5
MAX_GENERATION_TOKENS = 300


def filter_evidence(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only positively reranked, non-empty evidence."""

    accepted: list[dict[str, Any]] = []

    for result in results:
        reranker_score = float(
            result.get("reranker_score", -999.0)
        )

        text = str(result.get("text", "")).strip()

        if reranker_score < MIN_RERANKER_SCORE:
            continue

        if not text:
            continue

        accepted.append(result)

        if len(accepted) >= MAX_EVIDENCE_PASSAGES:
            break

    return accepted


def build_evidence_context(
    evidence: list[dict[str, Any]],
) -> str:
    """Create numbered citation blocks for the local model."""

    blocks: list[str] = []

    for number, record in enumerate(evidence, start=1):
        blocks.append(
            f"[{number}]\n"
            f"Paper title: {record.get('title', 'Unknown')}\n"
            f"PMCID: {record.get('pmcid', 'Unknown')}\n"
            f"Section: {record.get('section', 'Unknown')}\n"
            f"Evidence text: {record.get('text', '')}"
        )

    return "\n\n".join(blocks)


def create_system_prompt() -> str:
    return """
You are HumanCellAI.

Answer only from the supplied numbered evidence.

Mandatory rules:

1. Every scientific sentence must end with one or more citations:
   [1], [2], or [1][2].
2. Do not use outside knowledge.
3. Do not claim that a paper introduced, discovered, proved, or first
   described something unless that exact claim appears in the evidence.
4. Do not invent genes, methods, statistics, citations, or conclusions.
5. If evidence is insufficient, state:
   "The indexed research papers do not provide sufficient evidence to
   answer this question."
6. Keep the answer to 2-5 sentences.
7. End with a Sources section.

Required structure:

Answer:
<scientific sentences with citations>

Limitations:
<brief limitation>

Sources:
[1] Paper title — PMCID — Section
""".strip()


def validate_generated_answer(
    answer: str,
    evidence_count: int,
) -> tuple[bool, list[str]]:
    """
    Validate that the generated answer contains usable citations and
    avoids common unsupported historical claims.
    """

    problems: list[str] = []

    citation_numbers = [
        int(number)
        for number in re.findall(r"\[(\d+)\]", answer)
    ]

    if not citation_numbers:
        problems.append("No numbered citations were generated.")

    invalid_citations = [
        number
        for number in citation_numbers
        if number < 1 or number > evidence_count
    ]

    if invalid_citations:
        problems.append(
            f"Invalid citation numbers: {invalid_citations}"
        )

    answer_section = answer.split("Sources:", maxsplit=1)[0]

    scientific_sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            answer_section,
        )
        if len(sentence.split()) >= 5
        and not sentence.lower().startswith(
            ("answer:", "limitations:")
        )
    ]

    uncited_sentences = [
        sentence
        for sentence in scientific_sentences
        if not re.search(r"\[\d+\]", sentence)
    ]

    if uncited_sentences:
        problems.append(
            f"{len(uncited_sentences)} scientific sentence(s) "
            "lack citations."
        )

    risky_claims = re.findall(
        r"\b(introduced|discovered|proved|first described|first "
        r"reported|established)\b",
        answer,
        flags=re.IGNORECASE,
    )

    if risky_claims:
        problems.append(
            "Potentially unsupported historical claim detected."
        )

    return len(problems) == 0, problems


def first_sentence(text: str) -> str:
    """Return the first usable sentence from an evidence passage."""

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence.split()) >= 8:
            return sentence

    return text.strip()


def build_safe_fallback_answer(
    question: str,
    evidence: list[dict[str, Any]],
) -> str:
    """
    Produce a citation-safe answer directly from retrieved evidence.

    This fallback is used whenever the local model generates an answer
    without valid citations.
    """

    if not evidence:
        return (
            "The indexed research papers do not provide sufficient "
            "evidence to answer this question."
        )

    statements: list[str] = []

    for number, record in enumerate(evidence[:3], start=1):
        passage = str(record.get("text", "")).strip()

        if not passage:
            continue

        sentence = first_sentence(passage).rstrip()

        if sentence[-1:] not in ".!?":
            sentence += "."

        statements.append(f"{sentence} [{number}]")

    source_lines = []

    for number, record in enumerate(evidence[:3], start=1):
        source_lines.append(
            f"[{number}] {record.get('title', 'Unknown')} — "
            f"{record.get('pmcid', 'Unknown')} — "
            f"{record.get('section', 'Unknown')}"
        )

    return (
        "Answer:\n"
        + " ".join(statements)
        + "\n\nLimitations:\n"
        + "This response is an extractive summary of the retrieved "
        + "passages and should be interpreted within the scope of the "
        + "currently indexed papers."
        + "\n\nSources:\n"
        + "\n".join(source_lines)
    )


def generate_grounded_answer(
    question: str,
    evidence: list[dict[str, Any]],
) -> str:
    """Generate and validate a grounded local-model answer."""

    if not evidence:
        return (
            "The indexed research papers do not provide sufficient "
            "evidence to answer this question."
        )

    evidence_context = build_evidence_context(evidence)

    user_prompt = f"""
Question:
{question}

Evidence:
{evidence_context}

Answer only from the evidence.

Every scientific sentence must end with a citation such as [1] or [2].
Do not claim that any paper introduced or discovered a concept unless
that exact claim is explicitly stated in the supplied evidence.
""".strip()

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": create_system_prompt(),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0.0,
            "num_predict": MAX_GENERATION_TOKENS,
        },
    )

    answer = response["message"]["content"].strip()

    if not answer:
        return build_safe_fallback_answer(
            question=question,
            evidence=evidence,
        )

    is_valid, problems = validate_generated_answer(
        answer=answer,
        evidence_count=len(evidence),
    )

    if not is_valid:
        print("\nGenerated answer failed validation:")
        for problem in problems:
            print(" -", problem)

        print(
            "\nUsing citation-safe extractive fallback instead."
        )

        return build_safe_fallback_answer(
            question=question,
            evidence=evidence,
        )

    return answer


def print_retrieved_evidence(
    evidence: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 80)
    print("RETRIEVED SUPPORTING EVIDENCE")
    print("=" * 80)

    if not evidence:
        print("No acceptable supporting evidence was retrieved.")
        return

    for number, record in enumerate(evidence, start=1):
        print("\n" + "-" * 80)
        print(f"EVIDENCE [{number}]")
        print("-" * 80)

        print("Paper:", record.get("title", "Unknown"))
        print("PMCID:", record.get("pmcid", "Unknown"))
        print("Section:", record.get("section", "Unknown"))
        print(
            "Passage number:",
            record.get("passage_number", "Unknown"),
        )
        print(
            "Semantic score:",
            f"{float(record.get('semantic_score', 0.0)):.4f}",
        )
        print(
            "Reranker score:",
            f"{float(record.get('reranker_score', 0.0)):.4f}",
        )

        print("\nPassage:\n")
        print(record.get("text", ""))

        pmcid = str(record.get("pmcid", "")).strip()

        if pmcid.startswith("PMC"):
            print(
                "\nArticle URL:",
                f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
            )


def main() -> None:
    print("=" * 80)
    print("HumanCellAI - Fail-Safe Grounded Research RAG")
    print("=" * 80)
    print("Loading retrieval and generation resources...")

    try:
        (
            index,
            records,
            embedding_model,
            reranker,
            configuration,
        ) = load_resources()

        print("Scientific passages loaded:", index.ntotal)
        print(
            "Embedding model:",
            configuration["model_name"],
        )
        print("Local generation model:", MODEL_NAME)

        question = input(
            "\nEnter your biological or single-cell question: "
        ).strip()

        if not question:
            print("\nNo question was entered.")
            return

        raw_results = retrieve_evidence(
            question=question,
            index=index,
            records=records,
            embedding_model=embedding_model,
            reranker=reranker,
        )

        evidence = filter_evidence(raw_results)

        print_retrieved_evidence(evidence)

        print("\n" + "=" * 80)
        print("GROUNDED ANSWER")
        print("=" * 80)

        answer = generate_grounded_answer(
            question=question,
            evidence=evidence,
        )

        print("\n" + answer)

    except ConnectionError:
        print(
            "\nERROR: Ollama is not running."
        )
        sys.exit(1)

    except ollama.ResponseError as error:
        print("\nOLLAMA ERROR:")
        print(error)
        sys.exit(1)

    except Exception as error:
        print("\nERROR:")
        print(type(error).__name__, "-", error)
        sys.exit(1)


if __name__ == "__main__":
    main()