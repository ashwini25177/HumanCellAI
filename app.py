"""
HumanCellAI - Free Deployment Version

This deployment uses:
- PMC research-paper corpus
- Sentence Transformer embeddings
- FAISS semantic retrieval
- Cross-encoder reranking
- Extractive citation-safe answers

It does not require OpenAI, Ollama, an API key, or a paid service.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import streamlit as st

from src.retrieval.search_papers import (
    load_resources,
    retrieve_evidence,
)


MIN_RERANKER_SCORE = 0.5
MIN_SEMANTIC_SCORE = 0.30
MAX_EVIDENCE = 4

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that",
    "the", "this", "to", "was", "what", "which", "with",
}


st.set_page_config(
    page_title="HumanCellAI",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 HumanCellAI")
st.caption(
    "A research-paper-grounded assistant for human cell development "
    "and single-cell transcriptomics"
)

st.info(
    "This deployment produces extractive answers from indexed research "
    "papers. It does not generate unsupported biological information."
)


@st.cache_resource(show_spinner="Loading scientific retrieval models...")
def load_rag_resources():
    return load_resources()


def normalize_sentence(sentence: str) -> str:
    """Remove list numbering and normalize whitespace."""

    sentence = re.sub(
        r"^\s*(?:\d+[\.\)]|[-•])\s*",
        "",
        sentence,
    )

    sentence = re.sub(r"\s+", " ", sentence)

    return sentence.strip()


def split_sentences(text: str) -> list[str]:
    """Split a passage into usable sentences."""

    raw_sentences = re.split(
        r"(?<=[.!?])\s+",
        str(text).strip(),
    )

    sentences = []

    for sentence in raw_sentences:
        sentence = normalize_sentence(sentence)

        if len(sentence.split()) >= 8:
            sentences.append(sentence)

    return sentences


def question_terms(question: str) -> set[str]:
    """Extract useful terms from the user's question."""

    words = re.findall(
        r"[A-Za-z0-9_-]+",
        question.lower(),
    )

    return {
        word
        for word in words
        if word not in STOPWORDS and len(word) > 2
    }


def choose_best_sentence(
    question: str,
    passage: str,
) -> str:
    """
    Choose the sentence with the greatest lexical overlap
    with the question.
    """

    sentences = split_sentences(passage)

    if not sentences:
        return normalize_sentence(passage)

    terms = question_terms(question)

    if not terms:
        return sentences[0]

    sentence_scores = []

    for sentence in sentences:
        sentence_words = set(
            re.findall(
                r"[A-Za-z0-9_-]+",
                sentence.lower(),
            )
        )

        overlap = len(terms.intersection(sentence_words))

        sentence_scores.append(
            (
                overlap,
                -len(sentence.split()),
                sentence,
            )
        )

    sentence_scores.sort(reverse=True)

    return sentence_scores[0][2]


def filter_accepted_evidence(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reject weak and unsupported retrieval results."""

    accepted = []
    paper_counts: Counter[str] = Counter()

    for result in results:
        reranker_score = float(
            result.get("reranker_score", -999.0)
        )

        semantic_score = float(
            result.get("semantic_score", 0.0)
        )

        text = str(result.get("text", "")).strip()
        pmcid = str(result.get("pmcid", "Unknown"))

        if not text:
            continue

        if reranker_score < MIN_RERANKER_SCORE:
            continue

        if semantic_score < MIN_SEMANTIC_SCORE:
            continue

        # Maximum two pieces of evidence from one paper.
        if paper_counts[pmcid] >= 2:
            continue

        accepted.append(result)
        paper_counts[pmcid] += 1

        if len(accepted) >= MAX_EVIDENCE:
            break

    return accepted


def create_extractive_answer(
    question: str,
    evidence: list[dict[str, Any]],
) -> str:
    """Create a citation-safe answer directly from paper sentences."""

    if not evidence:
        return (
            "The indexed research papers do not provide sufficient "
            "evidence to answer this question."
        )

    answer_lines = []
    source_lines = []

    for number, result in enumerate(evidence, start=1):
        sentence = choose_best_sentence(
            question=question,
            passage=str(result.get("text", "")),
        )

        sentence = sentence.rstrip()

        if sentence and sentence[-1] not in ".!?":
            sentence += "."

        answer_lines.append(f"{sentence} [{number}]")

        source_lines.append(
            f"[{number}] {result.get('title', 'Unknown paper')} — "
            f"{result.get('pmcid', 'Unknown')} — "
            f"{result.get('section', 'Unknown')}"
        )

    return (
        "### Evidence-based answer\n\n"
        + "\n\n".join(answer_lines)
        + "\n\n### Limitations\n\n"
        + "This answer is extracted from the currently indexed papers. "
        + "It should not be interpreted beyond the supplied evidence."
        + "\n\n### Sources\n\n"
        + "\n\n".join(source_lines)
    )


def display_evidence(
    evidence: list[dict[str, Any]],
) -> None:
    """Show exact retrieved passages for verification."""

    if not evidence:
        return

    st.markdown("### Supporting evidence")

    for number, result in enumerate(evidence, start=1):
        title = str(result.get("title", "Unknown paper"))
        pmcid = str(result.get("pmcid", "Unknown"))
        section = str(result.get("section", "Unknown"))

        with st.expander(
            f"[{number}] {title} — {pmcid}",
            expanded=(number == 1),
        ):
            st.write(f"**Section:** {section}")
            st.write(
                f"**Passage number:** "
                f"{result.get('passage_number', 'Unknown')}"
            )

            st.write(
                f"**Semantic score:** "
                f"{float(result.get('semantic_score', 0.0)):.4f}"
            )

            st.write(
                f"**Reranker score:** "
                f"{float(result.get('reranker_score', 0.0)):.4f}"
            )

            st.markdown("**Exact paper passage:**")
            st.write(result.get("text", ""))

            if pmcid.startswith("PMC"):
                url = (
                    "https://pmc.ncbi.nlm.nih.gov/articles/"
                    f"{pmcid}/"
                )

                st.markdown(
                    f"[Open article in PubMed Central]({url})"
                )


try:
    (
        index,
        records,
        embedding_model,
        reranker,
        configuration,
    ) = load_rag_resources()

except Exception as error:
    st.error("The scientific retrieval system could not be loaded.")
    st.exception(error)
    st.stop()


with st.sidebar:
    st.header("Knowledge base")

    st.metric(
        "Indexed passages",
        int(index.ntotal),
    )

    st.write(
        "**Embedding model:**",
        configuration["model_name"],
    )

    st.write(
        "**Current subjects:**",
        "Human heart development, pseudotime, Slingshot, "
        "Palantir, Monocle and RNA velocity.",
    )

    st.caption(
        "Add new PMC BioC papers and rebuild the corpus and index "
        "to expand the assistant."
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("evidence"):
            display_evidence(message["evidence"])


question = st.chat_input(
    "Ask about human cell development or single-cell RNA-seq..."
)


if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching indexed research papers..."):
            raw_results = retrieve_evidence(
                question=question,
                index=index,
                records=records,
                embedding_model=embedding_model,
                reranker=reranker,
            )

            evidence = filter_accepted_evidence(raw_results)

            answer = create_extractive_answer(
                question=question,
                evidence=evidence,
            )

        st.markdown(answer)
        display_evidence(evidence)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "evidence": evidence,
        }
    )