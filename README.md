# HumanCellAI

## Citation-Grounded Retrieval-Augmented Generation (RAG) System for Human Cell Development and Single-Cell RNA Sequencing Research

HumanCellAI is an evidence-grounded Retrieval-Augmented Generation (RAG) system designed for scientific question answering in computational biology. The system retrieves relevant information from peer-reviewed research articles and generates answers using a local large language model while grounding every response in published scientific evidence.

Unlike conventional AI chatbots that may hallucinate unsupported information, HumanCellAI combines semantic retrieval, cross-encoder reranking, and local language model generation to provide scientifically grounded answers with supporting citations.

---

# Overview

HumanCellAI was developed to assist researchers, students, and computational biologists working in:

- Human cell development
- Developmental biology
- Single-cell RNA sequencing (scRNA-seq)
- Trajectory inference
- Cell differentiation
- Cell fate determination
- Human heart development

The system currently indexes peer-reviewed literature from PubMed Central (PMC) and retrieves the most relevant scientific passages before generating an answer.

---

# Key Features

- Citation-grounded Retrieval-Augmented Generation
- Retrieval from peer-reviewed scientific literature
- Semantic search using Sentence Transformers
- Cross-encoder reranking for evidence refinement
- FAISS vector database for efficient retrieval
- Local large language model using Ollama (Qwen2)
- Human heart developmental atlas integration
- Single-cell RNA-seq literature retrieval
- Expandable research-paper knowledge base
- Streamlit-based interactive interface
- Designed to minimize hallucinations through evidence grounding

---

# System Architecture

```
                    Scientific Papers (PMC)

                              │
                              ▼

                    BioC JSON Paper Parser

                              │
                              ▼

                    Passage Extraction

                              │
                              ▼

                     Text Chunk Generation

                              │
                              ▼

                 SentenceTransformer Embeddings

                              │
                              ▼

                     FAISS Vector Database

                              │
                              ▼

                    Semantic Similarity Search

                              │
                              ▼

                 Cross-Encoder Evidence Reranking

                              │
                              ▼

                  Local LLM (Qwen2 via Ollama)

                              │
                              ▼

                Citation-Grounded Scientific Answer
```

---

# Retrieval Pipeline

The complete retrieval pipeline consists of the following stages:

1. Scientific literature ingestion
2. BioC JSON parsing
3. Passage extraction
4. Text chunking
5. Dense embedding generation
6. FAISS indexing
7. Semantic retrieval
8. Cross-encoder reranking
9. Local LLM generation
10. Citation-grounded answer generation

---

# Technologies Used

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| User Interface | Streamlit |
| Embedding Model | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS |
| Evidence Reranker | Cross-Encoder (MS MARCO MiniLM) |
| Local LLM | Ollama + Qwen2 |
| Scientific Corpus | PubMed Central (PMC) BioC JSON |
| Retrieval Method | Dense Semantic Search |
| AI Framework | Retrieval-Augmented Generation (RAG) |

---

# Scientific Topics Covered

Current indexed literature includes:

## Human Development

- Human fetal development
- Cell differentiation
- Cell fate commitment
- Developmental biology

## Human Heart Development

- Human fetal heart atlas
- Cardiogenesis
- Cardiac development
- Heart cell populations

## Single-Cell Transcriptomics

- scRNA-seq analysis
- Cell clustering
- Marker genes
- Gene expression profiling

## Trajectory Inference

- Monocle
- Slingshot
- Palantir
- RNA Velocity
- Pseudotime analysis

---

# Indexed Research Papers

Current version indexes peer-reviewed publications including:

| Research Area | Publication |
|---------------|-------------|
| Heart Development | Cell atlas of the foetal human heart |
| Human Development | Human fetal gene expression atlas |
| Cardiogenesis | Integrative single-cell analysis of cardiogenesis |
| Trajectory Inference | Slingshot |
| Trajectory Inference | Monocle |
| Trajectory Inference | Palantir |
| Trajectory Inference | RNA Velocity |

Additional papers can be incorporated into the knowledge base without modifying the retrieval pipeline.

---

# Example Questions

The system is capable of answering questions such as:

- What is pseudotime?
- Explain RNA velocity.
- How does Palantir estimate cell fate probabilities?
- What is Slingshot?
- Explain developmental trajectories.
- What is cell differentiation?
- What is dedifferentiation?
- What are cardiomyocyte marker genes?
- Which cell populations are present in the developing human heart?
- Explain lineage inference in single-cell RNA sequencing.

---

# Repository Structure

```
HumanCellAI
│
├── app.py
├── README.md
├── requirements.txt
│
├── data
│   ├── metadata
│   ├── processed
│   └── vectorstore
│
├── documents
│   └── papers
│
└── src
    ├── ingestion
    ├── embedding
    ├── retrieval
    └── generation
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/ashwini25177/HumanCellAI.git
```

Move into the project

```bash
cd HumanCellAI
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

# Running the Application

Launch the Streamlit application

```bash
streamlit run app.py
```

---

# Citation-Grounded Answer Generation

For every user query, HumanCellAI performs:

1. Semantic retrieval of relevant scientific passages.
2. Evidence reranking using a cross-encoder model.
3. Local answer generation using Ollama (Qwen2).
4. Validation that the generated answer is supported by retrieved evidence.
5. Presentation of supporting citations to the user.

If the indexed literature does not contain sufficient evidence, the system reports that it cannot answer the question reliably instead of generating unsupported content.

---

# Future Work

Future extensions planned for HumanCellAI include:

- Human Cell Atlas integration
- Cell Ontology integration
- Gene Ontology enrichment
- BioBERT and PubMedBERT embeddings
- Hybrid BM25 + Dense Retrieval
- Multi-paper evidence aggregation
- Automatic figure retrieval
- PDF evidence highlighting
- Live PubMed search
- Marker gene database
- Spatial transcriptomics support
- Multi-modal scientific retrieval

---

# Research Applications

HumanCellAI can be applied to:

- Computational Biology
- Bioinformatics
- Single-cell Transcriptomics
- Developmental Biology
- Human Cell Atlas Research
- Cardiovascular Research
- Graduate Research Projects
- Scientific Literature Review
- Biomedical Education

---

# Author

**Ashwini Gudekar**

M.Tech in Computational Biology

Indraprastha Institute of Information Technology Delhi (IIIT-Delhi)

---

# Acknowledgements

The scientific literature indexed in this project has been obtained from publicly available peer-reviewed publications hosted by:

- PubMed Central (PMC)
- National Center for Biotechnology Information (NCBI)
- Human Cell Atlas Consortium

The original scientific findings remain the intellectual property of their respective authors.

---

# License

This project is intended for academic, educational, and research purposes.

Users are encouraged to cite the original publications when using scientific findings retrieved through HumanCellAI.
