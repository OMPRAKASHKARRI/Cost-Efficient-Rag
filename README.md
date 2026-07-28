# Cost-Efficient RAG Application

A production-ready Retrieval-Augmented Generation (RAG) application built with **FastAPI**, **LanceDB**, **Sentence Transformers**, and **Groq LLMs**. The system ingests PDF, HTML, and Markdown documents, retrieves relevant context using semantic search, and generates grounded responses with source citations.

---

## Features

- Document ingestion for PDF, HTML, and Markdown files
- Automatic document chunking with configurable chunk size and overlap
- Semantic embeddings using Sentence Transformers
- Vector storage using LanceDB
- Similarity-based retrieval with configurable thresholds
- Grounded answer generation using Groq LLM
- Source citations for every generated response
- Hallucination prevention through fallback responses
- Duplicate chunk detection for idempotent ingestion
- Token usage tracking and cost estimation
- Query latency metrics
- REST API built with FastAPI

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI |
| Vector Database | LanceDB |
| Embedding Model | all-MiniLM-L6-v2 |
| LLM Provider | Groq |
| LLM Model | openai/gpt-oss-120b |
| Document Parsing | PyPDF2, BeautifulSoup |
| Language | Python 3.11+ |

---

## Project Structure

```text
cost-efficient-rag/
│
├── data/
│   ├── lancedb_store/
│   └── raw_documents/
│
├── logs/
│
├── src/
│   ├── api.py
│   ├── config.py
│   ├── ingestion.py
│   ├── logger.py
│   ├── rag_pipeline.py
│   └── vector_store.py
│
├── tests/
│
├── requirements.txt
├── .env.example
└── README.md
```

---

# Architecture

```text
             Upload Documents
                    │
                    ▼
          Document Loader
                    │
                    ▼
          Text Chunking
                    │
                    ▼
      Sentence Transformer
          Embeddings
                    │
                    ▼
              LanceDB
          Vector Database
                    │
                    ▼
             User Query
                    │
                    ▼
      Semantic Retrieval
                    │
                    ▼
      Retrieved Context
                    │
                    ▼
       Groq LLM (GPT-OSS)
                    │
                    ▼
       Grounded Response
        + Source Citations
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/yourusername/cost-efficient-rag.git

cd cost-efficient-rag
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file.

```env
LLM_PROVIDER=groq

GROQ_API_KEY=your_groq_api_key

LLM_MODEL_NAME=openai/gpt-oss-120b

VECTOR_STORE_PATH=./data/lancedb_store

VECTOR_TABLE_NAME=rag_chunks

EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

DEFAULT_CHUNK_SIZE=500

DEFAULT_CHUNK_OVERLAP=50

DEFAULT_TOP_K=5

SIMILARITY_THRESHOLD=0.35
```

---

# Running the Application

```bash
uvicorn src.api:app --reload
```

Server

```
http://127.0.0.1:8000
```

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## Health Check

```http
GET /health
```

Response

```json
{
    "status": "ok"
}
```

---

## Ingest Documents

```http
POST /ingest
```

Upload one or more files using **multipart/form-data**.

Supported formats

- PDF
- HTML
- Markdown

Example Response

```json
{
    "total_chunks_ingested": 12,
    "files": [
        {
            "source": "sample.pdf",
            "chunks_ingested": 12,
            "chunks_skipped_duplicate": 0
        }
    ]
}
```

---

## Query Documents

```http
POST /query
```

Example Request

```json
{
    "query": "What is Artificial Intelligence?",
    "top_k": 3
}
```

Example Response

```json
{
    "answer": "...",
    "citations": [
        {
            "source": "sample.pdf",
            "chunk_id": "...",
            "similarity": 0.72
        }
    ],
    "fallback_triggered": false,
    "retrieved_chunk_count": 3,
    "retrieval_latency_ms": 122,
    "generation_latency_ms": 2015,
    "total_latency_ms": 2140,
    "prompt_tokens": 451,
    "completion_tokens": 356,
    "estimated_cost_usd": 0.00028
}
```

---

# Retrieval Pipeline

1. User uploads documents.
2. Documents are parsed.
3. Text is chunked.
4. Embeddings are generated.
5. Chunks are stored in LanceDB.
6. User submits a query.
7. Query embedding is generated.
8. Similar chunks are retrieved.
9. Retrieved context is passed to Groq.
10. A grounded answer with citations is returned.

---

# Hallucination Prevention

The application minimizes hallucinations using:

- Similarity threshold filtering
- Retrieval-Augmented Generation (RAG)
- Strict grounding prompts
- Citation-based responses
- Fallback response when insufficient context exists

Fallback message:

```
I do not have sufficient information in the provided context to answer this question.
```

---

# Cost Estimation

Each query reports an estimated inference cost based on:

- Prompt tokens
- Completion tokens
- Configurable per-million token pricing

This enables transparent monitoring of LLM usage.

---

# Performance Metrics

Each query includes:

- Retrieval latency
- Generation latency
- Total latency
- Retrieved chunk count
- Token usage
- Estimated cost
- Fallback status

---

# Future Improvements

- Hybrid Search (Keyword + Semantic)
- Streaming responses
- Conversation memory
- Authentication
- Docker support
- Kubernetes deployment
- Redis caching
- Evaluation dashboard
- Multiple LLM providers
- OCR support

---

# Author

**Omprakash Karri**

Backend Developer | AI Enthusiast

GitHub: https://github.com/OMPRAKASHKARRI

LinkedIn: https://www.linkedin.com/in/your-linkedin

---

# License

This project was developed as part of a technical assessment and is intended for educational and demonstration purposes.