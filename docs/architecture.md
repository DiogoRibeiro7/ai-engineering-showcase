# Architecture

## Goal

The system converts unstructured customer feedback into evidence-grounded product insights.

It is intentionally small, but it uses boundaries that mirror a real production AI engineering system.

## Flow

```text
CSV feedback
   │
   ▼
Validation and ingestion
   │
   ▼
Chunking
   │
   ▼
Hashing embeddings
   │
   ▼
Local vector store
   │
   ▼
Query engine
   │
   ▼
Route selection
   │
   ▼
Grounded prompt
   │
   ▼
LLM provider
   │
   ▼
Cited answer + recommended actions + diagnostics
```

## Components

### Ingestion

`ingestion.py` loads CSV data and validates each row with Pydantic. Invalid rows are reported with line numbers, which makes data quality issues easier to debug.

### Chunking

`chunking.py` splits feedback into overlapping word chunks. The current dataset has short feedback, but the logic also works for longer support tickets or interview transcripts.

### Embeddings

`embeddings.py` implements deterministic feature hashing with unigrams and bigrams. This is useful for local development because it avoids external APIs. In production, this component can be replaced by OpenAI embeddings, Bedrock embeddings, or an internal embedding service.

### Vector store

`vector_store.py` provides cosine search and JSON persistence. It is deliberately simple. Production alternatives include pgvector, OpenSearch, Pinecone, Weaviate, Qdrant, or FAISS.

### Agent

`agent.py` performs query routing, retrieval, prompt building, generation, response parsing, citation construction, and confidence scoring.

### LLM provider

`llm.py` defines an `LLMProvider` protocol. The default provider is deterministic and local. The optional OpenAI-compatible provider shows how to connect to external inference while keeping the rest of the system unchanged.

### Evaluation

`evaluation.py` includes retrieval metrics and answer-quality checks. This is important because AI engineering should not stop at prompt writing. Retrieval quality, citation coverage, and grounding need continuous measurement.

### API

`api.py` exposes the system through FastAPI. The API validates input and returns typed responses.

## Extension points

- Replace the hashing embedding model with a neural embedding provider.
- Replace the local vector store with a managed vector database.
- Add a streaming ingestion layer using Kafka, Kinesis, or Pub/Sub.
- Add tracing with OpenTelemetry.
- Add human feedback capture for answer quality.
- Add regression tests for prompts and retrieval behavior.
- Add role-based access control around the API.

## Production considerations

A production version should include:

- Tenant isolation.
- PII redaction before indexing.
- Rate limiting and authentication.
- Prompt injection checks.
- Data lineage for every generated answer.
- Human feedback loops.
- Monitoring for retrieval drift and answer degradation.
- Canary evaluation before prompt or model changes.
