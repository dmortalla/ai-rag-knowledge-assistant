
# AI RAG Knowledge Assistant

A full-stack Retrieval-Augmented Generation (RAG) application that answers questions using a custom knowledge base angit diff --check

git status --shortd provides source attribution for retrieved context.

The project combines document ingestion, text chunking, OpenAI embeddings, FAISS vector search, grounded prompt construction, OpenAI generation, a FastAPI backend, and a Streamlit interface in a modular Python architecture.

## Overview

Large language models can generate answers from their pretrained knowledge even when the information required by an application is missing or domain-specific.

Retrieval-Augmented Generation addresses this by retrieving relevant information from a knowledge base and supplying that context to the language model during generation.

This application implements the following workflow:

```text
Raw Documents
      |
      v
Document Ingestion
      |
      v
Text Chunking
      |
      v
OpenAI Embeddings
      |
      v
FAISS Vector Store
      |
      v
Semantic Retrieval
      |
      v
Prompt Construction
      |
      v
OpenAI Chat Model
      |
      v
Grounded Answer + Sources
```

Users interact with the system through a Streamlit frontend backed by a FastAPI API.

## Key Features

### Retrieval-Augmented Generation

The application retrieves relevant chunks from a custom knowledge base before generating an answer.

The retrieved context is passed to the language model so responses can be grounded in application-specific information rather than relying solely on the model's pretrained knowledge.

### Semantic Retrieval

Documents are converted into embeddings using OpenAI's embedding API and indexed in FAISS.

At query time, the application performs vector similarity search to retrieve the most relevant chunks.

### Source Attribution

The API and user interface expose the sources used during retrieval.

Retrieved records include metadata such as:

- source document
- chunk identifier
- retrieved content

This makes it possible to inspect the context supplied to the language model.

### Grounding Signal

The Streamlit interface displays a retrieval-based grounding signal.

This signal is a simple heuristic based on the amount of retrieved source context. It is intended to communicate retrieval support and is **not a calibrated probability that an answer is correct**.

### Insufficient-Context Behavior

The prompt instructs the model to rely on the supplied knowledge-base context.

When the retrieved information does not support an answer, the application can respond that the information is not available in the knowledge base rather than filling the gap with unsupported information.

### Backend-Agnostic Vector Store Interface

Vector storage is isolated behind a `BaseVectorStore` abstraction.

The current implementation uses FAISS, while the rest of the RAG pipeline interacts with the vector store through the shared interface.

This architecture makes additional vector-store providers easier to introduce without coupling retrieval and orchestration logic directly to FAISS.

### REST API

FastAPI provides application endpoints including:

```text
GET  /health
POST /query
```

The query endpoint returns:

- the original query
- retrieved context chunks
- source metadata
- the generated answer

### Streamlit Interface

The frontend provides:

- question input
- generated answers
- retrieved context
- source attribution
- retrieval and grounding information
- query-term highlighting
- conversation/session behavior
- backend status indicators

## Architecture

The runtime request flow is:

```text
                         KNOWLEDGE-BASE BUILD

Raw Text Documents
        |
        v
Document Loader
        |
        v
Text Chunker
        |
        v
OpenAI Embedder
        |
        v
VectorStore Interface
        |
        v
FAISS Index


                           QUERY FLOW

User
 |
 v
Streamlit UI
 |
 v
FastAPI
 |
 v
RAG Pipeline
 |
 +--------------------+
 |                    |
 v                    |
Retriever             |
 |                    |
 v                    |
VectorStore Interface |
 |                    |
 v                    |
FAISS                  |
 |                    |
 v                    |
Retrieved Context -----+
 |
 v
Prompt Builder
 |
 v
Answer Generator
 |
 v
OpenAI Chat Model
 |
 v
Answer + Sources
 |
 v
FastAPI
 |
 v
Streamlit UI
```

## Project Structure

```text
ai-rag-knowledge-assistant/
|
|-- app/
|   |-- api/
|   |   |-- main.py
|   |   |-- routes.py
|   |   `-- schemas.py
|   |
|   |-- core/
|   |   |-- config.py
|   |   `-- paths.py
|   |
|   |-- embeddings/
|   |   `-- openai_embedder.py
|   |
|   |-- generation/
|   |   |-- answer_generator.py
|   |   |-- llm_client.py
|   |   `-- prompt_builder.py
|   |
|   |-- ingestion/
|   |   `-- loaders.py
|   |
|   |-- orchestration/
|   |   `-- rag_pipeline.py
|   |
|   |-- processing/
|   |   `-- chunking.py
|   |
|   |-- retrieval/
|   |   `-- retriever.py
|   |
|   |-- ui/
|   |   `-- streamlit_app.py
|   |
|   |-- utils/
|   |
|   `-- vectorstore/
|       |-- base.py
|       |-- faiss_store.py
|       `-- store_manager.py
|
|-- assets/
|
|-- data/
|   |-- raw/
|   |   `-- customer_retention.txt
|   |
|   |-- processed/
|   |
|   `-- vector_store/        # Generated locally and ignored by Git
|
|-- scripts/
|   |-- build_vector_store.py
|   `-- ingest_documents.py
|
|-- tests/
|   |-- test_answer_generator.py
|   |-- test_api.py
|   |-- test_chunking.py
|   |-- test_embedder.py
|   |-- test_llm_client.py
|   |-- test_prompt_builder.py
|   |-- test_rag_pipeline.py
|   |-- test_retriever.py
|   `-- test_vectorstore.py
|
|-- .env.example
|-- .gitignore
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Technology Stack

| Component       | Technology                 |
| --------------- | -------------------------- |
| Language        | Python 3.11                |
| LLM             | OpenAI                     |
| Chat model      | `gpt-4o-mini`            |
| Embeddings      | `text-embedding-3-small` |
| Vector database | FAISS                      |
| RAG integration | LangChain                  |
| Backend         | FastAPI                    |
| Frontend        | Streamlit                  |
| Testing         | pytest                     |

Model names are configurable through environment variables and therefore can be changed without modifying application code.

## Getting Started

The following instructions use Conda and Windows PowerShell.

### 1. Clone the repository

```powershell
git clone https://github.com/dmortalla/ai-rag-knowledge-assistant.git
cd ai-rag-knowledge-assistant
```

### 2. Create a Python environment

Python 3.11 is the currently verified development version.

```powershell
conda create -n rag_env python=3.11
conda activate rag_env
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example configuration:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and provide your OpenAI API key:

```text
OPENAI_API_KEY=your_openai_api_key_here
VECTOR_DB=faiss
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
TOP_K=4
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

The `.env` file is excluded from Git and should not be committed.

## Building the Knowledge Base

The repository contains a small example knowledge-base document:

```text
data/raw/customer_retention.txt
```

Build the FAISS vector index with:

```powershell
python scripts\build_vector_store.py
```

The generated FAISS artifacts are stored under:

```text
data/vector_store/faiss_index/
```

Generated vector-store artifacts are ignored by Git and can be rebuilt from the source documents.

## Running the Application

The backend and frontend run as separate local processes.

### Start FastAPI

From the project root:

```powershell
python -m uvicorn app.api.main:app --reload
```

The API is available locally at:

```text
http://127.0.0.1:8000
```

### Start Streamlit

Open another terminal, activate the same environment, and run:

```powershell
python -m streamlit run app\ui\streamlit_app.py
```

Streamlit will display the local application URL in the terminal.

## API Example

With the FastAPI server running, a query can be submitted from PowerShell:

```powershell
$body = @{
    query = "What is customer churn?"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/query" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body |
    ConvertTo-Json -Depth 10
```

A successful response contains:

```text
query
context_chunks
sources
answer
```

## Example Queries

The included sample knowledge base supports questions related to customer retention and churn.

Examples include:

```text
What is customer churn?
```

and:

```text
What is customer retention?
```

A question requiring information that is not sufficiently represented in the knowledge base may produce an insufficient-context response rather than an unsupported answer.

## Testing

Run the automated test suite with:

```powershell
python -m pytest
```

The current verified baseline contains:

```text
34 passed
```

The tests cover major components including:

- chunking
- embedding behavior
- LLM client behavior
- prompt construction
- retrieval
- RAG orchestration
- API behavior
- FAISS vector-store behavior and guardrails

### Compile Validation

Python source compilation can also be checked with:

```powershell
python -m compileall app tests scripts
```

### Git Whitespace Validation

Before committing changes:

```powershell
git diff --check
```

## Verified End-to-End Behavior

The application has been manually exercised through the complete local workflow:

```text
Raw document
    ->
Chunking
    ->
OpenAI embeddings
    ->
FAISS persistence
    ->
Semantic retrieval
    ->
Prompt construction
    ->
OpenAI generation
    ->
FastAPI
    ->
Streamlit
```

The verified workflow includes both:

1. a query supported by the knowledge base, producing a grounded answer with source metadata; and
2. a query for which the available context was insufficient, producing an explicit knowledge-base limitation rather than an unsupported answer.

## Design Decisions

### Modular RAG Components

Ingestion, chunking, embeddings, retrieval, prompting, generation, orchestration, API handling, and UI rendering are separated into dedicated modules.

This keeps responsibilities explicit and makes individual components easier to test or replace.

### Dependency Injection

Several components accept injected dependencies, allowing deterministic test doubles to replace network-backed services during unit tests.

### Vector-Store Abstraction

The application does not require retrieval code to construct FAISS directly.

Instead, vector-store operations are defined through a shared abstraction and selected through a factory.

### Configuration Through Environment Variables

Model names, retrieval depth, chunk size, chunk overlap, and vector-store selection are centralized in application configuration rather than scattered throughout the codebase.

### Generated Artifacts Stay Out of Git

FAISS index files and local environment secrets are excluded from version control.

The vector index can be reproduced from the raw source documents.

## Current Limitations

The current version intentionally keeps the knowledge base and deployment architecture small.

Current limitations include:

- FAISS is the only implemented vector-store backend.
- Knowledge-base ingestion currently focuses on local text documents.
- The included demonstration corpus is intentionally small.
- The grounding signal is heuristic and is not a calibrated correctness probability.
- The application does not currently implement authentication.
- Automated RAG evaluation metrics are not yet implemented.
- Responses are not currently streamed token-by-token.
- The application is currently verified as a local deployment.

## Potential Extensions

Possible future improvements include:

- Pinecone vector-store backend
- additional document formats
- document upload workflow
- automated retrieval and generation evaluation
- reranking
- hybrid lexical/vector retrieval
- authentication and authorization
- streaming responses
- containerization
- CI/CD
- observability and tracing
- cloud deployment

The existing vector-store abstraction provides a foundation for adding another vector backend without coupling the rest of the RAG pipeline directly to a specific provider.

## What This Project Demonstrates

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation
- semantic search
- vector embeddings
- vector databases
- prompt grounding
- source attribution
- modular AI application architecture
- backend API development
- frontend integration
- dependency injection
- configuration management
- automated testing
- defensive validation
- end-to-end AI application integration

## Author

**Darrell Mortalla**

AI / ML Engineer

GitHub: https://github.com/dmortalla
