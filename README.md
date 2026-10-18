# Banking Support AI

A retrieval-augmented generation (RAG) application that answers banking-support questions using a small synthetic policy knowledge base. The project demonstrates practical LLM engineering: document ingestion, semantic retrieval, grounded generation, citations, abstention, LangGraph orchestration, FastAPI, testing, Docker, and GitHub Actions CI.

> **Status:** Local development and container smoke tests have been completed, and CI has been added. Generation evaluation and deployment readiness remain in progress. The policy documents are synthetic; this application is not connected to a real bank.

## Features

- **Policy ingestion:** Loads Markdown policies, splits them into overlapping chunks, and preserves source metadata.
- **Semantic retrieval:** Uses `sentence-transformers/all-MiniLM-L6-v2` embeddings with a persistent Chroma vector store.
- **Grounded generation:** Uses a hosted OpenRouter-compatible chat model instructed to answer only from retrieved evidence.
- **Citations and abstention:** Returns source identifiers such as `[S1]` and is designed to abstain when evidence is insufficient.
- **Workflow orchestration:** Uses LangGraph to connect retrieval and generation.
- **API:** FastAPI endpoints for the web interface, health checks, and chat.
- **Web interface:** Chat UI with suggested questions and an evidence panel for returned sources.
- **Testing and CI:** pytest suite and GitHub Actions workflow that does not require a live model API key.
- **Containerization:** Dockerfile and Docker Compose configuration for local runs and persistent Chroma data.

## Architecture

```text
User
 |
 v
FastAPI
 |-- GET /          Web interface
 |-- GET /health    Health check
 `-- POST /chat
          |
          v
 BankingSupportService
          |
          v
       LangGraph
          |
          +--> PolicyRetriever
          |       |
          |       +--> Hugging Face embeddings
          |       `--> Chroma vector store
          |
          `--> GroundedAnswerGenerator
                  |
                  `--> OpenRouter-compatible chat model
          |
          v
 Response: answer, sources, abstained, error
```

## Technology stack

| Area | Technology |
|---|---|
| Language | Python 3.11 |
| API | FastAPI, Uvicorn |
| LLM orchestration | LangChain, LangGraph |
| Vector store | Chroma |
| Embeddings | Hugging Face Sentence Transformers |
| Generation | OpenRouter-compatible API |
| Testing | pytest |
| Containers | Docker, Docker Compose |
| CI | GitHub Actions |

## Knowledge base

The initial synthetic policy set includes:

- `card_fees.md`
- `refunds.md`

Documents are split with `RecursiveCharacterTextSplitter` using a chunk size of 700 and overlap of 100. Metadata connects retrieved chunks to their source documents. These example policies do not represent any real financial institution.

## Initial retrieval baseline

A small evaluation used five question/evidence cases and `k=2`.

| Metric | Initial result |
|---|---:|
| Mean Recall@2 | 1.00 |
| Mean Precision@2 | 0.60 |
| Mean Reciprocal Rank | 1.00 |

This is a small development baseline, not a claim of production-grade retrieval performance.

## Getting started

### Prerequisites

- Python 3.11
- Conda or another Python environment manager
- An OpenRouter API key for live generation
- Docker and Docker Compose for container execution

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Banking-Support-AI
```

Replace `<YOUR_REPOSITORY_URL>` with the actual repository URL.

### 2. Set up the environment

```bash
conda create -n banking-support-genai python=3.11
conda activate banking-support-genai
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the environment already exists, activate it instead of recreating it.

### 3. Configure environment variables

Create a local `.env` file using `.env.example` if the repository includes it. Example names and development defaults recorded during implementation:

```dotenv
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=your_model_identifier
OPENROUTER_MAX_TOKENS=12000
OPENROUTER_TIMEOUT_SECONDS=60
CHROMA_PERSIST_DIRECTORY=data/vector_store/chroma
```

Verify the current implementation and `.env.example` for the authoritative values. Never commit `.env` or expose a real key in documentation or CI.

### 4. Run tests

```bash
python -m pytest -q
```

The API tests mock service initialization so they do not need a live hosted model. Live generation still requires valid runtime model configuration.

### 5. Run the application

Use the current Uvicorn command documented by the repository. The application exposes:

- `GET /` — web interface
- `GET /health` — health check
- `POST /chat` — ask a policy question

Open the local root URL in a browser once the server is running. Use `/health` to verify the API process is responding.

## Docker Compose

After configuring the local `.env`:

```bash
docker compose up --build -d
curl http://localhost:8000/health
docker compose logs
```

Stop the service:

```bash
docker compose down
```

The Compose configuration uses a named volume for Chroma persistence. A normal `docker compose down` preserves named volumes; `docker compose down -v` deletes them. Verify the current port mapping before relying on the health URL above.

## API overview

### `GET /health`

Example response:

```json
{
  "status": "ok"
}
```

### `POST /chat`

Example request:

```json
{
  "question": "What should I do if I am charged twice?"
}
```

The response includes answer text, source metadata, an abstention flag, and an error field. Blank questions and questions longer than 2,000 characters are rejected. Check the current API implementation for the exact response schema.

## Continuous integration

The workflow at `.github/workflows/ci.yml` runs on pushes and pull requests targeting `main` or `master`, and on manual dispatch. It sets up Python 3.11, installs `requirements.txt`, and runs:

```bash
python -m pytest -q
```

The workflow intentionally does not require a real OpenRouter API key. API tests mock service initialization so endpoint tests can run without contacting a hosted model.

## Evaluation and limitations

The first generation evaluation contained eight cases. Recorded initial results:

| Metric | Initial result |
|---|---:|
| Overall pass rate | 12.5% |
| Abstention accuracy | 85.71% |
| Fact-check pass rate | 14.29% |
| Source-coverage pass rate | 85.71% |
| Citation-validity rate | 100% |

The overall pass rate is low and is treated as work to improve, not hidden. Issues identified included overly strict phrase matching, an intermittent empty model response, and an abstention expectation that needs review.

Planned improvements:

- Improve generation evaluation and add broader test cases.
- Investigate empty model responses and abstention behavior.
- Verify the actual frontend's integration with the live API.
- Review logging, health checks, and error handling.
- Review container size and runtime-port configuration.
- Select a cloud platform only after cost and storage needs are understood.

## Security and responsible use

- The included policies are synthetic.
- Do not use this project as a source of real financial advice.
- Never commit API keys or `.env`.
- Inspect cited evidence; generated answers can be wrong.
- Add suitable authentication, rate limiting, secret management, and abuse protection before public deployment.

## Documentation

See [`README_DEVELOPMENT.md`](README_DEVELOPMENT.md) for the detailed engineering journal: project history, architecture decisions, implementation milestones, commands, troubleshooting, test snapshots, CI, and next steps.

## License

Add the license appropriate for this repository before distributing or reusing the project.
