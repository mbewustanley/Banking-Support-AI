# Banking Support AI

An LLM-powered banking support platform combining **Retrieval-Augmented Generation (RAG), Model Context Protocol (MCP), and LangGraph** to answer banking-support questions using a synthetic policy knowledge base.

The platform demonstrates practical LLM engineering, including document ingestion, semantic retrieval, grounded generation, citations, abstention, tool integration, conversational state management, FastAPI, automated testing, Docker, and GitHub Actions CI.

> **Project status:** The core RAG workflow, MCP-enabled agent workflow, FastAPI integration, and interactive frontend have been implemented and tested locally. Generation evaluation, persistent conversation storage, and production-readiness improvements remain ongoing. All banking policies are synthetic; this application is not connected to a real financial institution.

## Features

- **Policy ingestion:** Loads Markdown policy documents, splits them into overlapping chunks, and preserves source metadata.
- **Semantic retrieval:** Uses `sentence-transformers/all-MiniLM-L6-v2` embeddings with a persistent Chroma vector store.
- **Grounded generation:** Uses a hosted OpenRouter-compatible chat model instructed to answer using retrieved policy evidence.
- **Citations and abstention:** Returns source identifiers such as `[S1]` and is designed to abstain when available evidence is insufficient.
- **LangGraph orchestration:** Implements graph-based workflows for retrieval, generation, tool selection, and conversation handling.
- **MCP integration:** Exposes banking policy retrieval through an MCP server and allows the agent to discover and invoke the available MCP tools.
- **Dual assistant workflows:** Provides a choice between the standard RAG workflow and the MCP-enabled agent through the web interface.
- **Conversational context:** Uses LangGraph checkpointing to maintain conversation history across multiple turns during the application process lifetime.
- **FastAPI backend:** Provides endpoints for health checks, standard RAG chat, and MCP-enabled chat.
- **Interactive frontend:** Includes a banking support chat interface, workflow selector, suggested questions, and an evidence panel.
- **Testing and CI:** Uses pytest and GitHub Actions to automate tests without requiring a live hosted-model API key.
- **Containerization:** Includes Docker and Docker Compose configuration for containerized execution and Chroma persistence.

## Architecture

The application supports two distinct conversational workflows.

### 1. Standard RAG workflow

```text
                         User
                           |
                           v
                         FastAPI
                           |
                       POST /chat
                           |
                           v
                BankingSupportService
                           |
                           v
                       LangGraph
                           |
                           v
                    PolicyRetriever
                           |
                 +---------+---------+
                 |                   |
                 v                   v
          Hugging Face           Chroma
           Embeddings          Vector Store
                 |                   |
                 +---------+---------+
                           |
                           v
                 GroundedAnswerGenerator
                           |
                           v
                OpenRouter Chat Model
                           |
                           v
                Answer + Sources + Status
```

### 2. MCP-enabled agent workflow

```text
                         User
                           |
                           v
                         FastAPI
                           |
                      POST /mcp/chat
                           |
                           v
                   MCP Support Graph
                        (LangGraph)
                           |
                           v
                  Tool Selection Model
                           |
                 +---------+---------+
                 |                   |
          Direct Response       Tool Invocation
                                     |
                                     v
                              MCP Adapter
                                     |
                              stdio Transport
                                     |
                                     v
                           Banking Knowledge
                               MCP Server
                                     |
                                     v
                             MCP Tool Call
                          search_banking_policies
                                     |
                                     v
                               PolicyRetriever
                                     |
                                     v
                               Chroma Store
                                     |
                                     v
                              Policy Evidence
                                     |
                                     v
                         GroundedAnswerGenerator
                                     |
                                     v
                           OpenRouter Chat Model
                                     |
                                     v
                         Answer + Sources + Status
```

The MCP server runs as a separate process and communicates with the application through the MCP adapter using standard input/output (stdio) transport.

The MCP tool exposes the existing policy retrieval capability rather than implementing a separate vector database or duplicating the ingestion pipeline.

The agent can also respond directly to general conversational questions when policy retrieval is unnecessary.

## Technology stack

| Area                | Technology                         |
| ------------------- | ---------------------------------- |
| Language            | Python 3.11                        |
| API                 | FastAPI, Uvicorn                   |
| LLM framework       | LangChain                          |
| Agent orchestration | LangGraph                          |
| Tool integration    | Model Context Protocol (MCP)       |
| MCP communication   | MCP SDK, stdio transport           |
| Vector database     | ChromaDB                           |
| Embeddings          | Hugging Face Sentence Transformers |
| Generation          | OpenRouter-compatible API          |
| Frontend            | HTML, CSS, JavaScript              |
| Testing             | pytest                             |
| Containers          | Docker, Docker Compose             |
| CI                  | GitHub Actions                     |

## Knowledge base

The initial synthetic policy collection includes:

- `card_fees.md`
- `refunds.md`

Documents are processed using `RecursiveCharacterTextSplitter` with:

- Chunk size: 700
- Chunk overlap: 100

Metadata connects retrieved passages to their original policy documents.

The same indexed policy collection is used by both the standard RAG workflow and the MCP-enabled workflow.

These example policies do not represent any real financial institution.

## Initial retrieval baseline

An initial retrieval evaluation used five question/evidence cases with `k=2`.

| Metric               | Initial result |
| -------------------- | -------------: |
| Mean Recall@2        |           1.00 |
| Mean Precision@2     |           0.60 |
| Mean Reciprocal Rank |           1.00 |

This is a small development baseline and should not be interpreted as evidence of production-grade retrieval performance.

## Getting started

### Prerequisites

- Python 3.11
- Conda or another Python environment manager
- An OpenRouter API key for live model generation
- Docker and Docker Compose for containerized execution

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

Create a local `.env` file using `.env.example`, if available.

Example configuration:

```dotenv
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=your_model_identifier
OPENROUTER_MAX_TOKENS=12000
OPENROUTER_TIMEOUT_SECONDS=60
CHROMA_PERSIST_DIRECTORY=data/vector_store/chroma
```

Check the current implementation and `.env.example` for the authoritative configuration.

Never commit `.env` or expose an actual API key in documentation, source code, or CI.

### 4. Build the policy index

Before starting the MCP-enabled application, ensure the Chroma vector store has been populated using the existing policy ingestion workflow.

The MCP server connects to the existing index; it does not automatically rebuild the policy collection during startup.

The configured Chroma directory defaults to:

```text
data/vector_store/chroma
```

If `CHROMA_PERSIST_DIRECTORY` is configured, both the application and MCP server must resolve to the same persistent directory.

### 5. Run tests

```bash
python -m pytest -q
```

The test suite is designed to exercise application components without requiring a live hosted-model API key where dependencies are mocked.

Live generation and complete end-to-end model tests require valid runtime model configuration.

### 6. Start the application

Run the FastAPI application from the project root:

```bash
uvicorn src.api:app --reload
```

The FastAPI lifespan initializes the application services and MCP adapter. The adapter launches the banking knowledge MCP server as a subprocess.

Open the frontend:

```text
http://127.0.0.1:8000/
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Using the application

The frontend provides two assistant modes:

### Standard RAG

Uses the existing retrieval and grounded generation pipeline.

The application retrieves relevant policy passages and passes them to the generation model to formulate an evidence-based response.

### MCP Agent

Uses the MCP-enabled LangGraph workflow.

The agent determines whether a question requires banking policy retrieval, invokes the exposed MCP tool when necessary, and passes retrieved evidence to the existing grounded answer generator.

The MCP workflow also supports multi-turn conversations using conversation identifiers and LangGraph checkpointing.

**Current persistence limitation:** The implementation uses `InMemorySaver`. Conversation history is maintained in application memory and is lost when the application process restarts. Persistent conversation storage is a planned improvement.

## API overview

| Method | Endpoint    | Description                    |
| ------ | ----------- | ------------------------------ |
| GET    | `/`         | Banking support web interface  |
| GET    | `/health`   | Application health check       |
| POST   | `/chat`     | Standard RAG conversation      |
| POST   | `/mcp/chat` | MCP-enabled agent conversation |

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

Returns an answer, source metadata, an abstention indicator, and an error field.

### `POST /mcp/chat`

Example request:

```json
{
  "question": "Are card fee refunds guaranteed?",
  "conversation_id": null
}
```

A conversation identifier can be supplied on subsequent requests to continue an existing conversation. When omitted, the application creates a new conversation identifier.

The MCP response includes the conversation identifier, answer, retrieval status, sources, abstention status, and error information.

Consult the current FastAPI response models for the authoritative field definitions.

## Continuous integration

The GitHub Actions workflow at `.github/workflows/ci.yml` runs on pushes and pull requests targeting `main` or `master`, as well as manual dispatch.

The workflow:

1. Sets up Python 3.11.
2. Installs project dependencies from `requirements.txt`.
3. Runs the automated test suite.

```bash
python -m pytest -q
```

The workflow does not require a live OpenRouter API key. Tests mock external model dependencies where appropriate.

## Evaluation and limitations

The initial generation evaluation contained eight test cases.

Recorded initial results:

| Metric                    | Initial result |
| ------------------------- | -------------: |
| Overall pass rate         |          12.5% |
| Abstention accuracy       |         85.71% |
| Fact-check pass rate      |         14.29% |
| Source-coverage pass rate |         85.71% |
| Citation-validity rate    |           100% |

The initial overall pass rate is low and is treated as an area for improvement rather than hidden.

Issues identified included:

- Overly strict phrase matching that failed to recognize some valid paraphrases.
- Intermittent empty model responses.
- An abstention expectation that required further review against the intended policy-grounding behavior.

These results are an early development snapshot, not a final evaluation of the MCP-enabled system.

### Current limitations

- Conversation checkpoints are stored in process memory rather than persistent storage.
- Generation evaluation requires further refinement and broader test coverage.
- The initial knowledge base is small and synthetic.
- The application does not yet implement production customer authentication or authorization.
- The application has not been deployed as a production banking service.
- The MCP workflow depends on the existing Chroma policy index being available at startup.

## Planned improvements

- Replace in-memory LangGraph checkpointing with persistent SQLite storage.
- Improve conversation resumption and customer-specific conversation isolation.
- Expand MCP integration and automated workflow testing.
- Improve contextual retrieval and query rewriting.
- Refine generation evaluation, abstention behavior, and citation verification.
- Introduce MLflow tracing and GenAI observability.
- Improve authentication, authorization, rate limiting, and secret management.
- Review deployment architecture, container size, and cloud infrastructure requirements.

## Security and responsible use

- All included banking policies are synthetic.
- This application is not a source of real financial advice.
- Never commit API keys, credentials, or `.env` files.
- Retrieved policy evidence should be inspected before acting on generated responses.
- Generated answers can be incomplete or incorrect.
- Authentication, authorization, rate limiting, secure secret management, and appropriate data protection must be implemented before public deployment.

## Documentation

See [`README_DEVELOPMENT.md`](README_DEVELOPMENT.md) for the detailed engineering journal, including:

- Project history and development milestones.
- Architecture and implementation decisions.
- Ingestion, retrieval, and generation workflows.
- MCP server and client integration.
- Testing and troubleshooting notes.
- Evaluation snapshots.
- CI and containerization details.
- Future development priorities.

## License

Add the license appropriate for this repository before distributing or reusing the project.
