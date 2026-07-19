# RAG Document QA System

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/🦜_LangChain-0.3+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.120+-009688.svg)
![React](https://img.shields.io/badge/React-18-61DAFB.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-336791.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A session-based document Q&A application built with **LangChain + FastAPI + React + PostgreSQL/pgvector + Google Gemini**. Upload PDF, DOCX, TXT, or Markdown files and ask questions — answers are grounded in your documents with source attribution and streaming responses.

Built on the **LangChain framework** for modular RAG orchestration, document processing, and multi-provider LLM/embedding integration.

## 📸 Demo

> **Note**: Add screenshots or a demo GIF of your application here

<details>
<summary>📷 Click to see example UI (add your own screenshots)</summary>

```
[Screenshot 1: Document Upload Interface]
[Screenshot 2: Chat Interface with Streaming Response]
[Screenshot 3: Source Attribution Display]
```

</details>

## Features

- **LangChain-powered RAG pipeline** — modular document processing, embeddings, and LLM orchestration
- **Dynamic document upload** — PDF, DOCX, TXT, MD with LangChain document loaders
- **Hybrid retrieval** — pgvector semantic search + PostgreSQL full-text, fused with Reciprocal Rank Fusion (RRF)
- **Multi-provider support** — Google Gemini, OpenAI, or Ollama (via LangChain integrations)
- **Streaming answers** — token-by-token SSE responses in the chat UI
- **Session persistence** — sessions and chat history survive page reloads
- **Source attribution** — see which document chunks were used
- **Multi-session isolation** — each session has its own document scope

## Architecture

```
React Frontend (:3000)
        │  REST + SSE
        ▼
FastAPI Backend (:8000)
        │  LangChain RAG Pipeline
        │
        ├── LangChain Document Loaders & Text Splitters
        ├── LangChain Embeddings (Gemini/OpenAI/Ollama)
        ├── Custom LangChain Retriever (Hybrid Search)
        ├── LangChain Chat Models (Gemini LLM)
        └── PostgreSQL + pgvector (:5432)
              ├── sessions
              ├── documents
              ├── document_chunks (vectors)
              └── chat_history
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker (for PostgreSQL + pgvector)
- Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### 1. Clone and configure

```bash
cd RAGproject-main
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your-key
```

### 2. Start PostgreSQL

```bash
docker compose up -d
```

### 3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd frontend && npm install && cd ..
```

### 4. Run the app

**Option A — startup script:**

```bash
chmod +x start_fullstack.sh stop_fullstack.sh
./start_fullstack.sh
```

**Option B — manual:**

```bash
# Terminal 1 — backend
cd backend && python api.py

# Terminal 2 — frontend
cd frontend && npm start
```

Open **http://localhost:3000**

API docs: **http://localhost:8000/docs**

## Configuration

Settings live in `config/config.yml`. Environment variables override database credentials:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | Gemini API key (required) |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `rag_db` | Database name |
| `DB_USER` | `rag_user` | Database user |
| `DB_PASSWORD` | `rag_password` | Database password |

Key settings in `config/config.yml`:

```yaml
llm:
  model: "gemini-1.5-flash"  # Stable production model

retrieval:
  top_k: 5
  hybrid_search:
    enabled: true
    use_rrf: true

text_splitter:
  chunk_size: 128
  chunk_overlap: 15
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/api/config` | Public config for frontend |
| POST | `/api/session/create` | Create session |
| POST | `/api/documents/upload` | Upload document |
| POST | `/api/query` | Ask question |
| POST | `/api/query/stream` | Stream answer (SSE) |
| GET | `/api/session/{id}` | Session info |
| GET | `/api/session/{id}/history` | Chat history |
| DELETE | `/api/session/{id}` | Delete session |
| DELETE | `/api/session/{id}/documents/{doc_id}` | Delete document |

## LangChain Integration

This project leverages **LangChain** as the core RAG framework:

### LangChain Components Used

- **langchain-core** — Base abstractions (`Document`, `BaseRetriever`, `CallbackManager`, message types)
- **langchain-google-genai** — Google Gemini LLM and embeddings integration
- **langchain-openai** — OpenAI embeddings and chat models (alternative)
- **langchain-ollama** — Local Ollama embeddings (alternative)
- **langchain-text-splitters** — `RecursiveCharacterTextSplitter` for intelligent chunking
- **langchain-postgres** — PostgreSQL + pgvector integration

### Custom Implementations

- **PgHybridRetriever** (extends `BaseRetriever`) — Custom hybrid search with semantic + full-text retrieval
- **Document processing pipeline** — Uses LangChain `Document` objects throughout
- **Multi-provider embeddings** — Abstracted via LangChain's embedding interfaces

### Why LangChain?

- ✅ **Modularity** — Easy to swap LLM/embedding providers (Gemini ↔ OpenAI ↔ Ollama)
- ✅ **Standardization** — Consistent interfaces across components
- ✅ **Extensibility** — Custom retrievers integrate seamlessly with LangChain ecosystem
- ✅ **Production-ready** — Built-in callbacks, error handling, and streaming support

## Project Structure

```
RAGproject-main/
├── backend/           # FastAPI API server
├── frontend/          # React SPA
├── src/rag_qa/        # Core RAG library
│   ├── core/          # Retriever, LLM, document processing
│   └── utils/         # Config loader
├── config/config.yml  # Central configuration
├── tests/             # Unit tests
├── docker-compose.yml # PostgreSQL + pgvector
└── requirements.txt
```

## Testing

```bash
source venv/bin/activate
pytest tests/ -v
```

## Troubleshooting

**Database connection failed:**
```bash
docker compose up -d
docker compose exec postgres pg_isready -U rag_user -d rag_db
```

**Missing API key:**
```bash
export GOOGLE_API_KEY=your-key
# or add to .env file
```

**Port in use:**
```bash
lsof -ti:8000 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **RAG Framework** | **LangChain 0.3+** (core, community, google-genai, openai, ollama, postgres) |
| Backend | FastAPI, Uvicorn, SSE-Starlette |
| Frontend | **React 18** (functional components + hooks), Axios, react-dropzone, react-markdown |
| Vector DB | PostgreSQL 17 + pgvector (HNSW index) |
| Embeddings | Google Gemini `gemini-embedding-001` (768-dim via Matryoshka) |
| LLM | Google Gemini `gemini-1.5-flash` (supports OpenAI/Ollama alternatives) |
| Text Processing | LangChain RecursiveCharacterTextSplitter, tiktoken encoding |
| Retrieval | Hybrid semantic + full-text with RRF (custom LangChain BaseRetriever) |
| Document Loaders | PyPDF2, python-docx, BeautifulSoup4 (via LangChain Documents) |

> **Note**: GitHub shows the frontend as "JavaScript" because React is JavaScript. The frontend is built with React 18 using modern functional components and hooks.

## License

MIT - See [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Roadmap

Future improvements planned:

- [ ] Add support for more document formats (PPTX, CSV, JSON)
- [ ] Implement conversation memory across sessions
- [ ] Add user authentication and multi-user support
- [ ] Support for image/table extraction from documents
- [ ] Add reranking with cross-encoder models
- [ ] Multi-language support for UI
- [ ] Citation highlighting in source documents
- [ ] Performance monitoring dashboard
- [ ] Export chat history (PDF/TXT)
- [ ] Batch document upload

## Acknowledgments

Built with these amazing technologies:
- [LangChain](https://www.langchain.com/) - RAG framework and LLM orchestration
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://react.dev/) - Frontend UI library
- [PostgreSQL](https://www.postgresql.org/) + [pgvector](https://github.com/pgvector/pgvector) - Vector database
- [Google Gemini](https://ai.google.dev/) - LLM and embeddings API

## Contact & Support

- **Author**: Charan7221
- **GitHub**: [@Charan7221](https://github.com/Charan7221)
- **Project Link**: [https://github.com/Charan7221/RAGproject](https://github.com/Charan7221/RAGproject)

## Star History

⭐ **If you find this project helpful, please give it a star!** ⭐

It helps others discover the project and motivates continued development.

---

<div align="center">
Made with ❤️ using LangChain and FastAPI
</div>
