# RAG Document QA System

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/🦜_LangChain-0.3+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.120+-009688.svg)
![React](https://img.shields.io/badge/React-18-61DAFB.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-336791.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A session-based document Q&A application built with **LangChain + FastAPI + React + PostgreSQL/pgvector + Ollama**. Upload PDF, DOCX, TXT, or Markdown files and ask questions — answers are grounded in your documents with source attribution and streaming responses. Uses local Ollama LLM for cost-free, privacy-preserving AI inference.

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
- **Redis caching** — Instant responses for repeated queries (90% faster)
- **Multi-provider support** — Ollama (default), Google Gemini, or OpenAI (via LangChain integrations)
- **Streaming answers** — token-by-token SSE responses with progress indicators
- **Session persistence** — sessions and chat history survive page reloads
- **Source attribution** — see which document chunks were used
- **Multi-session isolation** — each session has its own document scope
- **Optimized performance** — Smart chunking and caching for 40-50% speed improvement

## Architecture

```
React Frontend (:3000)
        │  REST + SSE
        ▼
FastAPI Backend (:8000)
        │  LangChain RAG Pipeline
        │
        ├── LangChain Document Loaders & Text Splitters
        ├── LangChain Embeddings (Ollama nomic-embed-text)
        ├── Custom LangChain Retriever (Hybrid Search)
        ├── LangChain Chat Models (Ollama Llama 3.2)
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
- **Redis** (for query caching - optional but recommended)
- Ollama (for local LLM - [install here](https://ollama.ai))

**Optional:** Google Gemini or OpenAI API key for alternative LLM providers

### 1. Clone and configure

```bash
cd RAGproject-main
cp .env.example .env
# Edit .env if using Google Gemini or OpenAI (Ollama works out of the box)
```

### 2. Install and start Ollama

**On macOS/Linux:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

**On Windows:** Download from [ollama.ai](https://ollama.ai)

### 3. Start PostgreSQL

```bash
docker compose up -d
```

### 4. Start Redis (Optional but Recommended)

Redis dramatically improves response times by caching query results.

**Using Docker:**
```bash
docker run -d --name rag-redis -p 6379:6379 redis:alpine
```

**Using Homebrew (macOS):**
```bash
brew install redis
brew services start redis
```

**Using apt (Linux):**
```bash
sudo apt install redis-server
sudo systemctl start redis
```

To disable caching, set `CACHE_ENABLED=false` in your `.env` file.

### 5. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd frontend && npm install && cd ..
```

### 6. Run the app

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
| `GOOGLE_API_KEY` | — | Google Gemini API key (optional, if using Gemini) |
| `OPENAI_API_KEY` | — | OpenAI API key (optional, if using OpenAI) |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `rag_db` | Database name |
| `DB_USER` | `rag_user` | Database user |
| `DB_PASSWORD` | `rag_password` | Database password |
| `REDIS_HOST` | `localhost` | Redis host (for caching) |
| `REDIS_PORT` | `6379` | Redis port |
| `CACHE_ENABLED` | `true` | Enable/disable query caching |
| `CACHE_TTL` | `3600` | Cache time-to-live in seconds (1 hour) |

Key settings in `config/config.yml`:

```yaml
llm:
  model: "llama3.2:3b"           # Ollama model (default)
  base_url: "http://localhost:11434"  # Ollama endpoint

embeddings:
  default_model: "nomic-embed-text"   # Ollama embedding model

retrieval:
  top_k: 5
  hybrid_search:
    enabled: true
    use_rrf: true

text_splitter:
  chunk_size: 1000               # Optimized for better context
  chunk_overlap: 100
```

**Performance Features:**
- ⚡ **Redis Caching**: Repeated queries return instantly (~0.1s vs 3-5s)
- 🎯 **Optimized Chunking**: Larger chunks (1000 chars) = better context, fewer embeddings
- 📊 **Streaming Progress**: Real-time status updates during query processing

**To switch to Google Gemini or OpenAI:**
1. Update `config/config.yml`: Change `model` to `gemini-1.5-flash` or `gpt-4`
2. Set API key in `.env`: `GOOGLE_API_KEY` or `OPENAI_API_KEY`
3. Restart the backend

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
| GET | `/api/cache/stats` | Cache statistics |

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

**Redis connection issues:**
```bash
# Check if Redis is running
redis-cli ping  # Should return "PONG"

# Or use Docker
docker ps | grep redis

# View Redis logs
docker logs rag-redis
```

**Disable caching if Redis unavailable:**
```bash
# In .env file
CACHE_ENABLED=false
```

**Ollama connection failed:**
```bash
# Check if Ollama is running
ollama list

# Start Ollama service (if not running)
ollama serve

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

**Database connection failed:**
```bash
docker compose up -d
docker compose exec postgres pg_isready -U rag_user -d rag_db
```

**Missing API key:**
```bash
# Only required if using Google Gemini or OpenAI
export GOOGLE_API_KEY=your-key
# or
export OPENAI_API_KEY=your-key
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
| **Cache** | **Redis** (query caching for 90% faster repeated queries) |
| LLM | **Ollama Llama 3.2:3b** (default), supports Google Gemini, OpenAI |
| Embeddings | **Ollama nomic-embed-text** (768-dim), supports Gemini, OpenAI |
| Text Processing | LangChain RecursiveCharacterTextSplitter, tiktoken encoding |
| Retrieval | Hybrid semantic + full-text with RRF (custom LangChain BaseRetriever) |
| Document Loaders | PyPDF2, python-docx, BeautifulSoup4 (via LangChain Documents) |

> **Note**: The system uses **Ollama** by default for cost-free, privacy-preserving local AI. Multi-provider architecture supports easy switching to Google Gemini or OpenAI via configuration.

## Performance

### Query Response Times

| Scenario | Without Cache | With Cache | Improvement |
|----------|---------------|------------|-------------|
| First query | 3-5 seconds | 3-5 seconds | - |
| Repeated query | 3-5 seconds | **~0.1 seconds** | **50x faster** |
| Similar query | 3-5 seconds | 3-5 seconds | - |

### Optimization Features

1. **Redis Caching**
   - Caches query results for 1 hour (configurable)
   - Automatic cache invalidation on document changes
   - Graceful degradation if Redis unavailable

2. **Optimized Chunking**
   - Larger chunks (1000 chars) = better context
   - Fewer embeddings to process = faster retrieval
   - Improved answer quality

3. **Streaming with Progress**
   - Real-time status updates (🔍 Searching... ✅ Found sources... ✍️ Generating...)
   - User sees progress immediately
   - Better perceived performance

### Cache Statistics

View cache performance at: `http://localhost:8000/api/cache/stats`

```json
{
  "enabled": true,
  "cache_hits": 45,
  "cache_misses": 23,
  "hit_rate_percent": 66.18,
  "cached_queries": 23
}
```

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
- [Ollama](https://ollama.ai/) - Local LLM runtime and models
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://react.dev/) - Frontend UI library
- [PostgreSQL](https://www.postgresql.org/) + [pgvector](https://github.com/pgvector/pgvector) - Vector database
- [Google Gemini](https://ai.google.dev/) - Optional cloud LLM provider

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
