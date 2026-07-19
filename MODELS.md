# LLM & Embedding Models Guide

## Current Configuration

✅ **LLM Model**: `llama3.2:3b` (Ollama - Free, Local, No API costs)  
✅ **Embedding Model**: `nomic-embed-text` (Ollama - 768-dimensional output)  
✅ **Base URL**: `http://localhost:11434` (Ollama local server)

---

## Why Ollama? (Default Choice)

### Advantages:
- ✅ **100% Free** - No API costs, unlimited queries
- ✅ **Privacy** - Data stays on your machine
- ✅ **No Rate Limits** - Query as much as you want
- ✅ **Offline Capable** - Works without internet
- ✅ **Fast** - Local inference, no network latency
- ✅ **Open Source** - Full transparency

### System Requirements:
- **RAM**: 8GB minimum (16GB recommended for larger models)
- **Storage**: ~4GB per model
- **OS**: macOS, Linux, Windows

---

## Ollama Models for RAG

### LLM Models (Text Generation)

| Model | Size | RAM | Best For | Performance |
|-------|------|-----|----------|-------------|
| `llama3.2:3b` | 2GB | 8GB | **Recommended** - Fast, accurate, great for RAG | ⭐⭐⭐⭐⭐ |
| `llama3.2:1b` | 1.3GB | 4GB | Lightweight, very fast | ⭐⭐⭐ |
| `llama3.1:8b` | 4.7GB | 16GB | Higher quality responses | ⭐⭐⭐⭐ |
| `mistral:7b` | 4.1GB | 16GB | Alternative to Llama, good reasoning | ⭐⭐⭐⭐ |
| `phi3:mini` | 2.2GB | 8GB | Microsoft model, efficient | ⭐⭐⭐⭐ |

### Embedding Models

| Model | Dimensions | Size | Best For |
|-------|------------|------|----------|
| `nomic-embed-text` | 768 | 274MB | **Recommended** - Fast, accurate |
| `mxbai-embed-large` | 1024 | 670MB | Higher quality embeddings |
| `all-minilm` | 384 | 45MB | Lightweight, fast |

---

## Installation & Usage

### 1. Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai)

### 2. Pull Models

```bash
# LLM for text generation
ollama pull llama3.2:3b

# Embedding model for vector search
ollama pull nomic-embed-text
```

### 3. Verify Installation

```bash
# List installed models
ollama list

# Test the LLM
ollama run llama3.2:3b "Hello!"

# Check if Ollama is running
curl http://localhost:11434/api/tags
```

---

## Alternative Cloud LLMs (Optional)

The system supports multiple providers via LangChain:

### Google Gemini (Cloud)

**Pros:**
- High quality responses
- Fast API
- Free tier available

**Cons:**
- Requires API key
- Rate limits on free tier
- Data sent to Google

**Models:**

| Model | Context | Best For | Cost |
|-------|---------|----------|------|
| `gemini-1.5-flash` | 1M tokens | Fast, efficient | Free tier |
| `gemini-1.5-pro` | 2M tokens | Complex reasoning | Paid |

**Setup:**
1. Get API key: https://aistudio.google.com/apikey
2. Add to `.env`: `GOOGLE_API_KEY=your-key`
3. Update `config/config.yml`:
   ```yaml
   llm:
     model: "gemini-1.5-flash"
   ```

### OpenAI (Cloud)

**Pros:**
- Industry-leading quality
- Well-documented
- Reliable

**Cons:**
- Paid only (no free tier)
- Expensive for high volume
- Rate limits

**Models:**

| Model | Context | Best For | Cost |
|-------|---------|----------|------|
| `gpt-4o-mini` | 128K | Cost-effective | $0.15/1M tokens |
| `gpt-4o` | 128K | Best quality | $2.50/1M tokens |
| `gpt-3.5-turbo` | 16K | Legacy, cheap | $0.50/1M tokens |

**Setup:**
1. Get API key: https://platform.openai.com/api-keys
2. Add to `.env`: `OPENAI_API_KEY=your-key`
3. Update `config/config.yml`:
   ```yaml
   llm:
     model: "gpt-4o-mini"
   ```

---

## Switching Between Providers

### Method 1: Configuration File (Permanent)

Edit `config/config.yml`:

**For Ollama (default):**
```yaml
llm:
  model: "llama3.2:3b"
  base_url: "http://localhost:11434"

embeddings:
  default_model: "nomic-embed-text"
```

**For Google Gemini:**
```yaml
llm:
  model: "gemini-1.5-flash"
  # Remove or comment out base_url

embeddings:
  default_model: "gemini-embedding-001"  # Not implemented yet
```

**For OpenAI:**
```yaml
llm:
  model: "gpt-4o-mini"
  # Remove or comment out base_url

embeddings:
  default_model: "text-embedding-3-small"  # Not implemented yet
```

### Method 2: Environment Variables (Temporary)

```bash
# No changes needed for Ollama (default)

# For Google Gemini
export GOOGLE_API_KEY=your-key

# For OpenAI
export OPENAI_API_KEY=your-key
```

Then restart the backend:
```bash
pkill -f "python.*api.py"
cd backend && python api.py
```

---

## Performance Comparison

| Provider | Speed | Cost | Privacy | Quality | Offline |
|----------|-------|------|---------|---------|---------|
| **Ollama** | ⭐⭐⭐⭐ | Free | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| Gemini | ⭐⭐⭐⭐⭐ | Free tier | ⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |
| OpenAI | ⭐⭐⭐⭐ | Paid | ⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |

---

## Common Issues

### Ollama: "connection refused"
```bash
# Check if Ollama is running
ollama serve

# Or restart Ollama
pkill ollama
ollama serve
```

### Ollama: "model not found"
```bash
# Pull the model
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Verify
ollama list
```

### Gemini: "API key error"
```bash
# Check if key is set
echo $GOOGLE_API_KEY

# Set in .env file
echo "GOOGLE_API_KEY=your-key" >> .env
```

### OpenAI: "Rate limit exceeded"
- Upgrade to paid tier
- Reduce request frequency
- Use smaller batch sizes

---

## Recommendations

### For Development:
**Use Ollama** - Free, fast iteration, no API costs

### For Production (Low Budget):
**Use Ollama** - No ongoing costs, privacy-preserving

### For Production (High Quality):
**Use Google Gemini** - Best quality/cost ratio

### For Enterprise:
**Use Ollama** - Data privacy, no external dependencies

---

## Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama Model Library](https://ollama.ai/library)
- [LangChain Ollama Docs](https://python.langchain.com/docs/integrations/llms/ollama)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)
- [OpenAI API Docs](https://platform.openai.com/docs)

---

**Questions?** Open an issue or check the [README](README.md) for more info.
