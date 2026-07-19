# Google Gemini API - Available Models Reference

## Current Configuration
✅ **LLM Model**: `gemini-1.5-flash`  
✅ **Embedding Model**: `models/gemini-embedding-001` (with 768-dimensional output)

## Recommended Models for RAG Applications

### LLM Models (for text generation/Q&A)

| Model Name | Description | Best For | Status |
|------------|-------------|----------|--------|
| `gemini-1.5-flash` | **Recommended** - Fast, efficient, cost-effective | Most RAG applications, high-volume queries | ✅ Stable |
| `gemini-1.5-pro` | Higher quality, more capable | Complex reasoning, longer contexts | ✅ Stable |
| `gemini-2.5-flash` | Newer generation (if available) | Latest features | ⚠️ Check availability |
| `gemini-1.5-flash-8b` | Smaller, faster variant | High-throughput applications | ✅ Stable |

### Embedding Models

| Model Name | Dimensions | Description |
|------------|------------|-------------|
| `models/gemini-embedding-001` | 3072 (768 with Matryoshka) | **Current** - Supports truncation to 768 dims |
| `text-embedding-004` | 768 | Alternative stable embedding model |

## Why We Use gemini-1.5-flash

1. **Stable and Reliable**: Production-ready, not experimental
2. **Fast Response Time**: Optimized for speed
3. **Cost-Effective**: Good balance of quality and cost
4. **Well-Documented**: Extensive documentation and support
5. **1M Token Context**: Supports long documents (though we chunk them)

## Configuration File Location
`config/config.yml` → `llm.model`

## How to Change the Model

1. Open `config/config.yml`
2. Update the `llm.model` field:
   ```yaml
   llm:
     model: "gemini-1.5-flash"  # Change this line
   ```
3. Restart the backend server:
   ```bash
   pkill -f "python.*api.py"
   ./start_backend.sh
   ```

## Common Issues

### ❌ "Model not found" Error
**Cause**: Model name is incorrect or model is deprecated  
**Solution**: Use `gemini-1.5-flash` or `gemini-1.5-pro`

### ❌ "API version not supported" Error
**Cause**: Using experimental model names (e.g., `-exp` suffix)  
**Solution**: Remove `-exp` suffix and use stable model names

### ❌ Rate Limit Errors
**Cause**: Too many requests to the API  
**Solution**: 
- The code already includes retry logic with exponential backoff
- Consider upgrading to paid tier for higher limits
- Reduce batch size in document processing

## Testing the Model

You can test if the model works by querying the health endpoint:
```bash
curl http://localhost:8000/api/config
```

Or by uploading a test document through the web interface.

## Additional Resources

- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Available Models List](https://ai.google.dev/gemini-api/docs/models)
- [Model Pricing](https://ai.google.dev/pricing)
- [Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits)

## Notes

- **Free Tier**: Gemini 1.5 Flash is available for free with rate limits
- **API Key**: Required (set in `.env` as `GOOGLE_API_KEY` or `GEMINI_API_KEY`)
- **Model Updates**: Google regularly updates models; check documentation for latest versions
- **Experimental Models**: Avoid using `-exp` or `-preview` suffixed models in production
