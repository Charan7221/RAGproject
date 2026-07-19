# Contributing to RAG Document QA System

Thank you for considering contributing to this project! 🎉

## How to Contribute

### Reporting Bugs 🐛

If you find a bug, please open an issue with:
- **Clear description** of the problem
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Your environment** (OS, Python version, Node version, etc.)
- **Screenshots** if applicable

### Suggesting Features 💡

Feature requests are welcome! Please:
- Check if the feature already exists or is planned
- Explain the **use case** and why it's valuable
- Describe the **expected behavior**
- Consider **implementation complexity**

### Pull Requests 🔀

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/RAGproject.git
   cd RAGproject
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
4. **Make your changes** following our coding standards
5. **Test thoroughly**
6. **Commit** with clear messages:
   ```bash
   git commit -m "feat: add amazing feature"
   ```
7. **Push** to your branch:
   ```bash
   git push origin feature/amazing-feature
   ```
8. **Open a Pull Request** with a clear description

## Development Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/Charan7221/RAGproject.git
cd RAGproject

# Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up frontend
cd frontend
npm install
cd ..

# Configure environment
cp .env.example .env
# Edit .env and add your API keys

# Start PostgreSQL
docker compose up -d

# Run tests
pytest tests/ -v
```

## Coding Standards

### Python Code Style

- Follow **PEP 8** style guide
- Use **type hints** where appropriate
- Add **docstrings** to all functions and classes
- Keep functions **small and focused** (< 50 lines ideally)
- Use **meaningful variable names**
- Avoid deep nesting (max 3-4 levels)

**Example:**
```python
def process_document(file_path: str, chunk_size: int = 512) -> List[Document]:
    """
    Process a document and split it into chunks.
    
    Args:
        file_path: Path to the document file
        chunk_size: Size of each text chunk (default: 512)
        
    Returns:
        List of Document objects with chunked text
    """
    # Implementation
    pass
```

### JavaScript/React Code Style

- Use **functional components** with hooks
- Follow **React best practices**
- Use **meaningful component and variable names**
- Keep components **small and reusable**
- Use **PropTypes** or TypeScript for type checking
- Avoid inline styles (use CSS modules or styled-components)

**Example:**
```javascript
const DocumentUpload = ({ onUpload, sessionId }) => {
  const [uploading, setUploading] = useState(false);
  
  // Component implementation
};
```

### Git Commit Messages

Follow **conventional commits** format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style/formatting changes
- `refactor:` Code refactoring without feature changes
- `test:` Adding or updating tests
- `chore:` Maintenance tasks, dependency updates

**Examples:**
```
feat: add support for PPTX document upload
fix: resolve session persistence issue on page reload
docs: update README with deployment instructions
refactor: simplify retriever query logic
test: add unit tests for document processor
```

## Testing Guidelines

### Writing Tests

- Write tests for **all new features**
- Ensure **existing tests pass**
- Aim for **good test coverage** (>70%)
- Test **edge cases** and error conditions
- Use **descriptive test names**

### Running Tests

```bash
# Run all Python tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pg_retriever.py -v

# Run frontend tests (if available)
cd frontend && npm test
```

## Project Structure

Understanding the codebase:

```
RAGproject/
├── backend/              # FastAPI backend
│   ├── api.py           # Main API endpoints
│   ├── middleware.py    # CORS, error handling
│   └── logging_config.py
├── frontend/            # React frontend
│   └── src/
│       ├── components/  # React components
│       └── services/    # API client
├── src/rag_qa/          # Core RAG library
│   ├── core/            # Core RAG components
│   │   ├── database.py
│   │   ├── document_processor.py
│   │   ├── pg_retriever.py
│   │   ├── query_expander.py
│   │   ├── rag_openai.py
│   │   ├── reranker.py
│   │   └── text_splitter.py
│   └── utils/           # Utilities
│       ├── config_loader.py
│       └── helpers.py
├── config/              # Configuration files
├── tests/               # Test files
└── requirements.txt     # Python dependencies
```

## Areas for Contribution

### High Priority
- [ ] Add more unit and integration tests
- [ ] Improve error handling and user feedback
- [ ] Add support for more document formats
- [ ] Optimize query performance
- [ ] Add user authentication

### Medium Priority
- [ ] Improve UI/UX design
- [ ] Add conversation memory
- [ ] Implement document caching
- [ ] Add monitoring and analytics
- [ ] Create Docker production setup

### Documentation
- [ ] Add API documentation
- [ ] Create video tutorials
- [ ] Write deployment guides
- [ ] Add architecture diagrams
- [ ] Document configuration options

## Questions?

- Open an **issue** for questions about the codebase
- Check existing **issues** and **pull requests** first
- Reach out to [@Charan7221](https://github.com/Charan7221)

## Code of Conduct

### Our Standards

- **Be respectful** and inclusive
- **Welcome newcomers** and help them learn
- **Provide constructive feedback**
- **Focus on the issue**, not the person
- **Help create a positive environment**

### Unacceptable Behavior

- Harassment, trolling, or insulting comments
- Personal attacks or political arguments
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes
- Credited in commit history

Thank you for contributing! 🙏

---

**Happy Coding!** 💻
