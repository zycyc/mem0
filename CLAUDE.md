# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Python Development
- **Install dependencies**: `pip install -e .` or `hatch env create`
- **Install all extras**: `make install_all`
- **Run tests**: `hatch run test` or `pytest tests/`
- **Run specific test**: `pytest tests/path/to/test.py::TestClass::test_method`
- **Format code**: `hatch run format` or `ruff format`
- **Lint code**: `hatch run lint` or `ruff check`
- **Fix linting issues**: `hatch run lint-fix` or `ruff check --fix`
- **Sort imports**: `hatch run isort mem0/`

### TypeScript/JavaScript Development (mem0-ts)
- **Install dependencies**: `pnpm install` (in mem0-ts directory)
- **Build package**: `npm run build`
- **Run tests**: `npm test`
- **Format code**: `npm run format`
- **Check formatting**: `npm run format:check`

### Documentation
- **Run docs locally**: `cd docs && mintlify dev`

### Building and Publishing
- **Build Python package**: `hatch build`
- **Build TypeScript package**: `npm run build` (in mem0-ts)

## Architecture

### Core Components

**mem0/** - Main Python package
- `memory/main.py` - Core Memory and AsyncMemory classes for managing long-term memory
- `client/main.py` - MemoryClient and AsyncMemoryClient for API interactions
- `llms/` - LLM provider integrations (OpenAI, Anthropic, Ollama, etc.)
- `embeddings/` - Embedding model integrations
- `vector_stores/` - Vector database integrations (Qdrant, Chroma, Pinecone, etc.)
- `graphs/` - Graph database support for relationship mapping

**mem0-ts/** - TypeScript/JavaScript SDK
- Provides Node.js and browser support for Mem0
- Mirrors Python API functionality

**openmemory/** - Self-hosted solution
- `api/` - FastAPI backend server
- `ui/` - Next.js frontend interface
- Docker Compose configuration for vector stores

### Key Patterns

1. **Factory Pattern**: Used extensively for creating LLMs, embedders, and vector stores
   - See `mem0/utils/factory.py`
   
2. **Async Support**: Both sync and async versions of main classes
   - Memory/AsyncMemory, MemoryClient/AsyncMemoryClient

3. **Configuration**: Pydantic models for configuration validation
   - Base configs in `mem0/configs/base.py`
   - Provider-specific configs in respective subdirectories

4. **Memory Operations**: Core operations include add, search, update, delete
   - Memories are stored with metadata (user_id, agent_id, run_id)
   - Support for multimodal content (text and images)

## Testing Approach
- Unit tests in `tests/` directory mirror source structure
- Use pytest-mock for mocking external services
- Run tests with `pytest tests/` or `hatch run test`
- Test multiple Python versions with `hatch run dev_py_3_X:test`

## Development Notes
- Ruff is configured with line length 120 and excludes embedchain/ and openmemory/ directories
- Python 3.9-3.12 supported
- Use hatch for environment management
- TypeScript uses pnpm as package manager