# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install dependencies
pdm install -G dev,test,lint

# Install for GPU support
pdm install -G gpu
```

### Build & Test
```bash
# Run tests with coverage
pdm run test

# Run specific test
pytest tests/test_specific.py -v

# Build package
pdm build
```

### Code Quality
```bash
# Run linting and auto-fix
pdm run ruff check src/ tests/ --fix
```

### Documentation
```bash
# Serve docs locally
pdm run docs

# Build docs
pdm run docs-build
```

### Applications
```bash
# Start web interface
pdm run web

# Start API server
pdm run api
```

## Architecture Overview

Milvus Dataset is a Python library for managing large-scale datasets, optimized for vector databases but useful as a standalone tool.

### Core Components

1. **ConfigManager** (`src/milvus_dataset/core.py`): Singleton for global storage configuration
2. **Dataset/DatasetDict** (`src/milvus_dataset/core.py`): Main dataset management classes
3. **Storage Layer** (`src/milvus_dataset/storage.py`): Abstraction for local/S3/GCS storage backends
4. **Writer** (`src/milvus_dataset/writer.py`): Context-managed data writing with buffering
5. **Reader** (`src/milvus_dataset/reader.py`): Efficient batch/streaming data reading
6. **Neighbors** (`src/milvus_dataset/neighbors.py`): k-NN computation with optional GPU support

### Data Flow

1. Initialize storage via `ConfigManager.init_storage()`
2. Load/create dataset with schema using `load_dataset()`
3. Write data using context managers: `with dataset["train"].get_writer() as writer:`
4. Read data: `dataset["train"].read()` or iterate in batches
5. Export to Milvus or Hugging Face Hub

### Key Design Patterns

- **Storage Agnostic**: Same API works across local, S3, and GCS storage
- **Context Managers**: Safe resource handling for writers
- **Parquet Format**: Efficient columnar storage with PyArrow
- **Schema Validation**: Pydantic models for configuration
- **Buffer Management**: Configurable write buffers for performance

### File Structure

- `src/milvus_dataset/`: Core library code
  - `core.py`: Main dataset classes and configuration
  - `storage.py`: Storage backend implementations
  - `writer.py`/`reader.py`: Data I/O
  - `api/`: FastAPI endpoints
  - `web/`: Streamlit interface
- `tests/`: Test suite
- `example/`: Usage examples
- `docs/`: MkDocs documentation