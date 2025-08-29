# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install dependencies
pdm install -G dev,test,lint

# Install for GPU support (optional, for CUVS acceleration)
pdm install -G gpu

# Install for API server
pdm install -G api

# Install for web interface
pdm install -G web
```

### Build & Test
```bash
# Run full test suite with coverage
pdm run test

# Run specific test file
pytest tests/testcases/test_e2e.py -v

# Run with specific markers or filters
pytest tests/ -k "test_name" -v

# Build distribution package
pdm build
```

### Code Quality
```bash
# Run linting and auto-fix issues
pdm run ruff check src/ tests/ --fix

# Just check without fixing
ruff check src/ tests/

# Format code
ruff format src/ tests/
```

### Documentation
```bash
# Serve documentation locally (http://localhost:8000)
pdm run docs

# Build static documentation
pdm run docs-build
```

### Applications
```bash
# Start Streamlit web interface (port 8501)
pdm run web

# Start FastAPI server (port 8000)
pdm run api
```

## Architecture Overview

Milvus Dataset is a Python library for managing large-scale datasets, optimized for vector databases but also useful as a standalone dataset management tool. The library emphasizes storage-agnostic operations and efficient handling of vector data.

### Core Components

1. **ConfigManager** (`src/milvus_dataset/core.py:48`): Thread-safe singleton that manages global storage configuration
2. **Dataset/DatasetDict** (`src/milvus_dataset/core.py`): Primary dataset management classes supporting train/test splits
3. **Storage Layer** (`src/milvus_dataset/storage.py`): Storage abstraction supporting local, S3, and GCS backends via fsspec
4. **Writer** (`src/milvus_dataset/writer.py:29`): Context-managed data writing with buffering, queue management, and automatic file size control
5. **Reader** (`src/milvus_dataset/reader.py:23`): Efficient data reading supporting batch and streaming modes
6. **Neighbors** (`src/milvus_dataset/neighbors.py`): k-NN computation with CPU/GPU acceleration support

### Data Flow & Usage Patterns

1. **Initialize Storage**: `ConfigManager().init_storage()` configures backend (local/S3/GCS)
2. **Load Dataset**: `load_dataset()` creates or loads existing datasets with Milvus-compatible schemas
3. **Write Data**: Use context managers for safe writing: `with dataset["train"].get_writer() as writer:`
4. **Read Data**: Access via `dataset["train"].read()` or iterate in configurable batches
5. **Export Options**: Export to Milvus database, Hugging Face Hub, or ModelScope

### Key Design Principles

- **Storage Agnostic**: Unified API across local, S3, and GCS storage via fsspec
- **Resource Safety**: Context managers ensure proper cleanup of file handles and buffers
- **Parquet-Based**: Leverages PyArrow for efficient columnar storage and schema enforcement  
- **Schema-First**: Pydantic models and PyMilvus schemas for data validation
- **Performance Optimized**: Configurable buffering, concurrent I/O, and optional GPU acceleration
- **Vector-Aware**: Native support for float/binary/sparse vectors and specialized vector operations

### Storage Backend Configuration

The library supports three storage types via the `StorageConfig` model:
- **LOCAL**: Standard filesystem operations
- **S3**: S3-compatible storage (AWS S3, MinIO) with credential management
- **GCS**: Google Cloud Storage with service account support

Configuration is managed globally through the singleton `ConfigManager`, enabling seamless storage backend switching without code changes.

### Testing & Examples

- **Test Structure**: Main test logic in `tests/testcases/test_e2e.py`
- **Example Usage**: Comprehensive examples in `example/` directory covering all storage backends and use cases
- **Documentation**: MkDocs-based docs with API reference and usage guides