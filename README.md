# Vector DB Migration

A Python library for migrating vector embeddings between different vector database systems. This tool simplifies the process of moving vector data between different database implementations while maintaining data consistency.


[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/itaybenhaim/vectordb-migration/blob/main/LICENSE)

## Features

- Seamless migration between different vector database systems
- Support for pgvector (PostgreSQL), Qdrant, Pinecone, and Milvus
- Configurable data transformation during migration
- Easy-to-use command-line interface
- Extendable adapter architecture for additional database support

## Installation

```bash
pip install vectordb-migration
```

## Quick Start

1. Create a configuration file `config.json`:

```json
{
  "source": {
    "type": "pgvector",
    "connection": {
      "host": "localhost",
      "dbname": "vectordb",
      "user": "postgres",
      "password": "password"
    },
    "query": {
      "table_name": "items",
      "id_column": "id",
      "vector_column": "embedding",
      "metadata_columns": ["name", "description"]
    }
  },
  "target": {
    "type": "qdrant",
    "connection": {
      "host": "localhost",
      "port": 6333
    },
    "load": {
      "collection_name": "items_collection",
      "recreate_collection": true,
      "distance": "Cosine"
    }
  }
}
```

Or for Milvus:

```json
{
  "source": {
    "type": "qdrant",
    "connection": {
      "host": "localhost",
      "port": 6333
    },
    "query": {
      "collection_name": "source_items"
    }
  },
  "target": {
    "type": "milvus",
    "connection": {
      "uri": "http://localhost:19530",
      "token": "your_milvus_token_if_any",
      "db_name": "default"
    },
    "load": {
      "collection_name": "target_items_milvus"
    }
  }
}
```

Or for Pinecone:

```json
{
  "source": {
    "type": "pgvector",
    "connection": {
      "host": "localhost",
      "dbname": "vectordb",
      "user": "postgres",
      "password": "password"
    },
    "query": {
      "table_name": "items",
      "id_column": "id",
      "vector_column": "embedding",
      "metadata_columns": ["name", "description"]
    }
  },
  "target": {
    "type": "pinecone",
    "connection": {
      "api_key": "YOUR_PINECONE_API_KEY",
      "environment": "YOUR_PINECONE_ENVIRONMENT"
    },
    "load": {
      "index_name": "items_collection",
      "create_index": true,
      "dimension": 1536,
      "metric": "cosine"
    }
  }
}
```

2. Run the migration:

```bash
vectordb-migrate --config config.json
```

Or use the Python API:

```python
import vectordb_migration as vdbm

vdbm.run_migration("config.json")
```

## Supported Databases

- **pgvector**: PostgreSQL with the pgvector extension
- **Qdrant**: Qdrant vector database
- **Pinecone**: Pinecone vector database
- **Milvus**: Milvus vector database

## Command-Line Usage

```
usage: vectordb-migrate [-h] --config CONFIG [--transform TRANSFORM] [--verbose] [--version]

Vector Database Migration Tool

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the migration configuration JSON file
  --transform TRANSFORM
                        Path to a Python module with a 'transform' function for customizing data
  --verbose             Enable verbose logging
  --version             Print version information and exit
```

## Custom Transformations

You can apply custom transformations to your data during migration by creating a Python module with a `transform` function:

```python
# transform_module.py

def transform(data):
    """
    Transform data during migration.
    
    Args:
        data: List of items with id, vector, and metadata
        
    Returns:
        Transformed list of items
    """
    for item in data:
        # Add source information to metadata
        if "metadata" not in item:
            item["metadata"] = {}
        item["metadata"]["source"] = "original_database"
        
        # Add other custom transformations here
        
    return data
```

Then use it in your migration:

```bash
vectordb-migrate --config config.json --transform transform_module.py
```

## Python API

```python
import vectordb_migration as vdbm

# List available adapters
adapters = vdbm.list_adapters()
print(f"Available adapters: {adapters}")

# Run a migration with a configuration file
success = vdbm.run_migration("config.json")

# Run a migration with a custom transformation
success = vdbm.run_migration("config.json", "transform_module.py")

# Enable verbose logging
success = vdbm.run_migration("config.json", verbose=True)
```

## Qwen3 Local Processing and Migration Demo (pgvector to Qdrant)

This demo showcases an automated end-to-end pipeline using Docker Compose. The pipeline performs the following:
1.  Downloads and utilizes a local Qwen3 GGUF model for text embedding via `llama.cpp`.
2.  Loads a subset of the "CShorten/ML-ArXiv-Papers" dataset from Hugging Face.
3.  Generates embeddings for these academic papers.
4.  Stores the original text content (title, abstract) and the generated embeddings into a pgvector database.
5.  Migrates this data from pgvector to a Qdrant database using the `vectordb-migration` tool.

The entire process is orchestrated by the `qwen3-processor` service defined in `docker-compose.yml`.

### Services Involved

*   **`qwen3-processor`**: The main orchestrator service. Its Docker image is built with:
    *   `llama.cpp` compiled from source.
    *   The Qwen3 GGUF model downloaded from Hugging Face.
    *   All necessary Python dependencies, including `llama-cpp-python`, `datasets`, `psycopg2-binary`, and `qdrant-client`.
    *   The `vectordb-migration` tool itself (installed from project source).
    When this service starts, it automatically runs the `examples/qwen3_process_and_migrate.py` script, which drives the entire workflow.
*   **`pgvector-demo`**: A PostgreSQL database service with the pgvector extension enabled. It serves as the source database where texts and their embeddings are initially stored by the `qwen3-processor`. It is initialized with sample data via `docker/pgvector_init/init.sql`, though this demo script will create and use its own table (`arxiv_papers_demo`).
*   **`qdrant-demo`**: A Qdrant vector database service. It serves as the target database to which the data from pgvector will be migrated.

### Prerequisites

*   **Docker and Docker Compose**: Ensure they are installed and operational on your system.
*   **Disk Space**: Sufficient disk space is required for:
    *   Docker images (Python base, pgvector, Qdrant).
    *   The `llama.cpp` build artifacts within the `qwen3-processor` image (can be ~1-2GB).
    *   The downloaded Qwen3 GGUF model (`Qwen3-Embedding-0.6B-Q8_0.gguf` is ~0.6GB) within the `qwen3-processor` image.
*   **Internet Connection**: Required during the build of the `qwen3-processor` image for:
    *   Cloning the `llama.cpp` repository from GitHub.
    *   Downloading Python packages.
    *   Downloading the Qwen3 GGUF model and the "CShorten/ML-ArXiv-Papers" dataset from Hugging Face.

### How to Run the Demo

1.  **Build and Start Services:**
    Navigate to the root directory of this project in your terminal and run:
    ```bash
    docker-compose up -d --build
    ```
    *   The `--build` flag is essential, especially for the `qwen3-processor` service, as it needs to compile `llama.cpp` and download models. This might take several minutes on the first run.
    *   The `-d` flag runs the services in detached mode.

2.  **Monitor Progress:**
    The main processing script (`examples/qwen3_process_and_migrate.py`) starts automatically within the `qwen3-processor-demo` container. You can monitor its progress by viewing the container's logs:
    ```bash
    docker-compose logs -f qwen3-processor-demo
    ```
    Look for log messages indicating dataset loading, embedding generation, storage in pgvector, and the migration process.

3.  **Verification:**
    Once the script running in `qwen3-processor-demo` indicates completion (or if you encounter errors, check its logs):
    *   **pgvector**: Connect to the pgvector database. You can use any PostgreSQL client or `psql` CLI.
        *   Host: `localhost` (or your Docker host IP)
        *   Port: `5432`
        *   User: `testuser`
        *   Password: `testpassword`
        *   Database: `vectordb`
        Inspect the `arxiv_papers_demo` table (e.g., `SELECT COUNT(*) FROM arxiv_papers_demo; SELECT * FROM arxiv_papers_demo LIMIT 5;`). It should contain a number of rows equal to `DATASET_SUBSET_SIZE` (default 100 in the script) with titles, abstracts, and embeddings.
    *   **Qdrant**: Access the Qdrant dashboard, typically at `http://localhost:6333/dashboard`.
        Look for a collection named `migrated_arxiv_papers_demo`. It should contain the data migrated from pgvector, including vectors and payloads (title, abstract). You can check the point count and examine individual points.

4.  **Stopping the Demo:**
    To stop all services and remove the containers, networks, and volumes (including the data in pgvector and Qdrant, and the model downloaded inside the `qwen3-processor` container):
    ```bash
    docker-compose down -v
    ```

### Customization (Optional)

*   **Dataset Size**: You can change the number of papers processed by modifying `DATASET_SUBSET_SIZE` at the top of the `examples/qwen3_process_and_migrate.py` script. Rebuild the `qwen3-processor` image if you change the script and want it to take effect: `docker-compose build qwen3-processor && docker-compose up -d qwen3-processor`.
*   **Model & Paths**: The Qwen3 model and its path are defined in the `docker/qwen3_processor/Dockerfile` (for download) and `examples/qwen3_process_and_migrate.py` (for loading). Configurations like `PGVECTOR_TABLE_NAME` and `QDRANT_TARGET_COLLECTION_NAME` are also in the script.

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/itaybenhaim/vectordb-migration.git 
cd vectordb-migration

# Install development dependencies
pip install -e ".[dev]"
```

### Testing

```bash
pytest
```

### Docker Environment for Testing

A Docker Compose environment is included for development and testing:

```bash
docker-compose up -d
```

This will start PostgreSQL with pgvector and Qdrant services.

## Project Status

This project is in active development. Current focus:

- Adding support for more vector databases
- Enhancing migration performance for large datasets 
- Implementing incremental migration capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

