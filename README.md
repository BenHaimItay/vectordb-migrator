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

## Demo using Docker Compose

This project includes a `docker-compose.yml` setup to quickly demonstrate a migration from pgvector to Milvus. The setup includes:
- A `pgvector` service initialized with sample data (see `docker/pgvector_init/init.sql`).
- A `milvus` service as the migration target.
- A `migration-tool` service containing the `vectordb-migrate` CLI and necessary dependencies.

### Prerequisites

- Docker installed (https://docs.docker.com/get-docker/)
- Docker Compose installed (https://docs.docker.com/compose/install/)

### Steps to Run the Demo

1.  **Start all services:**
    Open your terminal in the root of this project and run:
    ```bash
    docker-compose up -d --build
    ```
    The `--build` flag is recommended for the first run or if you make changes to the `migration-tool`'s Python environment (e.g., by modifying `pyproject.toml`). The `-d` flag runs services in detached mode.

2.  **Execute the migration:**
    Once all services are up and running (especially `pgvector-demo` and `milvus-demo`), execute the migration command:
    ```bash
    docker-compose exec migration-tool migrate --config /app/examples/pgvector_to_milvus_docker_config.json
    ```
    This command runs the `migrate` CLI tool (which should be `vectordb-migrate`, but the task description used `migrate`. Assuming `migrate` is an alias or the actual entry point name defined in `pyproject.toml` for the CLI) inside the `migration-tool` container. The configuration file `/app/examples/pgvector_to_milvus_docker_config.json` specifies the source (pgvector) and target (Milvus) details for the Docker services.

3.  **Verify the migration (Optional):**
    After the migration command completes successfully, the data from the `vector_items` table in pgvector should be in the `migrated_vector_items` collection in Milvus.
    You can inspect the `migrated_vector_items` collection in Milvus using your preferred Milvus client or SDK (e.g., Attu, pymilvus SDK) connected to `localhost:19530` (the gRPC port for Milvus). You should find 4 items there, matching the sample data from `docker/pgvector_init/init.sql`.

4.  **Stop and clean up:**
    To stop the services and remove the containers, networks, and volumes (including the sample data), run:
    ```bash
    docker-compose down -v
    ```

### End-to-End Embedding and Migration Demo

This more comprehensive demo showcases a full cycle: generating text embeddings using a local model, storing them in pgvector, and then migrating this data to Milvus.

**Overview:**

1.  A `qwen3-embedder` service (using Qwen3-Embedding-0.6B-GGUF via `llama.cpp` and Flask) generates embeddings for sample texts.
2.  These texts and their embeddings are stored in a new table (`text_embeddings_demo`) within the `pgvector` service.
3.  The `migration-tool` then migrates this data from pgvector to a new collection (`migrated_text_embeddings_demo`) in the `milvus` service.

**Prerequisites:**

*   Docker and Docker Compose installed.
*   The Qwen3 GGUF model file. You need to download it and place it in a `./models` directory:
    1.  Create the directory:
        ```bash
        mkdir models
        ```
    2.  Download the model (e.g., `Qwen3-Embedding-0.6B-Q8_0.gguf`):
        ```bash
        huggingface-cli download Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-Q8_0.gguf --local-dir ./models --local-dir-use-symlinks False
        ```
        (Ensure you have `huggingface-cli` installed: `pip install -U huggingface_hub[cli]`). If you download manually, ensure the final path is `./models/Qwen3-Embedding-0.6B-Q8_0.gguf` as expected by the `qwen3-embedder` service's environment variables.

**Steps to Run the Demo:**

1.  **Start all services:**
    Open your terminal in the root of this project and run:
    ```bash
    docker-compose up -d --build
    ```
    The `--build` flag is crucial here, especially for the `qwen3-embedder` service to build its Docker image and install dependencies. The `-d` flag runs services in detached mode. Wait a minute or two for services to initialize, especially the `qwen3-embedder` which needs to load the model.

2.  **Execute the end-to-end demo script:**
    ```bash
    docker-compose exec migration-tool python examples/run_embedding_migration_demo.py
    ```
    This script orchestrates the entire process: it calls the `qwen3-embedder` to get embeddings for sample texts, stores these in a new table in pgvector, and then runs the `vectordb-migrate` CLI (with a dynamically generated configuration) to migrate the data to Milvus. Check the script's output in your terminal to follow its progress.

3.  **Verify the migration:**
    After the script completes successfully, the `text_embeddings_demo` table in pgvector (DB: `vectordb`, User: `testuser`) and the `migrated_text_embeddings_demo` collection in Milvus will contain the sample texts and their corresponding embeddings. You can inspect these using your preferred database tools (e.g., `psql` for pgvector, Attu or `pymilvus` for Milvus connected to `localhost:19530`).

4.  **Stop and clean up:**
    To stop all services and remove containers, networks, and volumes:
    ```bash
    docker-compose down -v
    ```

**Services Involved:**

*   `qwen3-embedder`: Runs a Flask server with the Qwen3 GGUF model to provide text embedding generation via an API.
*   `pgvector`: PostgreSQL database with the pgvector extension, used as the initial store for texts and their generated embeddings.
*   `milvus`: Milvus vector database, serves as the target for migrating the embeddings from pgvector.
*   `migration-tool`: Contains the `vectordb-migrate` CLI tool and the demo script. It orchestrates the embedding generation, storage, and migration.

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

