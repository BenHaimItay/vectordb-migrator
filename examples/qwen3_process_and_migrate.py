import os
import sys
import json
import subprocess
import time
import psycopg2
from psycopg2.extras import execute_values
import datasets
from llama_cpp import Llama

# --- Configuration ---
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/Qwen3-Embedding-0.6B-Q8_0.gguf") # Fixed path in Docker image
DATASET_NAME = os.getenv("DATASET_NAME", "CShorten/ML-ArXiv-Papers")
DATASET_SUBSET_SIZE = int(os.getenv("DATASET_SUBSET_SIZE", 100)) # Number of papers to process
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 1024)) # Qwen3-0.6B output dimension
TEXT_COLUMN_TITLE = "title"
TEXT_COLUMN_ABSTRACT = "abstract"

PGVECTOR_HOST = os.getenv("PGVECTOR_HOST", "pgvector-demo")
PGVECTOR_PORT = int(os.getenv("PGVECTOR_PORT", 5432))
PGVECTOR_USER = os.getenv("PGVECTOR_USER", "testuser")
PGVECTOR_PASSWORD = os.getenv("PGVECTOR_PASSWORD", "testpassword")
PGVECTOR_DB = os.getenv("PGVECTOR_DB", "vectordb")
PGVECTOR_TABLE_NAME = os.getenv("PGVECTOR_TABLE_NAME", "arxiv_papers_demo")

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-demo")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_TARGET_COLLECTION_NAME = os.getenv("QDRANT_TARGET_COLLECTION_NAME", "migrated_arxiv_papers_demo")

MIGRATE_CLI_COMMAND = os.getenv("MIGRATE_CLI_COMMAND", "vectordb-migrate")

# --- Initialize LlamaCpp Model ---
llm = None
try:
    print(f"Loading Llama model from: {MODEL_PATH}...")
    llm = Llama(
        model_path=MODEL_PATH,
        embedding=True,
        verbose=True,
        n_ctx=2048, # Context size for potentially longer abstracts
        n_threads=int(os.getenv("LLAMA_THREADS", 4)) # Example: Use 4 threads, adjust as needed
    )
    print("Llama model loaded successfully.")
except Exception as e:
    print(f"Fatal: Error loading Llama model: {e}", file=sys.stderr)
    sys.exit(1)

# --- Load and Prepare Dataset ---
def load_and_prepare_dataset():
    print(f"Loading dataset '{DATASET_NAME}' (first {DATASET_SUBSET_SIZE} records)...")
    try:
        dataset = datasets.load_dataset(DATASET_NAME, split=f"train[:{DATASET_SUBSET_SIZE}]", trust_remote_code=True)
        print(f"Dataset loaded with {len(dataset)} records.")

        print("Preparing dataset: combining title and abstract into 'full_text'...")
        # Ensure text columns exist and handle potential None values by replacing with empty string
        def combine_text(example):
            title = example.get(TEXT_COLUMN_TITLE, "") or ""
            abstract = example.get(TEXT_COLUMN_ABSTRACT, "") or ""
            example['full_text'] = f"{title} {abstract}".strip()
            return example

        dataset = dataset.map(combine_text)
        print("Dataset preparation complete.")
        return dataset
    except Exception as e:
        print(f"Fatal: Error loading or preparing dataset: {e}", file=sys.stderr)
        sys.exit(1)

# --- Generate Embeddings ---
def generate_embeddings_for_dataset(dataset_to_embed):
    print("Generating embeddings for the dataset...")

    def embed_batch(batch):
        texts = batch['full_text']
        try:
            # Filter out None or empty strings before sending to embedding model
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts: # if all texts in batch are empty after filtering
                return {'generated_embedding': [None] * len(texts)}

            # Get embeddings for valid texts
            embedding_results = llm.create_embedding(valid_texts) # input must be a list of strings
            valid_embeddings = [item['embedding'] for item in embedding_results['data']]

            # Map valid embeddings back to original batch structure, inserting None for empty texts
            embedding_map = {text: emb for text, emb in zip(valid_texts, valid_embeddings)}
            final_embeddings = [embedding_map.get(text) for text in texts]

            return {'generated_embedding': final_embeddings}

        except Exception as e:
            print(f"Error embedding batch: {e}", file=sys.stderr)
            # Return None for all items in batch if an error occurs during embedding
            return {'generated_embedding': [None] * len(texts)}

    try:
        embedded_dataset = dataset_to_embed.map(embed_batch, batched=True, batch_size=16)
        print("Embeddings generation complete.")
        return embedded_dataset
    except Exception as e:
        print(f"Fatal: Error during batch embedding process: {e}", file=sys.stderr)
        sys.exit(1)

# --- Store in pgvector ---
def store_in_pgvector(conn, processed_dataset):
    print(f"Storing data in pgvector table '{PGVECTOR_TABLE_NAME}'...")
    try:
        with conn.cursor() as cur:
            print("Enabling pgvector extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print(f"Creating table '{PGVECTOR_TABLE_NAME}' if it doesn't exist...")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {PGVECTOR_TABLE_NAME} (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    embedding vector({EMBEDDING_DIMENSION})
                );
            """)
            # Clear table before new insertion for demo purposes
            print(f"Clearing existing data from '{PGVECTOR_TABLE_NAME}'...")
            cur.execute(f"DELETE FROM {PGVECTOR_TABLE_NAME};")
            conn.commit()

            print(f"Inserting {len(processed_dataset)} items into '{PGVECTOR_TABLE_NAME}'...")

            insert_data = []
            for record in processed_dataset:
                title = record.get(TEXT_COLUMN_TITLE)
                abstract = record.get(TEXT_COLUMN_ABSTRACT)
                embedding = record.get('generated_embedding')

                if embedding is None: # Skip records where embedding failed
                    print(f"Skipping record due to missing embedding: Title='{title}'", file=sys.stderr)
                    continue

                insert_data.append((
                    title or "", # Ensure None is replaced by empty string for TEXT fields
                    abstract or "",
                    embedding
                ))

            if not insert_data:
                print("No valid data to insert into pgvector.")
                return

            execute_values(
                cur,
                f"INSERT INTO {PGVECTOR_TABLE_NAME} (title, abstract, embedding) VALUES %s",
                insert_data
            )
            conn.commit()
            print(f"Successfully inserted {len(insert_data)} items into pgvector.")
    except Exception as e:
        print(f"Fatal: Error storing data in pgvector: {e}", file=sys.stderr)
        conn.rollback() # Rollback on error
        sys.exit(1)

# --- Migrate Data (pgvector to Qdrant) ---
def prepare_and_run_migration():
    print("Preparing migration configuration for pgvector to Qdrant...")

    migration_config = {
        "source": {
            "type": "pgvector",
            "connection_params": {
                "host": PGVECTOR_HOST,
                "port": PGVECTOR_PORT,
                "user": PGVECTOR_USER,
                "password": PGVECTOR_PASSWORD,
                "dbname": PGVECTOR_DB
            },
            "query_params": {
                "collection_name": PGVECTOR_TABLE_NAME,
                "id_column": "id", # Default, but explicit
                "vector_column": "embedding", # Default, but explicit
                "metadata_columns": [TEXT_COLUMN_TITLE, TEXT_COLUMN_ABSTRACT]
            }
        },
        "target": {
            "type": "qdrant",
            "connection_params": {
                "host": QDRANT_HOST,
                "port": QDRANT_PORT
                # Add api_key, etc., if Qdrant is secured
            },
            "load_params": {
                "collection_name": QDRANT_TARGET_COLLECTION_NAME,
                "recreate_collection": True, # Create or recreate the collection
                "vector_size": EMBEDDING_DIMENSION, # Required for Qdrant collection creation
                "distance": "Cosine" # Common distance metric
            }
        }
        # Add transform or other top-level params if needed
    }

    temp_config_path = "/tmp/temp_qwen3_migration_config.json"
    try:
        print(f"Writing dynamic migration config to: {temp_config_path}")
        with open(temp_config_path, 'w') as f:
            json.dump(migration_config, f, indent=2)

        print(f"Running migration: {MIGRATE_CLI_COMMAND} --config {temp_config_path}")
        result = subprocess.run([MIGRATE_CLI_COMMAND, "--config", temp_config_path], capture_output=True, text=True, check=False)

        print(f"Migration CLI Output:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)

        print("Migration CLI executed successfully.")

    except Exception as e:
        print(f"Fatal: Error during migration process: {e}", file=sys.stderr)
        if isinstance(e, subprocess.CalledProcessError):
             print(f"Subprocess error details: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            print(f"Removed temporary config file: {temp_config_path}")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Qwen3 ArXiv Paper Processing and Migration Demo ---")
    start_time = time.time()

    # Step 1: Load and Prepare Dataset
    arxiv_dataset = load_and_prepare_dataset()

    # Step 2: Generate Embeddings
    arxiv_dataset_with_embeddings = generate_embeddings_for_dataset(arxiv_dataset)

    # Filter out records where embedding generation failed
    valid_embedded_records = arxiv_dataset_with_embeddings.filter(lambda x: x['generated_embedding'] is not None)
    print(f"Number of records with successfully generated embeddings: {len(valid_embedded_records)}")

    if len(valid_embedded_records) == 0:
        print("No embeddings were generated successfully. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Step 3: Store in pgvector
    pg_conn = None
    try:
        print(f"Connecting to pgvector database '{PGVECTOR_DB}' at {PGVECTOR_HOST}:{PGVECTOR_PORT}...")
        pg_conn = psycopg2.connect(
            host=PGVECTOR_HOST,
            port=PGVECTOR_PORT,
            user=PGVECTOR_USER,
            password=PGVECTOR_PASSWORD,
            dbname=PGVECTOR_DB
        )
        store_in_pgvector(pg_conn, valid_embedded_records)
    except psycopg2.Error as e:
        print(f"Fatal: pgvector connection or operation failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if pg_conn:
            pg_conn.close()
            print("Closed pgvector connection.")

    # Step 4: Migrate Data from pgvector to Qdrant
    prepare_and_run_migration()

    end_time = time.time()
    print(f"--- Demo completed in {end_time - start_time:.2f} seconds ---")
    print(f"Data from '{DATASET_NAME}' (subset of {DATASET_SUBSET_SIZE}) processed, embedded, stored in pgvector table '{PGVECTOR_TABLE_NAME}', and migrated to Qdrant collection '{QDRANT_TARGET_COLLECTION_NAME}'.")
