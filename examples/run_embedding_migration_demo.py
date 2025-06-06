import os
import json
import requests
import subprocess
import time
import psycopg2
from psycopg2.extras import execute_values

# --- Configuration ---
QWEN3_EMBEDDER_URL = "http://qwen3-embedder-demo:5000/embed"
PGVECTOR_HOST = os.getenv("PGVECTOR_HOST", "pgvector-demo")
PGVECTOR_PORT = int(os.getenv("PGVECTOR_PORT", 5432))
PGVECTOR_USER = os.getenv("PGVECTOR_USER", "testuser")
PGVECTOR_PASSWORD = os.getenv("PGVECTOR_PASSWORD", "testpassword")
PGVECTOR_DB = os.getenv("PGVECTOR_DB", "vectordb")
PGVECTOR_TABLE_NAME = "text_embeddings_demo"
MIGRATION_CONFIG_PATH = "/app/examples/pgvector_to_milvus_docker_config.json" # Path inside migration-tool container
MILVUS_TARGET_COLLECTION_NAME = "migrated_text_embeddings_demo"

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Exploring the vast universe and its many mysteries.",
    "Artificial intelligence is transforming industries worldwide.",
    "A delicious recipe for homemade pasta.",
    "The history of ancient civilizations is fascinating.",
    "Climate change poses a significant global challenge.",
    "Discovering new music is a joyful experience.",
    "The serene beauty of a mountain landscape at dawn."
]
EMBEDDING_DIMENSION = 1024 # For Qwen3-Embedding-0.6B

# --- Helper Function: Get Embeddings ---
def get_embeddings(texts: list[str], max_retries=5, retry_delay=10) -> list[list[float]]:
    """
    Sends texts to the Qwen3 embedder service and returns embeddings.
    Includes retries for service readiness.
    """
    print(f"Requesting embeddings for {len(texts)} texts from {QWEN3_EMBEDDER_URL}...")
    for attempt in range(max_retries):
        try:
            response = requests.post(QWEN3_EMBEDDER_URL, json={"texts": texts}, timeout=60) # Increased timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            if "embeddings" in data and isinstance(data["embeddings"], list):
                print(f"Successfully retrieved {len(data['embeddings'])} embeddings.")
                return data["embeddings"]
            else:
                print(f"Invalid response format from embedder: {data}")
                return [] # Or raise an error
        except requests.exceptions.RequestException as e:
            print(f"Embedding service request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached for embedding service.")
                raise
    return []

# --- Helper Function: Setup pgvector Table ---
def setup_pgvector_table(conn):
    """Creates the pgvector table if it doesn't exist."""
    with conn.cursor() as cur:
        print("Enabling pgvector extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print(f"Creating table '{PGVECTOR_TABLE_NAME}' if it doesn't exist...")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {PGVECTOR_TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                text_content TEXT,
                embedding vector({EMBEDDING_DIMENSION})
            );
        """)
        conn.commit()
        print(f"Table '{PGVECTOR_TABLE_NAME}' setup complete.")

# --- Helper Function: Insert into pgvector ---
def insert_into_pgvector(conn, texts: list[str], embeddings: list[list[float]]):
    """Inserts texts and their embeddings into the pgvector table."""
    if not texts or not embeddings or len(texts) != len(embeddings):
        print("Invalid input for pgvector insertion: texts and embeddings must be non-empty and of equal length.")
        return

    with conn.cursor() as cur:
        print(f"Inserting {len(texts)} items into '{PGVECTOR_TABLE_NAME}'...")
        data_to_insert = [(texts[i], embeddings[i]) for i in range(len(texts))]
        execute_values(
            cur,
            f"INSERT INTO {PGVECTOR_TABLE_NAME} (text_content, embedding) VALUES %s",
            data_to_insert
        )
        conn.commit()
        print(f"Successfully inserted {len(texts)} items into pgvector.")

# --- Helper Function: Run Migration ---
def run_migration():
    """Modifies the migration config and runs the migration CLI tool."""
    print(f"Preparing migration using base config: {MIGRATION_CONFIG_PATH}")
    try:
        with open(MIGRATION_CONFIG_PATH, 'r') as f:
            config_data = json.load(f)

        # Modify config for this specific demo run
        config_data["source"]["type"] = "pgvector" # Ensure source type is correct
        config_data["source"]["connection_params"] = {
            "host": PGVECTOR_HOST,
            "port": PGVECTOR_PORT,
            "user": PGVECTOR_USER,
            "password": PGVECTOR_PASSWORD,
            "dbname": PGVECTOR_DB
        }
        config_data["source"]["query_params"]["collection_name"] = PGVECTOR_TABLE_NAME
        # Ensure metadata_columns and other pgvector specific params are suitable or removed if not needed
        # For this specific table, we might not have pre-defined metadata columns in the same way.
        # The default pgvector adapter extracts 'id', 'vector', 'metadata'.
        # We need to ensure the adapter can get text_content as part of metadata or handle it.
        # For simplicity, let's assume the adapter will pick up all other columns as metadata.
        # Or, more explicitly, define how metadata should be extracted if default is not suitable.
        # The current generic pgvector adapter might need adjustment if we want 'text_content' in a specific metadata field.
        # For now, we assume default behavior is acceptable or the adapter is flexible.
        # If the default pgvector adapter uses `metadata_columns` from config, we might need to specify it:
        config_data["source"]["query_params"]["metadata_columns"] = ["text_content"]


        config_data["target"]["load_params"]["collection_name"] = MILVUS_TARGET_COLLECTION_NAME
        # Ensure target Milvus connection params are correct if they differ from the base config
        config_data["target"]["connection_params"] = {
            "alias": "default", # Or a specific alias if needed
            "host": "milvus-demo", # Service name in Docker
            "port": "19530"
        }


        temp_config_path = "/tmp/temp_embedding_migration_config.json"
        print(f"Writing modified migration config to: {temp_config_path}")
        with open(temp_config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        migrate_command = os.getenv("MIGRATE_CLI_COMMAND", "vectordb-migrate") # Use actual CLI command

        print(f"Running migration: {migrate_command} --config {temp_config_path}")
        # Using subprocess.run
        result = subprocess.run([migrate_command, "--config", temp_config_path], capture_output=True, text=True, check=False)

        print(f"Migration CLI Output:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)

        print("Migration CLI executed successfully.")

    except FileNotFoundError:
        print(f"Error: Base migration config file not found at {MIGRATION_CONFIG_PATH}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {MIGRATION_CONFIG_PATH}")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error during migration process: {e}")
        # Output is already printed
        raise
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            print(f"Removed temporary config file: {temp_config_path}")


# --- Main Logic ---
if __name__ == "__main__":
    print("--- Starting End-to-End Embedding and Migration Demo ---")

    # 1. Get embeddings for sample texts
    embeddings = []
    try:
        embeddings = get_embeddings(SAMPLE_TEXTS)
    except Exception as e:
        print(f"Failed to get embeddings: {e}. Exiting demo.")
        exit(1)

    if not embeddings or len(embeddings) != len(SAMPLE_TEXTS):
        print("Embeddings generation failed or returned incorrect number of vectors. Exiting.")
        exit(1)
    print(f"Successfully generated {len(embeddings)} embeddings of dimension {len(embeddings[0]) if embeddings else 'N/A'}.")

    # 2. Connect to pgvector and setup table
    pg_conn = None
    try:
        print(f"Connecting to pgvector at {PGVECTOR_HOST}:{PGVECTOR_PORT} DB: {PGVECTOR_DB} User: {PGVECTOR_USER}")
        pg_conn = psycopg2.connect(
            host=PGVECTOR_HOST,
            port=PGVECTOR_PORT,
            user=PGVECTOR_USER,
            password=PGVECTOR_PASSWORD,
            dbname=PGVECTOR_DB
        )
        setup_pgvector_table(pg_conn)
        insert_into_pgvector(pg_conn, SAMPLE_TEXTS, embeddings)
        print("Data successfully stored in pgvector.")
    except psycopg2.Error as e:
        print(f"pgvector operations failed: {e}. Exiting demo.")
        if pg_conn:
            pg_conn.close()
        exit(1)
    finally:
        if pg_conn:
            pg_conn.close()
            print("Closed pgvector connection.")

    # 3. Run the migration
    try:
        print("\n--- Starting Migration to Milvus ---")
        run_migration()
        print("--- Migration Process Completed Successfully ---")
    except Exception as e:
        print(f"Migration failed: {e}. Exiting demo.")
        exit(1)

    # (Optional) Verification step for Milvus can be added here if pymilvus is available
    print("\nDemo finished. To verify Milvus data, connect to milvus-demo:19530 and check collection '{}'.".format(MILVUS_TARGET_COLLECTION_NAME))
    print(f"The collection should contain {len(SAMPLE_TEXTS)} items.")
