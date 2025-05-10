"""
PostgreSQL with pgvector extension adapter

This module provides the adapter for PostgreSQL database with the pgvector extension.
"""

import logging
from typing import Dict, List, Any

from vectordb_migration.core.adapter import VectorDBAdapter


logger = logging.getLogger(__name__)


class PgVectorAdapter(VectorDBAdapter):
    """Adapter for PostgreSQL with pgvector extension."""
    
    def __init__(self):
        """Initialize a new PgVector adapter."""
        self.conn = None
        self.cursor = None
        
    def connect(self, **connection_params) -> bool:
        """Connect to the PostgreSQL database using psycopg2.
        
        Args:
            **connection_params: Connection parameters for PostgreSQL.
                - host: Database host address
                - dbname: Database name
                - user: Username
                - password: Password
                - port: Port number (default: 5432)
                
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        try:
            import psycopg2
            self.conn = psycopg2.connect(
                host=connection_params.get("host", "localhost"),
                dbname=connection_params.get("dbname", "vectordb"),
                user=connection_params.get("user", "user"),
                password=connection_params.get("password", "password"),
                port=connection_params.get("port", 5432)
            )
            self.cursor = self.conn.cursor()
            logger.debug(f"Connected to PostgreSQL: {connection_params.get('host')}:{connection_params.get('port')}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close the PostgreSQL connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        self.cursor = None
        self.conn = None
        logger.debug("Disconnected from PostgreSQL")
    
    def extract_data(self, **query_params) -> List[Dict[str, Any]]:
        """Extract vector data from a PostgreSQL table.
        
        Args:
            **query_params: Query parameters for extracting data.
                - table_name: Table name to query (default: "items")
                - id_column: Column name for IDs (default: "id")
                - vector_column: Column name for vector embeddings (default: "embedding")
                - metadata_columns: List of column names to include as metadata (default: ["name"])
                - limit: Maximum number of records to extract (default: None)
                - offset: Number of records to skip (default: 0)
                - filter_condition: Optional WHERE clause (default: None)
                
        Returns:
            List[Dict[str, Any]]: List of items with id, vector, and metadata.
            
        Raises:
            ConnectionError: If not connected to the database.
        """
        if not self.conn or not self.cursor:
            raise ConnectionError("Not connected to PostgreSQL database")
        
        table_name = query_params.get("table_name", "items")
        id_column = query_params.get("id_column", "id")
        vector_column = query_params.get("vector_column", "embedding")
        metadata_columns = query_params.get("metadata_columns", ["name"])
        limit = query_params.get("limit")
        offset = query_params.get("offset", 0)
        filter_condition = query_params.get("filter_condition")
        
        # Build a SELECT query with the specified columns
        columns = [id_column, vector_column] + metadata_columns
        query = f"SELECT {', '.join(columns)} FROM {table_name}"
        
        # Add filter condition if provided
        if filter_condition:
            query += f" WHERE {filter_condition}"
            
        # Add limit and offset if provided
        if limit is not None:
            query += f" LIMIT {limit}"
        if offset > 0:
            query += f" OFFSET {offset}"
            
        query += ";"
        
        try:
            logger.debug(f"Executing query: {query}")
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            
            # Transform rows into dictionaries
            data = []
            for row in rows:
                item = {
                    "id": row[0],
                    "vector": list(row[1]),  # Convert vector to list for JSON serialization
                    "metadata": {}
                }
                # Add metadata columns to the metadata dict
                for i, col_name in enumerate(metadata_columns):
                    item["metadata"][col_name] = row[i + 2]  # +2 because id and vector are first
                data.append(item)
            
            logger.debug(f"Extracted {len(data)} items from PostgreSQL")
            return data
        except Exception as e:
            logger.error(f"Error extracting data from PostgreSQL: {e}")
            return []
    
    def load_data(self, data: List[Dict[str, Any]], **load_params) -> bool:
        """Load vector data into a PostgreSQL table.
        
        Args:
            data: List of items with id, vector, and metadata.
            **load_params: Load parameters.
                - table_name: Table name to load data into (default: "items")
                - id_column: Column name for IDs (default: "id")
                - vector_column: Column name for vector embeddings (default: "embedding")
                - recreate_table: Whether to drop and recreate the table (default: False)
                - batch_size: Number of items per batch (default: 100)
                
        Returns:
            bool: True if loading was successful, False otherwise.
            
        Raises:
            ConnectionError: If not connected to the database.
        """
        if not self.conn or not self.cursor:
            raise ConnectionError("Not connected to PostgreSQL database")
        
        table_name = load_params.get("table_name", "items")
        id_column = load_params.get("id_column", "id")
        vector_column = load_params.get("vector_column", "embedding")
        recreate_table = load_params.get("recreate_table", False)
        batch_size = load_params.get("batch_size", 100)
        
        # Check if we need to create/recreate the table
        if recreate_table:
            if not data:
                logger.error("Cannot recreate table: No data provided to determine vector dimensions")
                return False
                
            vector_dim = len(data[0]["vector"])
            
            # Get all metadata keys from the first item
            metadata_columns = list(data[0]["metadata"].keys()) if data and "metadata" in data[0] else []
            
            # Drop table if it exists and recreate_table is True
            try:
                self.cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
                
                # Create pgvector extension if not exists
                self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Build column definitions for metadata
                metadata_defs = []
                for col in metadata_columns:
                    metadata_defs.append(f"{col} TEXT")
                
                # Create the table with id, vector, and metadata columns
                create_query = f"""
                CREATE TABLE {table_name} (
                    {id_column} SERIAL PRIMARY KEY,
                    {vector_column} VECTOR({vector_dim}),
                    {', '.join(metadata_defs)}
                );
                """
                self.cursor.execute(create_query)
                logger.info(f"Created table {table_name} with vector dimension {vector_dim}")
            except Exception as e:
                logger.error(f"Error creating table in PostgreSQL: {e}")
                return False
        
        # Insert data in batches
        try:
            count = 0
            batch_count = 0
            batch = []
            
            for item in data:
                # Get all metadata columns for this item
                metadata_columns = list(item.get("metadata", {}).keys())
                metadata_values = [item.get("metadata", {}).get(col) for col in metadata_columns]
                
                # Build the INSERT query
                columns = [id_column, vector_column] + metadata_columns
                placeholders = ["%s"] * len(columns)
                
                insert_query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)});
                """
                
                # Add to batch
                batch.append([item["id"], item["vector"]] + metadata_values)
                count += 1
                
                # Execute batch if full
                if len(batch) >= batch_size:
                    self.cursor.executemany(insert_query, batch)
                    batch = []
                    batch_count += 1
                    logger.debug(f"Inserted batch {batch_count} ({batch_size} items)")
            
            # Insert any remaining items
            if batch:
                self.cursor.executemany(insert_query, batch)
                batch_count += 1
                logger.debug(f"Inserted final batch {batch_count} ({len(batch)} items)")
            
            self.conn.commit()
            logger.info(f"Successfully loaded {count} items into PostgreSQL table {table_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading data into PostgreSQL: {e}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def get_schema_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Get information about the table schema including vector dimensions.
        
        Args:
            collection_name: Table name to inspect (default: "items")
            
        Returns:
            Dict[str, Any]: Schema information.
            
        Raises:
            ConnectionError: If not connected to the database.
        """
        if not self.conn or not self.cursor:
            raise ConnectionError("Not connected to PostgreSQL database")
        
        table_name = collection_name or "items"
        
        try:
            # Get column information
            self.cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}';
            """)
            columns = self.cursor.fetchall()
            
            # Try to find vector dimension by checking a sample row
            vector_columns = [col[0] for col in columns if col[1] == 'vector']
            vector_dim = None
            
            if vector_columns:
                vector_col = vector_columns[0]
                self.cursor.execute(f"SELECT {vector_col} FROM {table_name} LIMIT 1;")
                sample = self.cursor.fetchone()
                if sample and sample[0]:
                    vector_dim = len(sample[0])
            
            return {
                "table_name": table_name,
                "columns": [{"name": col[0], "type": col[1]} for col in columns],
                "vector_columns": vector_columns,
                "vector_dimension": vector_dim
            }
        except Exception as e:
            logger.error(f"Error getting schema info from PostgreSQL: {e}")
            return {}