"""
Vector Database Migration Core

This module provides the main migrator class for orchestrating migrations between
different vector database systems.
"""

import logging
from typing import Dict, List, Any, Callable, Optional, Type

from vectordb_migration.core.adapter import VectorDBAdapter


logger = logging.getLogger(__name__)


class DBMigrator:
    """Main class for orchestrating database-to-database migrations."""
    
    def __init__(self, adapters_registry: Dict[str, Type[VectorDBAdapter]], source_type: str, target_type: str):
        """
        Initialize the migrator with source and target database types.
        
        Args:
            adapters_registry: Dictionary mapping database types to adapter classes
            source_type: The type of the source database (e.g., "pgvector", "qdrant")
            target_type: The type of the target database (e.g., "qdrant", "pgvector")
            
        Raises:
            ValueError: If an unsupported database type is provided
        """
        if source_type not in adapters_registry:
            raise ValueError(f"Unsupported source database type: {source_type}")
        if target_type not in adapters_registry:
            raise ValueError(f"Unsupported target database type: {target_type}")
        
        self.source_adapter = adapters_registry[source_type]()
        self.target_adapter = adapters_registry[target_type]()
        self.source_type = source_type
        self.target_type = target_type
    
    def migrate(self, 
                source_params: Dict[str, Any], 
                target_params: Dict[str, Any],
                transform_func: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None) -> bool:
        """
        Perform the migration from source to target database.
        
        Args:
            source_params: Parameters for source connection and extraction
            target_params: Parameters for target connection and loading
            transform_func: Optional function to transform data between extraction and loading
            
        Returns:
            bool: True if migration was successful, False otherwise
        """
        logger.info(f"Starting migration from {self.source_type} to {self.target_type}")
        
        # Connect to source
        logger.info(f"Connecting to source ({self.source_type})")
        source_connection_params = source_params.get("connection", {})
        if not self.source_adapter.connect(**source_connection_params):
            logger.error("Failed to connect to source database. Migration aborted.")
            return False
        
        # Extract data
        logger.info(f"Extracting data from {self.source_type}")
        source_query_params = source_params.get("query", {})
        data = self.source_adapter.extract_data(**source_query_params)
        
        if not data:
            logger.warning("No data extracted from source. Migration aborted.")
            self.source_adapter.disconnect()
            return False
        
        logger.info(f"Extracted {len(data)} items from {self.source_type}")
        
        # Transform data if needed
        if transform_func:
            logger.info("Transforming data")
            try:
                data = transform_func(data)
                logger.info(f"Transformed data: {len(data)} items after transformation")
            except Exception as e:
                logger.error(f"Error during data transformation: {e}")
                self.source_adapter.disconnect()
                return False
        
        # Connect to target
        logger.info(f"Connecting to target ({self.target_type})")
        target_connection_params = target_params.get("connection", {})
        if not self.target_adapter.connect(**target_connection_params):
            logger.error("Failed to connect to target database. Migration aborted.")
            self.source_adapter.disconnect()
            return False
        
        # Load data to target
        logger.info(f"Loading data to {self.target_type}")
        target_load_params = target_params.get("load", {})
        success = self.target_adapter.load_data(data, **target_load_params)
        
        # Cleanup
        self.source_adapter.disconnect()
        self.target_adapter.disconnect()
        
        if success:
            logger.info(f"Migration from {self.source_type} to {self.target_type} completed successfully")
        else:
            logger.error(f"Migration from {self.source_type} to {self.target_type} failed")
        
        return success