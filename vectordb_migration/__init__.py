"""
Vector Database Migration Library

A library for migrating vector embeddings between different vector database systems.

Currently supported databases:
- pgvector (PostgreSQL with vector extension)
- Qdrant
"""

from vectordb_migration.core import VectorDBAdapter, DBMigrator
from vectordb_migration.adapters import ADAPTERS, list_adapters, get_adapter

__version__ = "0.1.0"

__all__ = [
    "VectorDBAdapter",
    "DBMigrator",
    "ADAPTERS",
    "list_adapters",
    "get_adapter",
]

def run_migration(config_file, transform_file=None, verbose=False):
    """
    Run a migration using a configuration file.
    
    This is a convenience function that can be imported directly from the package.
    
    Args:
        config_file: Path to the configuration JSON file
        transform_file: Optional path to a transformation module
        verbose: Whether to enable verbose logging
        
    Returns:
        bool: True if migration was successful, False otherwise
    """
    from vectordb_migration.cli.migrate import run_migration as _run_migration
    return _run_migration(config_file, transform_file, verbose)