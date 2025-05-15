"""
Vector database adapters package

This package contains adapters for different vector databases.
"""

from vectordb_migration.adapters.pgvector import PgVectorAdapter
from vectordb_migration.adapters.pinecone import PineconeAdapter
from vectordb_migration.adapters.qdrant import QdrantAdapter

# Registry of available adapters
ADAPTERS = {
    "pgvector": PgVectorAdapter,
    "qdrant": QdrantAdapter,
    "pinecone": PineconeAdapter
}


def list_adapters():
    """Return a list of available adapter names."""
    return list(ADAPTERS.keys())


def get_adapter(adapter_name):
    """Get an adapter class by name.
    
    Args:
        adapter_name: Name of the adapter.
        
    Returns:
        The adapter class or None if not found.
    """
    return ADAPTERS.get(adapter_name.lower())