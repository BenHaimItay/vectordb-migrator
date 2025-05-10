"""
Tests for the adapters module.

This module tests the adapters registry and adapter discovery.
"""

import unittest

from vectordb_migration.adapters import (
    ADAPTERS, list_adapters, get_adapter
)
from vectordb_migration.adapters.pgvector import PgVectorAdapter
from vectordb_migration.adapters.qdrant import QdrantAdapter


class TestAdaptersRegistry(unittest.TestCase):
    """Tests for the adapters registry."""
    
    def test_adapters_registry(self):
        """Test that the adapters registry contains the expected adapters."""
        self.assertIn("pgvector", ADAPTERS)
        self.assertIn("qdrant", ADAPTERS)
        self.assertEqual(ADAPTERS["pgvector"], PgVectorAdapter)
        self.assertEqual(ADAPTERS["qdrant"], QdrantAdapter)
    
    def test_list_adapters(self):
        """Test the list_adapters function."""
        adapters = list_adapters()
        self.assertIsInstance(adapters, list)
        self.assertIn("pgvector", adapters)
        self.assertIn("qdrant", adapters)
    
    def test_get_adapter(self):
        """Test the get_adapter function."""
        pgvector_adapter = get_adapter("pgvector")
        qdrant_adapter = get_adapter("qdrant")
        
        self.assertEqual(pgvector_adapter, PgVectorAdapter)
        self.assertEqual(qdrant_adapter, QdrantAdapter)
    
    def test_get_adapter_case_insensitive(self):
        """Test that get_adapter is case-insensitive."""
        pgvector_adapter = get_adapter("PGVECTOR")
        self.assertEqual(pgvector_adapter, PgVectorAdapter)
    
    def test_get_adapter_not_found(self):
        """Test that get_adapter returns None for unknown adapters."""
        unknown_adapter = get_adapter("unknown")
        self.assertIsNone(unknown_adapter)


if __name__ == "__main__":
    unittest.main()