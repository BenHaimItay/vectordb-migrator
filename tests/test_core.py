"""
Basic tests for the core functionality of the vectordb_migration package.

This module tests the adapter base class, the migrator class, and ensures
they work together properly using mock adapters.
"""

import unittest
from unittest.mock import MagicMock, patch

from vectordb_migration.core.adapter import VectorDBAdapter
from vectordb_migration.core.migrator import DBMigrator


class MockAdapter(VectorDBAdapter):
    """A simple mock adapter for testing the migrator."""
    
    def __init__(self):
        self.connected = False
        self.extracted_data = []
        self.loaded_data = []
    
    def connect(self, **connection_params):
        self.connected = True
        self.connection_params = connection_params
        return True
    
    def disconnect(self):
        self.connected = False
    
    def extract_data(self, **query_params):
        self.query_params = query_params
        return self.extracted_data
    
    def load_data(self, data, **load_params):
        self.loaded_data = data
        self.load_params = load_params
        return True
    
    def get_schema_info(self, collection_name=None):
        return {"name": collection_name or "default", "vector_dimension": 3}


class TestCoreClasses(unittest.TestCase):
    """Tests for the core functionality of the vectordb_migration package."""
    
    def test_migrator_initialization(self):
        """Test that the migrator initializes correctly with valid adapter types."""
        adapters_registry = {
            "source": MockAdapter,
            "target": MockAdapter
        }
        
        migrator = DBMigrator(adapters_registry, "source", "target")
        
        self.assertIsInstance(migrator.source_adapter, MockAdapter)
        self.assertIsInstance(migrator.target_adapter, MockAdapter)
        self.assertEqual(migrator.source_type, "source")
        self.assertEqual(migrator.target_type, "target")
    
    def test_migrator_initialization_invalid_type(self):
        """Test that the migrator raises ValueError for invalid adapter types."""
        adapters_registry = {
            "source": MockAdapter
        }
        
        with self.assertRaises(ValueError):
            DBMigrator(adapters_registry, "source", "invalid")
        
        with self.assertRaises(ValueError):
            DBMigrator(adapters_registry, "invalid", "source")
    
    def test_migrator_migrate(self):
        """Test the basic migration flow."""
        # Define test data
        test_data = [
            {"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": {"name": "item1"}},
            {"id": 2, "vector": [0.4, 0.5, 0.6], "metadata": {"name": "item2"}},
        ]
        
        # Set up the migrator with mock adapters
        source_adapter = MockAdapter()
        source_adapter.extracted_data = test_data
        
        target_adapter = MockAdapter()
        
        adapters_registry = {
            "source": lambda: source_adapter,
            "target": lambda: target_adapter
        }
        
        migrator = DBMigrator(adapters_registry, "source", "target")
        
        # Define source and target parameters
        source_params = {
            "connection": {"host": "source_host"},
            "query": {"limit": 100}
        }
        
        target_params = {
            "connection": {"host": "target_host"},
            "load": {"collection_name": "test_collection"}
        }
        
        # Run the migration
        success = migrator.migrate(source_params, target_params)
        
        # Verify results
        self.assertTrue(success)
        self.assertTrue(source_adapter.connected)
        self.assertEqual(source_adapter.connection_params, {"host": "source_host"})
        self.assertEqual(source_adapter.query_params, {"limit": 100})
        
        self.assertTrue(target_adapter.connected)
        self.assertEqual(target_adapter.connection_params, {"host": "target_host"})
        self.assertEqual(target_adapter.load_params, {"collection_name": "test_collection"})
        
        # Verify that data was correctly passed from source to target
        self.assertEqual(target_adapter.loaded_data, test_data)
    
    def test_migration_with_transform(self):
        """Test migration with a transformation function."""
        # Define test data
        test_data = [
            {"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": {"name": "item1"}},
            {"id": 2, "vector": [0.4, 0.5, 0.6], "metadata": {"name": "item2"}},
        ]
        
        # Define a transformation function
        def transform_func(data):
            for item in data:
                item["metadata"]["transformed"] = True
            return data
        
        # Expected transformed data
        transformed_data = [
            {"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": {"name": "item1", "transformed": True}},
            {"id": 2, "vector": [0.4, 0.5, 0.6], "metadata": {"name": "item2", "transformed": True}},
        ]
        
        # Set up the migrator with mock adapters
        source_adapter = MockAdapter()
        source_adapter.extracted_data = test_data
        
        target_adapter = MockAdapter()
        
        adapters_registry = {
            "source": lambda: source_adapter,
            "target": lambda: target_adapter
        }
        
        migrator = DBMigrator(adapters_registry, "source", "target")
        
        # Run the migration with the transform function
        success = migrator.migrate(
            source_params={"connection": {}, "query": {}},
            target_params={"connection": {}, "load": {}},
            transform_func=transform_func
        )
        
        # Verify results
        self.assertTrue(success)
        self.assertEqual(target_adapter.loaded_data, transformed_data)


if __name__ == "__main__":
    unittest.main()