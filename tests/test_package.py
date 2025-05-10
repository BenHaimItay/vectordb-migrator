"""
Tests for the package entry points and integration between components.

This module tests the main package functionality and integration points.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
import os
import tempfile

import vectordb_migration


class TestPackageIntegration(unittest.TestCase):
    """Tests for package-level functionality."""
    
    def test_version_exists(self):
        """Test that the version attribute exists."""
        self.assertTrue(hasattr(vectordb_migration, "__version__"))
        self.assertIsInstance(vectordb_migration.__version__, str)
    
    def test_exported_symbols(self):
        """Test that expected symbols are exported."""
        expected_exports = [
            "VectorDBAdapter", "DBMigrator", "ADAPTERS",
            "list_adapters", "get_adapter", "run_migration"
        ]
        for symbol in expected_exports:
            self.assertTrue(hasattr(vectordb_migration, symbol), 
                          f"Expected symbol {symbol} to be exported")
    
    @patch('vectordb_migration.cli.migrate.run_migration')
    def test_run_migration_convenience_function(self, mock_run_migration):
        """Test the run_migration convenience function."""
        # Setup mock
        mock_run_migration.return_value = True
        
        # Call the function
        result = vectordb_migration.run_migration(
            config_file="config.json",
            transform_file="transform.py",
            verbose=True
        )
        
        # Verify
        self.assertTrue(result)
        mock_run_migration.assert_called_once_with(
            "config.json", "transform.py", True
        )
    
    @patch('vectordb_migration.adapters.ADAPTERS')
    def test_list_adapters_integration(self, mock_adapters):
        """Test that list_adapters uses the ADAPTERS registry."""
        # Setup mock
        mock_adapters.keys.return_value = ["mock1", "mock2"]
        
        # Call the function
        result = vectordb_migration.list_adapters()
        
        # Verify
        self.assertEqual(result, ["mock1", "mock2"])
        mock_adapters.keys.assert_called_once()
    
    @patch('vectordb_migration.core.VectorDBAdapter')
    def test_adapter_base_class_exported(self, _):
        """Test that the VectorDBAdapter base class is exported."""
        # This test just verifies that the import works
        self.assertEqual(vectordb_migration.VectorDBAdapter, 
                       vectordb_migration.core.VectorDBAdapter)
    
    @patch('vectordb_migration.adapters.pgvector.PgVectorAdapter')
    @patch('vectordb_migration.core.DBMigrator')
    def test_pgvector_to_pgvector_migration_happy_path(self, mock_migrator_class, mock_adapter_class):
        """Test a complete happy flow migration from pgvector to pgvector."""
        # Setup mock adapters
        source_adapter = MagicMock()
        target_adapter = MagicMock()
        
        # Configure the source adapter to return test data when extract_data is called
        test_data = [
            {"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": {"name": "Item 1", "category": "product"}},
            {"id": 2, "vector": [0.4, 0.5, 0.6], "metadata": {"name": "Item 2", "category": "product"}},
            {"id": 3, "vector": [0.7, 0.8, 0.9], "metadata": {"name": "Item 3", "category": "product"}}
        ]
        source_adapter.extract_data.return_value = test_data
        
        # Configure the target adapter to return success when load_data is called
        target_adapter.load_data.return_value = True
        
        # Arrange for get_adapter to return our mocks
        def mock_get_adapter(adapter_type):
            return mock_adapter_class
        
        # Arrange for adapter instance creation
        mock_adapter_class.side_effect = [source_adapter, target_adapter]
        
        # Create a temporary config file
        config = {
            "source": {
                "type": "pgvector",
                "connection": {
                    "host": "source-host",
                    "dbname": "source-db",
                    "user": "user",
                    "password": "password"
                },
                "query": {
                    "table_name": "source_items",
                    "id_column": "id",
                    "vector_column": "embedding",
                    "metadata_columns": ["name", "category"],
                    "filter_condition": "category = 'product'"
                }
            },
            "target": {
                "type": "pgvector",
                "connection": {
                    "host": "target-host",
                    "dbname": "target-db",
                    "user": "user",
                    "password": "password"
                },
                "load": {
                    "table_name": "target_items",
                    "id_column": "id",
                    "vector_column": "embedding",
                    "recreate_table": True,
                    "batch_size": 100
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            json.dump(config, f)
        
        # Configure mock migrator
        mock_migrator = MagicMock()
        mock_migrator.execute_migration.return_value = True
        mock_migrator_class.return_value = mock_migrator
        
        # Run the migration using the package function
        with patch('vectordb_migration.get_adapter', mock_get_adapter):
            result = vectordb_migration.run_migration(config_path, verbose=True)
        
        # Clean up temporary file
        os.unlink(config_path)
        
        # Verify
        self.assertTrue(result)
        
        # Verify adapter creation
        self.assertEqual(mock_adapter_class.call_count, 2)
        
        # Verify source adapter was configured correctly
        source_adapter.connect.assert_called_once_with(
            host="source-host", 
            dbname="source-db", 
            user="user", 
            password="password"
        )
        source_adapter.extract_data.assert_called_once_with(
            table_name="source_items",
            id_column="id",
            vector_column="embedding",
            metadata_columns=["name", "category"],
            filter_condition="category = 'product'"
        )
        
        # Verify target adapter was configured correctly
        target_adapter.connect.assert_called_once_with(
            host="target-host", 
            dbname="target-db", 
            user="user", 
            password="password"
        )
        target_adapter.load_data.assert_called_once_with(
            test_data,
            table_name="target_items",
            id_column="id",
            vector_column="embedding",
            recreate_table=True,
            batch_size=100
        )
        
        # Verify disconnect was called
        source_adapter.disconnect.assert_called_once()
        target_adapter.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()