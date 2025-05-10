"""
Tests for the CLI module of the vectordb_migration package.

This module tests the command-line interface functionality.
"""

import unittest
import tempfile
import json
import os
from unittest.mock import patch, MagicMock

from vectordb_migration.cli.migrate import load_config, load_transform_function, run_migration


class TestCLI(unittest.TestCase):
    """Tests for the CLI functionality."""
    
    def test_load_config(self):
        """Test loading a valid configuration file."""
        config = {
            "source": {
                "type": "pgvector",
                "connection": {"host": "localhost"}
            },
            "target": {
                "type": "qdrant",
                "connection": {"host": "localhost"}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            loaded_config = load_config(config_path)
            self.assertEqual(loaded_config, config)
        finally:
            os.unlink(config_path)
    
    def test_load_config_invalid_json(self):
        """Test loading an invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("This is not JSON")
            config_path = f.name
        
        try:
            with self.assertRaises(ValueError):
                load_config(config_path)
        finally:
            os.unlink(config_path)
    
    def test_load_config_missing_required_keys(self):
        """Test loading a config file missing required keys."""
        # Missing 'target'
        config = {
            "source": {
                "type": "pgvector",
                "connection": {"host": "localhost"}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            with self.assertRaises(ValueError):
                load_config(config_path)
        finally:
            os.unlink(config_path)
    
    def test_load_transform_function(self):
        """Test loading a valid transform function."""
        transform_code = """
def transform(data):
    for item in data:
        if "metadata" not in item:
            item["metadata"] = {}
        item["metadata"]["transformed"] = True
    return data
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(transform_code)
            transform_path = f.name
        
        try:
            transform_func = load_transform_function(transform_path)
            self.assertIsNotNone(transform_func)
            
            # Test the loaded function
            test_data = [{"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": {}}]
            transformed = transform_func(test_data)
            self.assertTrue(transformed[0]["metadata"]["transformed"])
        finally:
            os.unlink(transform_path)
    
    def test_load_transform_function_missing_transform(self):
        """Test loading a module without a transform function."""
        module_code = """
# This module doesn't have a transform function
def some_other_function():
    pass
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(module_code)
            module_path = f.name
        
        try:
            transform_func = load_transform_function(module_path)
            self.assertIsNone(transform_func)
        finally:
            os.unlink(module_path)
    
    @patch('vectordb_migration.cli.migrate.load_config')
    @patch('vectordb_migration.cli.migrate.load_transform_function')
    @patch('vectordb_migration.cli.migrate.DBMigrator')
    def test_run_migration(self, mock_migrator_class, mock_load_transform, mock_load_config):
        """Test running a migration through the CLI interface."""
        # Mock the configuration
        mock_config = {
            "source": {"type": "pgvector"},
            "target": {"type": "qdrant"}
        }
        mock_load_config.return_value = mock_config
        
        # Mock the transform function
        mock_transform = MagicMock()
        mock_load_transform.return_value = mock_transform
        
        # Mock the migrator
        mock_migrator = MagicMock()
        mock_migrator.migrate.return_value = True
        mock_migrator_class.return_value = mock_migrator
        
        # Run the migration
        result = run_migration("config.json", "transform.py", verbose=True)
        
        # Verify the result
        self.assertTrue(result)
        mock_load_config.assert_called_once_with("config.json")
        mock_load_transform.assert_called_once_with("transform.py")
        mock_migrator.migrate.assert_called_once_with(
            source_params=mock_config["source"],
            target_params=mock_config["target"],
            transform_func=mock_transform
        )


if __name__ == "__main__":
    unittest.main()