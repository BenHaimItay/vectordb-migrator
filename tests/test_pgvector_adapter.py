"""
Tests for the PostgreSQL with pgvector adapter.

This module tests the PgVectorAdapter implementation using mocks.
"""

import unittest
from unittest.mock import MagicMock, patch

from vectordb_migration.adapters.pgvector import PgVectorAdapter


class TestPgVectorAdapter(unittest.TestCase):
    """Tests for the PgVectorAdapter."""
    
    @patch('vectordb_migration.adapters.pgvector.psycopg2')
    def test_connect(self, mock_psycopg2):
        """Test connecting to a PostgreSQL database."""
        # Setup mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg2.connect.return_value = mock_conn
        
        # Create adapter and connect
        adapter = PgVectorAdapter()
        result = adapter.connect(
            host="test-host",
            dbname="test-db",
            user="test-user",
            password="test-pass",
            port=5432
        )
        
        # Verify
        self.assertTrue(result)
        self.assertEqual(adapter.conn, mock_conn)
        self.assertEqual(adapter.cursor, mock_cursor)
        mock_psycopg2.connect.assert_called_once_with(
            host="test-host",
            dbname="test-db",
            user="test-user",
            password="test-pass",
            port=5432
        )
    
    @patch('vectordb_migration.adapters.pgvector.psycopg2')
    def test_connect_failure(self, mock_psycopg2):
        """Test handling connection failures."""
        # Setup mock to raise exception
        mock_psycopg2.connect.side_effect = Exception("Connection failed")
        
        # Create adapter and try to connect
        adapter = PgVectorAdapter()
        result = adapter.connect(host="test-host")
        
        # Verify
        self.assertFalse(result)
        self.assertIsNone(adapter.conn)
        self.assertIsNone(adapter.cursor)
    
    def test_disconnect(self):
        """Test disconnecting from PostgreSQL."""
        # Setup
        adapter = PgVectorAdapter()
        adapter.conn = MagicMock()
        adapter.cursor = MagicMock()
        
        # Disconnect
        adapter.disconnect()
        
        # Verify
        adapter.cursor.close.assert_called_once()
        adapter.conn.close.assert_called_once()
        self.assertIsNone(adapter.conn)
        self.assertIsNone(adapter.cursor)
    
    def test_extract_data(self):
        """Test extracting data from PostgreSQL."""
        # Setup
        adapter = PgVectorAdapter()
        adapter.conn = MagicMock()
        adapter.cursor = MagicMock()
        
        # Mock query result
        mock_rows = [
            (1, [0.1, 0.2, 0.3], "Item 1", "Description 1"),
            (2, [0.4, 0.5, 0.6], "Item 2", "Description 2")
        ]
        adapter.cursor.fetchall.return_value = mock_rows
        
        # Extract data
        result = adapter.extract_data(
            table_name="test_table",
            id_column="id",
            vector_column="embedding",
            metadata_columns=["name", "description"],
            filter_condition="category = 'test'"
        )
        
        # Verify SQL query construction
        adapter.cursor.execute.assert_called_once()
        query_arg = adapter.cursor.execute.call_args[0][0]
        self.assertIn("SELECT id, embedding, name, description FROM test_table", query_arg)
        self.assertIn("WHERE category = 'test'", query_arg)
        
        # Verify result format
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[0]["vector"], [0.1, 0.2, 0.3])
        self.assertEqual(result[0]["metadata"]["name"], "Item 1")
        self.assertEqual(result[0]["metadata"]["description"], "Description 1")
    
    def test_load_data(self):
        """Test loading data to PostgreSQL."""
        # Setup
        adapter = PgVectorAdapter()
        adapter.conn = MagicMock()
        adapter.cursor = MagicMock()
        
        # Test data
        test_data = [
            {"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": {"name": "Item 1", "category": "test"}},
            {"id": 2, "vector": [0.4, 0.5, 0.6], "metadata": {"name": "Item 2", "category": "test"}}
        ]
        
        # Load data
        result = adapter.load_data(
            test_data,
            table_name="test_table",
            id_column="id",
            vector_column="embedding",
            batch_size=2
        )
        
        # Verify
        self.assertTrue(result)
        adapter.cursor.executemany.assert_called_once()
        adapter.conn.commit.assert_called_once()
    
    def test_load_data_with_table_creation(self):
        """Test loading data with table creation."""
        # Setup
        adapter = PgVectorAdapter()
        adapter.conn = MagicMock()
        adapter.cursor = MagicMock()
        
        # Test data
        test_data = [
            {"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": {"name": "Item 1", "category": "test"}},
            {"id": 2, "vector": [0.4, 0.5, 0.6], "metadata": {"name": "Item 2", "category": "test"}}
        ]
        
        # Load data with table recreation
        result = adapter.load_data(
            test_data,
            table_name="test_table",
            id_column="id",
            vector_column="embedding",
            recreate_table=True
        )
        
        # Verify
        self.assertTrue(result)
        # Check for table creation queries
        create_calls = [call[0][0] for call in adapter.cursor.execute.call_args_list 
                     if "CREATE TABLE" in call[0][0]]
        self.assertTrue(any(create_calls))
        # Check for vector extension creation
        extension_calls = [call[0][0] for call in adapter.cursor.execute.call_args_list 
                        if "CREATE EXTENSION" in call[0][0]]
        self.assertTrue(any(extension_calls))


if __name__ == "__main__":
    unittest.main()