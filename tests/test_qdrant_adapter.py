"""
Tests for the Qdrant vector database adapter.

This module tests the QdrantAdapter implementation using mocks.
"""

import unittest
from unittest.mock import MagicMock, patch

from vectordb_migration.adapters.qdrant import QdrantAdapter


class TestQdrantAdapter(unittest.TestCase):
    """Tests for the QdrantAdapter."""
    
    @patch('vectordb_migration.adapters.qdrant.QdrantClient')
    def test_connect(self, mock_qdrant_client):
        """Test connecting to Qdrant."""
        # Setup mock
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        
        # Create adapter and connect
        adapter = QdrantAdapter()
        result = adapter.connect(
            host="test-host",
            port=6333,
            api_key="test-key",
            https=True
        )
        
        # Verify
        self.assertTrue(result)
        self.assertEqual(adapter.client, mock_client)
        mock_qdrant_client.assert_called_once_with(
            host="test-host",
            port=6333,
            api_key="test-key",
            https=True,
            grpc_port=None,
            prefer_grpc=False,
            timeout=None
        )
    
    @patch('vectordb_migration.adapters.qdrant.QdrantClient')
    def test_connect_failure(self, mock_qdrant_client):
        """Test handling connection failures."""
        # Setup mock to raise exception
        mock_qdrant_client.side_effect = Exception("Connection failed")
        
        # Create adapter and try to connect
        adapter = QdrantAdapter()
        result = adapter.connect(host="test-host")
        
        # Verify
        self.assertFalse(result)
        self.assertIsNone(adapter.client)
    
    def test_disconnect(self):
        """Test disconnecting from Qdrant."""
        # Setup
        adapter = QdrantAdapter()
        adapter.client = MagicMock()
        
        # Disconnect
        adapter.disconnect()
        
        # Verify
        self.assertIsNone(adapter.client)
    
    def test_extract_data(self):
        """Test extracting data from Qdrant."""
        # Setup
        adapter = QdrantAdapter()
        adapter.client = MagicMock()
        
        # Create mocked points
        class MockPoint:
            def __init__(self, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload
        
        mock_points = [
            MockPoint(1, [0.1, 0.2, 0.3], {"name": "Item 1", "category": "test"}),
            MockPoint(2, [0.4, 0.5, 0.6], {"name": "Item 2", "category": "test"})
        ]
        
        # Mock scroll response
        adapter.client.scroll.return_value = (mock_points, None)
        adapter.client.get_collection.return_value = MagicMock(vectors_count=2)
        
        # Extract data
        result = adapter.extract_data(
            collection_name="test_collection",
            limit=100,
            filter={"must": [{"key": "category", "match": {"value": "test"}}]}
        )
        
        # Verify
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[0]["vector"], [0.1, 0.2, 0.3])
        self.assertEqual(result[0]["metadata"]["name"], "Item 1")
        
        # Check that scroll was called with correct parameters
        adapter.client.scroll.assert_called_once()
        call_args = adapter.client.scroll.call_args[1]
        self.assertEqual(call_args["collection_name"], "test_collection")
        self.assertEqual(call_args["limit"], 100)
        self.assertEqual(call_args["with_payload"], True)
        self.assertEqual(call_args["with_vectors"], True)
        self.assertEqual(call_args["filter"], {"must": [{"key": "category", "match": {"value": "test"}}]})
    
    def test_extract_data_collection_not_found(self):
        """Test extracting data from a non-existent collection."""
        # Setup
        adapter = QdrantAdapter()
        adapter.client = MagicMock()
        
        # Mock get_collection to raise an exception
        adapter.client.get_collection.side_effect = Exception("Collection not found")
        
        # Extract data
        result = adapter.extract_data(collection_name="non_existent_collection")
        
        # Verify empty result
        self.assertEqual(result, [])
    
    @patch('vectordb_migration.adapters.qdrant.models')
    def test_load_data(self, mock_models):
        """Test loading data to Qdrant."""
        # Setup
        adapter = QdrantAdapter()
        adapter.client = MagicMock()
        
        # Mock models
        mock_models.Distance.COSINE = "Cosine"
        mock_models.VectorParams = MagicMock()
        mock_models.PointStruct = MagicMock(side_effect=lambda id, vector, payload: {
            "id": id, "vector": vector, "payload": payload
        })
        
        # Test data
        test_data = [
            {"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": {"name": "Item 1"}},
            {"id": 2, "vector": [0.4, 0.5, 0.6], "metadata": {"name": "Item 2"}}
        ]
        
        # Load data
        result = adapter.load_data(
            test_data,
            collection_name="test_collection",
            recreate_collection=False,
            batch_size=10
        )
        
        # Verify
        self.assertTrue(result)
        adapter.client.upsert.assert_called_once()
        upsert_args = adapter.client.upsert.call_args[1]
        self.assertEqual(upsert_args["collection_name"], "test_collection")
        self.assertEqual(len(upsert_args["points"]), 2)
    
    @patch('vectordb_migration.adapters.qdrant.models')
    def test_load_data_recreate_collection(self, mock_models):
        """Test loading data with collection recreation."""
        # Setup
        adapter = QdrantAdapter()
        adapter.client = MagicMock()
        
        # Mock distance enum
        mock_models.Distance.COSINE = "Cosine"
        mock_models.VectorParams = MagicMock()
        mock_models.PointStruct = MagicMock(side_effect=lambda id, vector, payload: {
            "id": id, "vector": vector, "payload": payload
        })
        
        # Test data
        test_data = [
            {"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": {"name": "Item 1"}},
            {"id": 2, "vector": [0.4, 0.5, 0.6], "metadata": {"name": "Item 2"}}
        ]
        
        # First, test when collection exists and needs recreation
        adapter.client.get_collection.return_value = MagicMock()
        
        # Load data with recreation
        result = adapter.load_data(
            test_data,
            collection_name="test_collection",
            recreate_collection=True,
            distance="cosine"
        )
        
        # Verify
        self.assertTrue(result)
        adapter.client.delete_collection.assert_called_once_with(collection_name="test_collection")
        adapter.client.create_collection.assert_called_once()
        
        # Now test when collection doesn't exist
        adapter.client.reset_mock()
        adapter.client.get_collection.side_effect = Exception("Collection not found")
        
        # Load data
        result = adapter.load_data(
            test_data,
            collection_name="new_collection",
            distance="dot"
        )
        
        # Verify
        self.assertTrue(result)
        adapter.client.delete_collection.assert_not_called()  # Shouldn't be called for non-existent collection
        adapter.client.create_collection.assert_called_once()
    
    def test_get_schema_info(self):
        """Test getting schema info from Qdrant."""
        # Setup
        adapter = QdrantAdapter()
        adapter.client = MagicMock()
        
        # Mock collection info
        collection_config = MagicMock()
        collection_config.config.params.vectors.size = 384
        collection_config.config.params.vectors.distance = "Cosine"
        collection_config.vectors_count = 1000
        adapter.client.get_collection.return_value = collection_config
        
        # Mock point sample
        class MockPoint:
            def __init__(self, id, payload):
                self.id = id
                self.payload = payload
        
        sample_point = MockPoint(1, {"name": "Sample", "category": "test"})
        adapter.client.scroll.return_value = ([sample_point], None)
        
        # Get schema info
        result = adapter.get_schema_info("test_collection")
        
        # Verify
        self.assertEqual(result["collection_name"], "test_collection")
        self.assertIn("vector_config", result)
        self.assertEqual(result["points_count"], 1000)
        self.assertEqual(result["payload_sample"], {"name": "Sample", "category": "test"})


if __name__ == "__main__":
    unittest.main()