import unittest
from unittest.mock import patch, MagicMock, call
import logging

# Assuming vectordb_migration is in PYTHONPATH or installed
from vectordb_migration.adapters.milvus import MilvusAdapter
from pymilvus import DataType # For schema creation in tests

# Configure logging to suppress INFO messages from the adapter during tests,
# but show warnings and errors if they occur.
logging.basicConfig(level=logging.WARNING)


class TestMilvusAdapter(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.adapter = MilvusAdapter(
            host="localhost",
            port="19530",
            alias="test_alias" 
            # Add other default connection params if your adapter's __init__ expects them
        )

    # Test __init__
    def test_adapter_initialization(self):
        self.assertIsNone(self.adapter.client)
        self.assertEqual(self.adapter.connection_params["host"], "localhost")
        self.assertEqual(self.adapter.connection_params["port"], "19530")
        self.assertEqual(self.adapter.connection_params["alias"], "test_alias")
        
        adapter_custom = MilvusAdapter(uri="http://anotherhost:19530", token="some_token", alias="custom_alias")
        self.assertEqual(adapter_custom.connection_params["uri"], "http://anotherhost:19530")
        self.assertEqual(adapter_custom.connection_params["token"], "some_token")
        self.assertEqual(adapter_custom.connection_params["alias"], "custom_alias")

    @patch('vectordb_migration.adapters.milvus.connections')
    def test_connect_successful(self, mock_connections):
        """Test successful connection to Milvus."""
        self.adapter.connect(host="testhost", port="12345", alias="conn_alias")
        mock_connections.connect.assert_called_once_with(
            host="testhost", port="12345", alias="conn_alias"
        )
        self.assertEqual(self.adapter.client, "connected") # As per current adapter logic

    @patch('vectordb_migration.adapters.milvus.connections')
    def test_connect_uses_init_params(self, mock_connections):
        """Test connect uses parameters from __init__ if not overridden."""
        self.adapter.connect() # Uses params from setUp
        mock_connections.connect.assert_called_once_with(
            host="localhost", port="19530", alias="test_alias" # alias is popped and used
        )
        self.assertEqual(self.adapter.client, "connected")

    @patch('vectordb_migration.adapters.milvus.connections')
    def test_connect_override_init_params(self, mock_connections):
        """Test connect overrides __init__ parameters."""
        self.adapter.connect(host="newhost", port="54321", alias="override_alias", user="testuser")
        mock_connections.connect.assert_called_once_with(
            host="newhost", port="54321", alias="override_alias", user="testuser"
        )
        self.assertEqual(self.adapter.client, "connected")


    @patch('vectordb_migration.adapters.milvus.connections')
    def test_connect_failure(self, mock_connections):
        """Test connection failure."""
        mock_connections.connect.side_effect = Exception("Connection refused")
        with self.assertRaises(Exception) as context:
            self.adapter.connect(host="badhost", port="0000")
        self.assertTrue("Connection refused" in str(context.exception))
        mock_connections.connect.assert_called_once_with(host="badhost", port="0000", alias="test_alias") # Default alias if not in kwargs
        self.assertIsNone(self.adapter.client)

    @patch('vectordb_migration.adapters.milvus.connections')
    def test_disconnect_successful(self, mock_connections):
        """Test successful disconnection."""
        # First, simulate a connection
        self.adapter.client = "connected" 
        self.adapter.connection_params = {"alias": "disconnect_alias"} # Set alias used for connection
        
        self.adapter.disconnect()
        
        mock_connections.disconnect.assert_called_once_with("disconnect_alias")
        self.assertIsNone(self.adapter.client)

    @patch('vectordb_migration.adapters.milvus.connections')
    def test_disconnect_uses_default_alias(self, mock_connections):
        """Test disconnect uses default alias if not set during connection."""
        self.adapter.client = "connected"
        # connection_params might not have 'alias' if connect was called with it directly
        # and it was popped. setUp sets it.
        self.adapter.connection_params = {"host":"localhost"} # alias not present
        
        self.adapter.disconnect()
        mock_connections.disconnect.assert_called_once_with("default") # As per adapter logic


    @patch('vectordb_migration.adapters.milvus.connections')
    def test_disconnect_failure(self, mock_connections):
        """Test disconnection failure."""
        self.adapter.client = "connected"
        self.adapter.connection_params = {"alias": "fail_alias"}
        mock_connections.disconnect.side_effect = Exception("Disconnection error")
        
        # Disconnect in adapter currently logs error but doesn't raise
        self.adapter.disconnect() 
        
        mock_connections.disconnect.assert_called_once_with("fail_alias")
        # Client is set to None even if disconnect fails, as per current adapter logic
        self.assertIsNone(self.adapter.client) 

    @patch('vectordb_migration.adapters.milvus.connections')
    def test_disconnect_when_not_connected(self, mock_connections):
        """Test disconnecting when already disconnected or not connected."""
        self.adapter.client = None # Ensure not connected
        self.adapter.disconnect()
        mock_connections.disconnect.assert_called_once_with("test_alias") # From setUp
        self.assertIsNone(self.adapter.client) # Should remain None

    def test_get_schema_info_not_connected(self):
        """Test get_schema_info when not connected."""
        self.adapter.client = None
        with self.assertRaisesRegex(ConnectionError, "Not connected to Milvus. Call connect() first."):
            self.adapter.get_schema_info("any_collection")

    @patch('vectordb_migration.adapters.milvus.utility')
    @patch('vectordb_migration.adapters.milvus.Collection')
    def test_get_schema_info_collection_does_not_exist(self, MockCollection, mock_utility):
        """Test get_schema_info for a non-existent collection."""
        self.adapter.client = "connected" # Simulate connected state
        mock_utility.has_collection.return_value = False
        
        schema_info = self.adapter.get_schema_info("non_existent_collection")
        
        mock_utility.has_collection.assert_called_once_with("non_existent_collection")
        self.assertIsNone(schema_info)
        MockCollection.assert_not_called() # Collection object should not be created

    @patch('vectordb_migration.adapters.milvus.utility')
    @patch('vectordb_migration.adapters.milvus.Collection')
    def test_get_schema_info_successful(self, MockCollection, mock_utility):
        """Test successful retrieval of schema information."""
        self.adapter.client = "connected"
        collection_name = "test_collection"

        # Mocking pymilvus objects
        mock_utility.has_collection.return_value = True
        mock_utility.list_aliases.return_value = ["alias1"]

        mock_collection_instance = MockCollection.return_value
        mock_collection_instance.name = collection_name
        mock_collection_instance.description = "Test Collection Description"
        mock_collection_instance.num_entities = 1000
        mock_collection_instance.consistency_level = "Strong" # Or an enum/int if that's what Milvus uses
        mock_collection_instance.properties = {"property1": "value1"}

        mock_schema = MagicMock()
        mock_collection_instance.schema = mock_schema
        mock_schema.description = "Schema Description"
        mock_schema.auto_id = False
        
        # Mock fields
        mock_pk_field = MagicMock()
        mock_pk_field.name = "id_field"
        mock_pk_field.dtype = DataType.INT64 
        mock_pk_field.is_primary = True
        mock_pk_field.description = "Primary Key Field"
        mock_pk_field.params = {}

        mock_vector_field = MagicMock()
        mock_vector_field.name = "vector_field"
        mock_vector_field.dtype = DataType.FLOAT_VECTOR
        mock_vector_field.is_primary = False
        mock_vector_field.description = "Vector Field"
        mock_vector_field.params = {"dim": 128}
        
        mock_scalar_field = MagicMock()
        mock_scalar_field.name = "scalar_field"
        mock_scalar_field.dtype = DataType.VARCHAR
        mock_scalar_field.is_primary = False
        mock_scalar_field.description = "Scalar Field"
        mock_scalar_field.params = {"max_length": 255}

        mock_schema.fields = [mock_pk_field, mock_vector_field, mock_scalar_field]
        mock_schema.primary_field = mock_pk_field

        expected_schema_info = {
            "name": collection_name,
            "description": "Test Collection Description",
            "num_entities": 1000,
            "consistency_level": "Strong",
            "aliases": ["alias1"],
            "properties": {"property1": "value1"},
            "schema": {
                "fields": [
                    {"name": "id_field", "type": "INT64", "is_primary": True, "description": "Primary Key Field", "params": {}},
                    {"name": "vector_field", "type": "FLOAT_VECTOR", "is_primary": False, "description": "Vector Field", "params": {"dim": 128}},
                    {"name": "scalar_field", "type": "VARCHAR", "is_primary": False, "description": "Scalar Field", "params": {"max_length": 255}},
                ],
                "description": "Schema Description",
                "auto_id": False,
                "primary_field": "id_field",
            }
        }

        schema_info = self.adapter.get_schema_info(collection_name)

        mock_utility.has_collection.assert_called_once_with(collection_name)
        MockCollection.assert_called_once_with(collection_name)
        mock_utility.list_aliases.assert_called_once_with(collection_name)
        self.assertEqual(schema_info, expected_schema_info)

    @patch('vectordb_migration.adapters.milvus.utility')
    @patch('vectordb_migration.adapters.milvus.Collection', side_effect=Exception("Schema retrieval error"))
    def test_get_schema_info_exception_during_retrieval(self, MockCollection, mock_utility):
        """Test exception handling during schema retrieval."""
        self.adapter.client = "connected"
        collection_name = "error_collection"
        mock_utility.has_collection.return_value = True # Collection exists

        # The side_effect on MockCollection will cause an error when Collection(collection_name) is called
        schema_info = self.adapter.get_schema_info(collection_name)
        
        self.assertIsNone(schema_info) # Adapter should catch exception and return None
        mock_utility.has_collection.assert_called_once_with(collection_name)
        MockCollection.assert_called_once_with(collection_name)

    # --- Tests for extract_data ---

    def test_extract_data_not_connected(self):
        """Test extract_data when not connected."""
        self.adapter.client = None
        with self.assertRaisesRegex(ConnectionError, "Not connected to Milvus. Call connect() first."):
            self.adapter.extract_data("any_collection")

    @patch('vectordb_migration.adapters.milvus.utility')
    def test_extract_data_collection_does_not_exist(self, mock_utility):
        """Test extract_data for a non-existent collection."""
        self.adapter.client = "connected"
        mock_utility.has_collection.return_value = False
        
        data = self.adapter.extract_data("non_existent_collection")
        
        mock_utility.has_collection.assert_called_once_with("non_existent_collection")
        self.assertEqual(data, [])

    @patch('vectordb_migration.adapters.milvus.Collection')
    @patch('vectordb_migration.adapters.milvus.utility')
    @patch.object(MilvusAdapter, 'get_schema_info') # Mocking the adapter's own method
    def test_extract_data_schema_retrieval_fails(self, mock_get_schema_info, mock_utility, MockCollection):
        """Test extract_data when schema retrieval fails."""
        self.adapter.client = "connected"
        collection_name = "test_coll"
        mock_utility.has_collection.return_value = True
        mock_get_schema_info.return_value = None # Simulate schema failure

        data = self.adapter.extract_data(collection_name)

        mock_get_schema_info.assert_called_once_with(collection_name)
        self.assertEqual(data, [])
        MockCollection.return_value.query.assert_not_called()

    @patch('vectordb_migration.adapters.milvus.Collection')
    @patch('vectordb_migration.adapters.milvus.utility')
    @patch.object(MilvusAdapter, 'get_schema_info')
    def test_extract_data_no_primary_key_in_schema(self, mock_get_schema_info, mock_utility, MockCollection):
        """Test extract_data when schema has no primary key."""
        self.adapter.client = "connected"
        collection_name = "test_coll_no_pk"
        mock_utility.has_collection.return_value = True
        mock_get_schema_info.return_value = {
            "schema": {
                "fields": [{"name": "vector_field", "type": "FLOAT_VECTOR", "is_primary": False}],
                "primary_field": None # No PK
            }
        }
        data = self.adapter.extract_data(collection_name)
        self.assertEqual(data, [])
        MockCollection.return_value.query.assert_not_called()


    @patch('vectordb_migration.adapters.milvus.Collection')
    @patch('vectordb_migration.adapters.milvus.utility')
    @patch.object(MilvusAdapter, 'get_schema_info')
    def test_extract_data_successful(self, mock_get_schema_info, mock_utility, MockCollection):
        """Test successful data extraction."""
        self.adapter.client = "connected"
        collection_name = "extract_success_collection"

        mock_utility.has_collection.return_value = True
        mock_schema_info = {
            "schema": {
                "primary_field": "pk_field",
                "fields": [
                    {"name": "pk_field", "type": "INT64", "is_primary": True},
                    {"name": "vec_field", "type": "FLOAT_VECTOR", "is_primary": False},
                    {"name": "meta_field1", "type": "VARCHAR", "is_primary": False},
                    {"name": "meta_field2", "type": "INT32", "is_primary": False},
                ]
            }
        }
        mock_get_schema_info.return_value = mock_schema_info
        
        mock_collection_instance = MockCollection.return_value
        # Simulate Milvus query results (list of dicts)
        query_results = [
            {"pk_field": 1, "vec_field": [0.1, 0.2], "meta_field1": "value1", "meta_field2": 100},
            {"pk_field": 2, "vec_field": [0.3, 0.4], "meta_field1": "value2", "meta_field2": 200},
        ]
        mock_collection_instance.query.return_value = query_results

        expected_data = [
            {"id": 1, "vector": [0.1, 0.2], "metadata": {"meta_field1": "value1", "meta_field2": 100}},
            {"id": 2, "vector": [0.3, 0.4], "metadata": {"meta_field1": "value2", "meta_field2": 200}},
        ]

        data = self.adapter.extract_data(collection_name, limit=10, offset=0, filter_expr="meta_field2 > 50")

        mock_utility.has_collection.assert_called_once_with(collection_name)
        mock_get_schema_info.assert_called_once_with(collection_name)
        mock_collection_instance.load.assert_called_once()
        mock_collection_instance.query.assert_called_once_with(
            expr="meta_field2 > 50",
            output_fields=['pk_field', 'vec_field', 'meta_field1', 'meta_field2'],
            limit=10,
            offset=0
        )
        self.assertEqual(data, expected_data)

    @patch('vectordb_migration.adapters.milvus.Collection')
    @patch('vectordb_migration.adapters.milvus.utility')
    @patch.object(MilvusAdapter, 'get_schema_info')
    def test_extract_data_no_vector_field_in_schema(self, mock_get_schema_info, mock_utility, MockCollection):
        """Test data extraction when schema has no vector field."""
        self.adapter.client = "connected"
        collection_name = "no_vector_collection"
        mock_utility.has_collection.return_value = True
        mock_schema_info = {
            "schema": {
                "primary_field": "id",
                "fields": [
                    {"name": "id", "type": "INT64", "is_primary": True},
                    {"name": "text_field", "type": "VARCHAR", "is_primary": False},
                ]
            }
        }
        mock_get_schema_info.return_value = mock_schema_info
        mock_collection_instance = MockCollection.return_value
        query_results = [{"id": 10, "text_field": "some text"}]
        mock_collection_instance.query.return_value = query_results

        expected_data = [{"id": 10, "vector": None, "metadata": {"text_field": "some text"}}]
        data = self.adapter.extract_data(collection_name)

        mock_collection_instance.query.assert_called_once_with(
            output_fields=['id', 'text_field'], # No vector field
            limit=100, # Default limit
            offset=0 # Default offset
            # No expr as filter_expr is None
        )
        self.assertEqual(data, expected_data)

    @patch('vectordb_migration.adapters.milvus.Collection')
    @patch('vectordb_migration.adapters.milvus.utility')
    @patch.object(MilvusAdapter, 'get_schema_info')
    def test_extract_data_with_pagination_and_no_filter(self, mock_get_schema_info, mock_utility, MockCollection):
        """Test data extraction with pagination and no filter expression."""
        self.adapter.client = "connected"
        collection_name = "paginated_collection"
        mock_utility.has_collection.return_value = True
        mock_schema_info = { # Same schema as successful test
            "schema": {
                "primary_field": "pk", "fields": [
                    {"name": "pk", "type": "INT64", "is_primary": True},
                    {"name": "v", "type": "FLOAT_VECTOR", "is_primary": False},
                    {"name": "m", "type": "VARCHAR", "is_primary": False}]}}
        mock_get_schema_info.return_value = mock_schema_info
        
        mock_collection_instance = MockCollection.return_value
        mock_collection_instance.query.return_value = [] # Actual data doesn't matter for this call check

        self.adapter.extract_data(collection_name, limit=5, offset=10) # No filter_expr

        mock_collection_instance.query.assert_called_once_with(
            output_fields=['pk', 'v', 'm'],
            limit=5,
            offset=10
            # expr should not be in kwargs if filter_expr is None
        )
        # Verify 'expr' is not in the actual call's kwargs
        args, kwargs = mock_collection_instance.query.call_args
        self.assertNotIn("expr", kwargs)


    @patch('vectordb_migration.adapters.milvus.Collection')
    @patch('vectordb_migration.adapters.milvus.utility')
    @patch.object(MilvusAdapter, 'get_schema_info')
    def test_extract_data_query_exception(self, mock_get_schema_info, mock_utility, MockCollection):
        """Test exception handling during Milvus query."""
        self.adapter.client = "connected"
        collection_name = "query_exception_collection"
        mock_utility.has_collection.return_value = True
        mock_schema_info = { # Schema that would normally work
             "schema": {"primary_field": "id", "fields": [{"name": "id", "type": "INT64", "is_primary": True}, {"name": "vec", "type": "FLOAT_VECTOR"}]}}
        mock_get_schema_info.return_value = mock_schema_info

        mock_collection_instance = MockCollection.return_value
        mock_collection_instance.query.side_effect = Exception("Milvus query failed")

        data = self.adapter.extract_data(collection_name)
        
        self.assertEqual(data, []) # Should return empty list on error
        mock_collection_instance.query.assert_called_once()

    # --- Tests for load_data ---

    def test_load_data_not_connected(self):
        """Test load_data when not connected."""
        self.adapter.client = None
        with self.assertRaisesRegex(ConnectionError, "Not connected to Milvus. Call connect() first."):
            self.adapter.load_data("any_collection", [{"id": 1, "vector": [0.1]}])

    def test_load_data_no_data_provided(self):
        """Test load_data with no data."""
        self.adapter.client = "connected"
        result = self.adapter.load_data("some_collection", [])
        self.assertEqual(result, {"insert_count": 0, "errors": [], "success_count":0, "failure_count":0})

    @patch('vectordb_migration.adapters.milvus.utility')
    def test_load_data_collection_does_not_exist(self, mock_utility):
        """Test load_data to a non-existent collection."""
        self.adapter.client = "connected"
        mock_utility.has_collection.return_value = False
        with self.assertRaisesRegex(ValueError, "Collection non_existent_collection does not exist."):
            self.adapter.load_data("non_existent_collection", [{"id": 1, "vector": [0.1]}])
        mock_utility.has_collection.assert_called_once_with("non_existent_collection")

    @patch('vectordb_migration.adapters.milvus.utility')
    @patch.object(MilvusAdapter, 'get_schema_info')
    def test_load_data_schema_retrieval_fails(self, mock_get_schema_info, mock_utility):
        """Test load_data when schema retrieval fails."""
        self.adapter.client = "connected"
        collection_name = "test_coll_load_schema_fail"
        mock_utility.has_collection.return_value = True
        mock_get_schema_info.return_value = None # Simulate schema failure

        with self.assertRaisesRegex(ValueError, f"Could not retrieve schema for {collection_name}."):
            self.adapter.load_data(collection_name, [{"id": 1, "vector": [0.1]}])
        
        mock_get_schema_info.assert_called_once_with(collection_name)

    @patch('vectordb_migration.adapters.milvus.utility')
    @patch.object(MilvusAdapter, 'get_schema_info')
    def test_load_data_no_primary_key_in_schema(self, mock_get_schema_info, mock_utility):
        """Test load_data when schema has no primary key."""
        self.adapter.client = "connected"
        collection_name = "load_coll_no_pk"
        mock_utility.has_collection.return_value = True
        mock_get_schema_info.return_value = {
            "schema": { "primary_field": None, "fields": [{"name": "v", "type": "FLOAT_VECTOR"}]}
        }
        with self.assertRaisesRegex(ValueError, f"Primary key not found in schema for {collection_name}"):
            self.adapter.load_data(collection_name, [{"id": 1, "vector": [0.1]}])


    @patch('vectordb_migration.adapters.milvus.Collection')
    @patch('vectordb_migration.adapters.milvus.utility')
    @patch.object(MilvusAdapter, 'get_schema_info')
    def test_load_data_successful(self, mock_get_schema_info, mock_utility, MockCollection):
        """Test successful data loading."""
        self.adapter.client = "connected"
        collection_name = "load_success_coll"

        mock_utility.has_collection.return_value = True
        mock_schema = {
            "schema": {
                "primary_field": "pk",
                "fields": [
                    {"name": "pk", "type": "INT64", "is_primary": True},
                    {"name": "vec", "type": "FLOAT_VECTOR", "is_primary": False, "params": {"dim": 2}},
                    {"name": "meta", "type": "VARCHAR", "is_primary": False, "params": {"max_length": 100}},
                ]
            }
        }
        mock_get_schema_info.return_value = mock_schema
        
        mock_collection_instance = MockCollection.return_value
        mock_insert_result = MagicMock()
        mock_insert_result.insert_count = 2
        mock_insert_result.primary_keys = [1, 2]
        mock_collection_instance.insert.return_value = mock_insert_result

        data_to_load = [
            {"id": 1, "vector": [0.1, 0.2], "metadata": {"meta": "item1"}},
            {"id": 2, "vector": [0.3, 0.4], "metadata": {"meta": "item2"}},
        ]
        
        # Expected data format for Milvus: list of columns
        # Order based on mock_schema.schema.fields
        expected_milvus_data = [
            [1, 2],             # pk
            [[0.1, 0.2], [0.3, 0.4]], # vec
            ["item1", "item2"], # meta
        ]

        result = self.adapter.load_data(collection_name, data_to_load)

        mock_utility.has_collection.assert_called_once_with(collection_name)
        MockCollection.assert_called_once_with(collection_name)
        mock_get_schema_info.assert_called_once_with(collection_name)
        mock_collection_instance.insert.assert_called_once_with(expected_milvus_data)
        # mock_collection_instance.flush.assert_called_once() # If flush is unconditionally called

        self.assertEqual(result["insert_count"], 2)
        self.assertEqual(result["total_processed_count"], 2)
        self.assertEqual(result["primary_keys_inserted"], [1,2])
        self.assertEqual(len(result["errors"]), 0)

    @patch('vectordb_migration.adapters.milvus.Collection')
    @patch('vectordb_migration.adapters.milvus.utility')
    @patch.object(MilvusAdapter, 'get_schema_info')
    def test_load_data_missing_vector_when_schema_has_it(self, mock_get_schema_info, mock_utility, MockCollection):
        """Test loading data where a record is missing a vector field defined in schema."""
        self.adapter.client = "connected"
        collection_name = "load_missing_vec_coll"
        mock_utility.has_collection.return_value = True
        mock_schema = {
            "schema": {
                "primary_field": "id_col",
                "fields": [
                    {"name": "id_col", "type": "INT64", "is_primary": True},
                    {"name": "vector_col", "type": "FLOAT_VECTOR", "is_primary": False, "params": {"dim": 2}},
                ]
            }
        }
        mock_get_schema_info.return_value = mock_schema
        
        data_to_load = [
            {"id": 1}, # Missing 'vector'
        ]
        # The adapter's load_data currently appends None if vector is missing but field exists.
        # This might or might not be what Milvus expects depending on schema strictness.
        # Here we test the adapter's behavior of appending None.
        
        mock_collection_instance = MockCollection.return_value
        mock_insert_result = MagicMock()
        mock_insert_result.insert_count = 1 
        mock_insert_result.primary_keys = [1]
        mock_collection_instance.insert.return_value = mock_insert_result

        # Expecting [id_values], [vector_values (with None for missing)]
        expected_milvus_data = [[1], [None]] 

        result = self.adapter.load_data(collection_name, data_to_load)

        mock_collection_instance.insert.assert_called_once_with(expected_milvus_data)
        self.assertEqual(result["insert_count"], 1) # Assuming Milvus accepts None here or mock allows it
        self.assertEqual(result["total_processed_count"], 1)


    @patch('vectordb_migration.adapters.milvus.Collection')
    @patch('vectordb_migration.adapters.milvus.utility')
    @patch.object(MilvusAdapter, 'get_schema_info')
    def test_load_data_record_missing_id(self, mock_get_schema_info, mock_utility, MockCollection):
        """Test loading data where a record is missing the 'id' field."""
        self.adapter.client = "connected"
        collection_name = "load_missing_id_coll"
        mock_utility.has_collection.return_value = True
        mock_schema = { "schema": { "primary_field": "pk", "fields": [{"name": "pk", "type": "INT64", "is_primary": True}]}}
        mock_get_schema_info.return_value = mock_schema

        data_to_load = [
            {"vector": [0.5, 0.6], "metadata": {"info": "no id"}}, # Missing 'id'
            {"id": 2, "vector": [0.7, 0.8]},
        ]
        
        mock_collection_instance = MockCollection.return_value
        mock_insert_result = MagicMock()
        mock_insert_result.insert_count = 1 # Only the valid record
        mock_insert_result.primary_keys = [2]
        mock_collection_instance.insert.return_value = mock_insert_result
        
        # Adapter skips record missing 'id'
        expected_milvus_data = [[2]] 

        result = self.adapter.load_data(collection_name, data_to_load)
        
        mock_collection_instance.insert.assert_called_once_with(expected_milvus_data)
        self.assertEqual(result["insert_count"], 1)
        self.assertEqual(result["total_processed_count"], 1) # One record processed
        self.assertEqual(result["total_input_count"], 2) # Two records in input
        self.assertEqual(result["primary_keys_inserted"], [2])


    @patch('vectordb_migration.adapters.milvus.Collection')
    @patch('vectordb_migration.adapters.milvus.utility')
    @patch.object(MilvusAdapter, 'get_schema_info')
    def test_load_data_partial_success_and_error_reporting(self, mock_get_schema_info, mock_utility, MockCollection):
        """Test load_data when Milvus reports fewer inserts than processed, indicating partial success/error."""
        self.adapter.client = "connected"
        collection_name = "load_partial_success"
        mock_utility.has_collection.return_value = True
        mock_schema = { "schema": { "primary_field": "id", "fields": [{"name": "id", "type": "INT64", "is_primary": True}]}}
        mock_get_schema_info.return_value = mock_schema

        data_to_load = [{"id": 1}, {"id": 2}, {"id": 3}]
        
        mock_collection_instance = MockCollection.return_value
        mock_insert_result = MagicMock()
        mock_insert_result.insert_count = 1 # Milvus only inserted 1
        mock_insert_result.primary_keys = [1]
        # MutationResult might also have succ_index, err_index for more detailed errors
        mock_collection_instance.insert.return_value = mock_insert_result
        
        expected_milvus_data = [[1, 2, 3]]

        result = self.adapter.load_data(collection_name, data_to_load)

        mock_collection_instance.insert.assert_called_once_with(expected_milvus_data)
        self.assertEqual(result["insert_count"], 1)
        self.assertEqual(result["total_processed_count"], 3)
        self.assertTrue(len(result["errors"]) > 0) # Expecting an error for discrepancy
        self.assertTrue("Discrepancy" in result["errors"][0])


    @patch('vectordb_migration.adapters.milvus.Collection')
    @patch('vectordb_migration.adapters.milvus.utility')
    @patch.object(MilvusAdapter, 'get_schema_info')
    def test_load_data_insert_exception(self, mock_get_schema_info, mock_utility, MockCollection):
        """Test exception handling during Milvus insert operation."""
        self.adapter.client = "connected"
        collection_name = "load_insert_exception_coll"
        mock_utility.has_collection.return_value = True
        mock_schema = { "schema": { "primary_field": "id", "fields": [{"name": "id", "type": "INT64", "is_primary": True}]}}
        mock_get_schema_info.return_value = mock_schema

        data_to_load = [{"id": 1}]
        
        mock_collection_instance = MockCollection.return_value
        mock_collection_instance.insert.side_effect = Exception("Milvus insert error")

        result = self.adapter.load_data(collection_name, data_to_load)

        self.assertEqual(result["insert_count"], 0)
        self.assertEqual(result["total_processed_count"], 1) # Processed 1 before error
        self.assertEqual(len(result["errors"]), 1)
        self.assertTrue("Milvus insert error" in result["errors"][0])
        self.assertEqual(result["failure_count"], len(data_to_load)) # Kept for qdrant consistency


if __name__ == '__main__':
    unittest.main()
