import logging
from pymilvus import connections, utility, Collection
from vectordb_migration.vector_db_adapter import VectorDBAdapter

logger = logging.getLogger(__name__)

class MilvusAdapter(VectorDBAdapter):
    def __init__(self, **kwargs):
        self.client = None
        self.connection_params = kwargs

    def connect(self, **kwargs):
        """
        Connects to the Milvus server.
        kwargs: connection parameters like host, port, user, password, secure, db_name, etc.
        """
        self.connection_params.update(kwargs)
        alias = self.connection_params.pop("alias", "default") # Default alias for Milvus connection

        try:
            logger.info(f"Connecting to Milvus with params: {self.connection_params} and alias: {alias}")
            connections.connect(alias=alias, **self.connection_params)
            self.client = "connected" # Placeholder, real client interaction will be through pymilvus classes
            logger.info("Successfully connected to Milvus.")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def disconnect(self):
        """Disconnects from the Milvus server."""
        alias = self.connection_params.get("alias", "default")
        try:
            logger.info(f"Disconnecting from Milvus alias: {alias}")
            connections.disconnect(alias)
            self.client = None
            logger.info("Successfully disconnected from Milvus.")
        except Exception as e:
            logger.error(f"Failed to disconnect from Milvus: {e}")
            # Not raising an error here, as disconnect might be called on an already disconnected client
            pass

    def extract_data(self, collection_name: str, limit: int = 100, offset: int = 0, filter_expr: str = None):
        """
        Extracts data from a Milvus collection with pagination and filtering.

        Args:
            collection_name (str): The name of the collection.
            limit (int): The maximum number of records to return.
            offset (int): The number of records to skip from the beginning.
            filter_expr (str, optional): A Milvus filter expression. Defaults to None.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains
                        'id', 'vector', and 'metadata' keys.
                        Returns an empty list if the collection is empty or an error occurs.
        """
        if not self.client:
            logger.error("Not connected to Milvus. Call connect() first.")
            raise ConnectionError("Not connected to Milvus. Call connect() first.")

        try:
            if not utility.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} does not exist.")
                return []

            collection = Collection(collection_name)
            collection.load() # Ensure collection is loaded for querying

            # Determine output fields: primary key, vector field, and all other scalar fields for metadata
            schema_info = self.get_schema_info(collection_name)
            if not schema_info or not schema_info.get("schema"):
                logger.error(f"Could not retrieve schema for collection {collection_name} to determine output fields.")
                return []

            primary_key_field_name = schema_info["schema"]["primary_field"]
            vector_field_name = None
            metadata_field_names = []

            for field in schema_info["schema"]["fields"]:
                # Heuristic: assume the first field of type FLOAT_VECTOR or BINARY_VECTOR is the vector field.
                # This might need to be more robust if multiple vector fields exist.
                if field["type"] in ["FLOAT_VECTOR", "BINARY_VECTOR"] and not vector_field_name:
                    vector_field_name = field["name"]
                elif not field["is_primary"] and field["type"] not in ["FLOAT_VECTOR", "BINARY_VECTOR"]:
                    metadata_field_names.append(field["name"])
            
            if not primary_key_field_name:
                logger.error(f"Primary key field could not be identified for collection {collection_name}.")
                return []
            if not vector_field_name:
                logger.warning(f"No vector field found in schema for collection {collection_name}. Returning data without vectors.")
                # If no vector field, we can still extract metadata and IDs
                output_fields = [primary_key_field_name] + metadata_field_names
            else:
                 output_fields = [primary_key_field_name, vector_field_name] + metadata_field_names


            logger.info(f"Extracting data from {collection_name} with limit={limit}, offset={offset}, filter='{filter_expr}', output_fields={output_fields}")

            # Milvus query parameters
            query_params = {
                "expr": filter_expr,
                "output_fields": output_fields,
                "limit": limit,
                "offset": offset,
                # "consistency_level": "Strong" # Or another appropriate level
            }
            
            # Remove None filter_expr for the query call
            if filter_expr is None:
                query_params.pop("expr")

            results = collection.query(**query_params)
            
            extracted_data = []
            for res in results:
                record = {
                    "id": res.get(primary_key_field_name),
                    "vector": res.get(vector_field_name) if vector_field_name else None,
                    "metadata": {mf: res.get(mf) for mf in metadata_field_names if res.get(mf) is not None}
                }
                extracted_data.append(record)
            
            logger.info(f"Successfully extracted {len(extracted_data)} records from {collection_name}.")
            return extracted_data

        except Exception as e:
            logger.error(f"Failed to extract data from collection {collection_name}: {e}")
            # raise # Optionally re-raise
            return []

    def load_data(self, collection_name: str, data: list[dict]):
        """
        Loads data into a Milvus collection.

        Args:
            collection_name (str): The name of the collection.
            data (list[dict]): A list of dictionaries, where each dictionary must contain
                               'id', and recommended to have 'vector', and 'metadata' keys. 
                               The 'metadata' should be a flat dictionary of field_name: value.

        Returns:
            dict: A dictionary containing the results of the insert operation, e.g., insert count.
                  Milvus insert operation returns MutationResult which includes insert_count.
        """
        if not self.client:
            logger.error("Not connected to Milvus. Call connect() first.")
            raise ConnectionError("Not connected to Milvus. Call connect() first.")

        if not data:
            logger.info("No data provided to load.")
            return {"insert_count": 0, "errors": [], "success_count": 0, "failure_count": 0}

        try:
            if not utility.has_collection(collection_name):
                logger.error(f"Collection {collection_name} does not exist. Data loading requires an existing collection.")
                # This adapter will not create collections. That should be a separate setup step.
                raise ValueError(f"Collection {collection_name} does not exist.")

            collection = Collection(collection_name)
            
            schema_info = self.get_schema_info(collection_name)
            if not schema_info or not schema_info.get("schema"):
                logger.error(f"Could not retrieve schema for collection {collection_name}. Cannot prepare data for insertion.")
                raise ValueError(f"Could not retrieve schema for {collection_name}.")

            primary_key_field_name = schema_info["schema"]["primary_field"]
            vector_field_name = None 
            metadata_field_names_from_schema = []
            schema_field_details = {field["name"]: field for field in schema_info["schema"]["fields"]}


            for field in schema_info["schema"]["fields"]:
                if field["type"] in ["FLOAT_VECTOR", "BINARY_VECTOR"] and not vector_field_name: # First vector field found
                    vector_field_name = field["name"]
                elif not field["is_primary"] and field["type"] not in ["FLOAT_VECTOR", "BINARY_VECTOR"]:
                    metadata_field_names_from_schema.append(field["name"])
            
            if not primary_key_field_name:
                 raise ValueError(f"Primary key not found in schema for {collection_name}")

            # Initialize data lists for each field defined in the schema
            # Milvus expects a list of lists, where each inner list is a column of data
            columns_for_milvus = {field_name: [] for field_name in schema_field_details.keys()}
            processed_ids = []
            
            for record_idx, record in enumerate(data):
                current_id = record.get('id')
                if current_id is None:
                    logger.warning(f"Record at index {record_idx} is missing 'id'. Skipping this record.")
                    continue
                processed_ids.append(current_id)

                # 1. Primary Key
                columns_for_milvus[primary_key_field_name].append(current_id)

                # 2. Vector Field
                if vector_field_name:
                    if 'vector' not in record or record['vector'] is None:
                        # If schema has a vector field, it's usually mandatory.
                        # Depending on Milvus version & schema, None might not be acceptable.
                        # For now, assume if vector field is in schema, vector must be provided.
                        logger.error(f"Record with id '{current_id}' is missing a vector for field '{vector_field_name}', but schema requires it. Skipping record or use default if schema allows.")
                        # This is problematic for batch insert, as one bad record can fail the batch.
                        # A strategy could be to exclude this record and report it.
                        # For simplicity here, we'll let it try and Milvus will error if it's an issue.
                        # Or, if the schema allows, a default/None could be inserted.
                        # For now, let's assume we must have the vector if the field exists.
                        # If you want to allow missing vectors, the logic here and data prep needs adjustment.
                        # A placeholder or specific handling might be needed if vectors can be optional.
                        # For now, if vector field exists, it must be in the record.
                        if schema_field_details[vector_field_name].get('auto_id', False) is False: # Not an auto-id field
                             # It's not clear if a vector field can be auto-id. Usually primary keys are.
                             # If the vector field is critical and not provided, this row is problematic.
                             # Let's add None and let Milvus decide. Or raise error earlier.
                             columns_for_milvus[vector_field_name].append(None) # This might fail in Milvus
                             logger.warning(f"Record id '{current_id}': Vector for '{vector_field_name}' is None/missing. Appending None.")

                        # raise ValueError(f"Record with id {current_id} is missing a vector for field {vector_field_name}")
                    else:
                        columns_for_milvus[vector_field_name].append(record['vector'])
                
                # 3. Metadata Fields
                record_metadata = record.get('metadata', {})
                for schema_meta_field_name in metadata_field_names_from_schema:
                    # If metadata from record is present for this schema field, add it.
                    # Otherwise, add None (Milvus may or may not accept None depending on field definition)
                    columns_for_milvus[schema_meta_field_name].append(record_metadata.get(schema_meta_field_name))
                
                # Warn about metadata in record that's not in schema
                for record_meta_key in record_metadata.keys():
                    if record_meta_key not in schema_field_details:
                        logger.warning(f"Metadata field '{record_meta_key}' from input data for id '{current_id}' not found in collection schema. It will be ignored.")
            
            # Ensure all columns have the same number of entries (equal to number of processed records)
            # This is critical for Milvus. If a primary key was processed, all other fields must have an entry for it.
            num_processed_records = len(processed_ids)
            final_ordered_milvus_data = [] # This will be the list of lists for Milvus
            
            if num_processed_records == 0:
                logger.info("No valid records to load after initial processing.")
                return {"insert_count": 0, "errors": ["No valid records processed"], "success_count": 0, "failure_count": len(data)}


            for field_schema in schema_info["schema"]["fields"]:
                field_name = field_schema["name"]
                if len(columns_for_milvus[field_name]) != num_processed_records:
                    # This case should ideally be handled by the appending logic above (e.g. by appending None for missing fields)
                    # If this error still occurs, it means there's a logic flaw in data preparation.
                    err_msg = (f"Mismatch in data length for field '{field_name}'. "
                               f"Expected {num_processed_records} entries, got {len(columns_for_milvus[field_name])}. "
                               "This indicates an issue with data preparation for batch loading.")
                    logger.error(err_msg)
                    raise ValueError(err_msg)
                final_ordered_milvus_data.append(columns_for_milvus[field_name])

            logger.info(f"Attempting to load {num_processed_records} records into {collection_name}.")
            
            insert_result = collection.insert(final_ordered_milvus_data)
            # collection.flush() # Consider if flush is needed here
            # logger.info(f"Flushed data for collection {collection_name} (if applicable).")

            # MutationResult contains insert_count, primary_keys (succ_index), err_index etc.
            # We need to interpret this to give a clear success/failure count.
            # Milvus's insert_count is the number of successfully inserted rows.
            success_count = insert_result.insert_count
            # Milvus does not directly return a list of errors for partial failures in the same way Qdrant might.
            # It throws an exception for batch failures or provides error indices for some operations.
            # For simplicity, if an exception wasn't thrown, we assume all processed records attempted were covered by insert_count.
            # If insert_count < num_processed_records, it implies partial success not fully detailed by MutationResult alone without deeper inspection.
            # However, typically if there's a row-level error in the batch, pymilvus raises an exception.
            
            errors_reported = []
            if success_count < num_processed_records :
                 # This implies partial success or an issue not caught by an exception.
                 # Milvus's MutationResult (succ_index, err_index) might be useful if available and detailed.
                 # For now, a generic message for discrepancy.
                 logger.warning(f"Milvus reported {success_count} inserts, but {num_processed_records} records were processed. Possible partial failure or miscount.")
                 errors_reported.append(f"Discrepancy: {num_processed_records} processed, {success_count} inserted.")
            
            failure_count = num_processed_records - success_count + (len(data) - num_processed_records)


            logger.info(f"Load operation complete for {collection_name}. Successfully inserted: {success_count}, Failed or skipped: {failure_count}.")
            return {
                "insert_count": success_count, # Actual count from Milvus
                "total_processed_count": num_processed_records,
                "total_input_count": len(data),
                "primary_keys_inserted": insert_result.primary_keys, # IDs of successfully inserted records
                "errors": errors_reported 
            }

        except Exception as e:
            logger.error(f"Failed to load data into collection {collection_name}: {e}", exc_info=True)
            # If an exception occurs, assume all processed records up to this point failed.
            # num_processed_records might not be accurate if error is early (e.g. schema fetch)
            # Fallback to assuming all input records failed if the error is general.
            processed_count_at_error = len(processed_ids) if 'processed_ids' in locals() else 0
            return {
                "insert_count": 0, 
                "total_processed_count": processed_count_at_error,
                "total_input_count": len(data),
                "errors": [str(e)],
                "success_count":0, #kept for consistency from qdrant
                "failure_count": len(data) #kept for consistency
                }


    def get_schema_info(self, collection_name: str):
        """
        Retrieves the schema information for a given Milvus collection.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            dict: A dictionary containing schema information, including fields and collection properties.
                  Returns None if the collection does not exist or an error occurs.
        """
        if not self.client:
            logger.error("Not connected to Milvus. Call connect() first.")
            raise ConnectionError("Not connected to Milvus. Call connect() first.")

        try:
            if not utility.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} does not exist.")
                return None

            collection = Collection(collection_name)
            schema = collection.schema
            
            fields_info = []
            for field in schema.fields:
                field_data = {
                    "name": field.name,
                    "type": field.dtype.name, # Convert Milvus DataType to string
                    "is_primary": field.is_primary,
                    "description": field.description,
                    # "auto_id": field.auto_id # This attribute might not be directly available or applicable for all field types
                }
                if hasattr(field, 'params'):
                    field_data["params"] = field.params
                fields_info.append(field_data)

            collection_info = {
                "name": collection.name,
                "description": collection.description,
                "num_entities": collection.num_entities,
                "consistency_level": collection.consistency_level, # This might be a string representation
                "aliases": utility.list_aliases(collection_name),
                "properties": collection.properties, # Other collection properties
                "schema": {
                    "fields": fields_info,
                    "description": schema.description,
                    "auto_id": schema.auto_id, # If the collection schema is set to auto-generate IDs
                    "primary_field": schema.primary_field.name if schema.primary_field else None,
                }
            }
            logger.info(f"Successfully retrieved schema for collection {collection_name}.")
            return collection_info
        except Exception as e:
            logger.error(f"Failed to get schema for collection {collection_name}: {e}")
            # raise # Optionally re-raise the exception if the caller should handle it
            return None
