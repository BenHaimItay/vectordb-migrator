"""
Qdrant vector database adapter

This module provides the adapter for the Qdrant vector database.
"""

import logging
from typing import Dict, List, Any, Optional, Union

from vectordb_migration.core.adapter import VectorDBAdapter


logger = logging.getLogger(__name__)


class QdrantAdapter(VectorDBAdapter):
    """Adapter for Qdrant vector database."""
    
    def __init__(self):
        """Initialize a new Qdrant adapter."""
        self.client = None
    
    def connect(self, **connection_params) -> bool:
        """Connect to the Qdrant server.
        
        Args:
            **connection_params: Connection parameters for Qdrant.
                - host: Server host address (default: "localhost")
                - port: Server port (default: 6333)
                - api_key: Optional API key for authentication
                - https: Whether to use HTTPS (default: False)
                - grpc_port: Optional gRPC port 
                - prefer_grpc: Whether to prefer gRPC over HTTP (default: False)
                - timeout: Connection timeout in seconds
                
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        try:
            from qdrant_client import QdrantClient
            
            self.client = QdrantClient(
                host=connection_params.get("host", "localhost"),
                port=connection_params.get("port", 6333),
                api_key=connection_params.get("api_key"),
                https=connection_params.get("https"),
                grpc_port=connection_params.get("grpc_port"),
                prefer_grpc=connection_params.get("prefer_grpc", False),
                timeout=connection_params.get("timeout")
            )
            logger.debug(f"Connected to Qdrant: {connection_params.get('host')}:{connection_params.get('port')}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close the Qdrant connection."""
        self.client = None
        logger.debug("Disconnected from Qdrant")
    
    def extract_data(self, **query_params) -> List[Dict[str, Any]]:
        """Extract vector data from a Qdrant collection.
        
        Args:
            **query_params: Query parameters for extracting data.
                - collection_name: Name of the collection (default: "default_collection")
                - limit: Maximum number of points to extract (default: 1000)
                - offset: Number of points to skip (default: 0)
                - filter: Optional filter condition (Qdrant filter format)
                
        Returns:
            List[Dict[str, Any]]: List of items with id, vector, and metadata.
            
        Raises:
            ConnectionError: If not connected to Qdrant.
        """
        if not self.client:
            raise ConnectionError("Not connected to Qdrant")
        
        collection_name = query_params.get("collection_name", "default_collection")
        limit = query_params.get("limit", 1000)
        offset = query_params.get("offset", 0)
        filter_condition = query_params.get("filter")
        
        try:
            # Import models here to avoid global dependency
            from qdrant_client.http import models as rest
            
            # Check if collection exists
            try:
                collection_info = self.client.get_collection(collection_name=collection_name)
                logger.debug(f"Found Qdrant collection: {collection_name} with {collection_info.vectors_count} vectors")
            except Exception as e:
                logger.error(f"Collection {collection_name} not found in Qdrant: {e}")
                return []
            
            # Use scrolling to get all points up to the limit
            result = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=True,
                filter=filter_condition
            )
            
            # Transform Qdrant points to our common format
            data = []
            for point in result[0]:
                item = {
                    "id": point.id,
                    "vector": point.vector,
                    "metadata": point.payload
                }
                data.append(item)
            
            logger.debug(f"Extracted {len(data)} items from Qdrant collection {collection_name}")
            return data
        except Exception as e:
            logger.error(f"Error extracting data from Qdrant: {e}")
            return []
    
    def load_data(self, data: List[Dict[str, Any]], **load_params) -> bool:
        """Load vector data into a Qdrant collection.
        
        Args:
            data: List of items with id, vector, and metadata.
            **load_params: Load parameters.
                - collection_name: Name of the collection (default: "default_collection")
                - recreate_collection: Whether to recreate the collection if it exists (default: False)
                - distance: Distance function to use (default: "Cosine")
                - batch_size: Number of points per batch (default: 100)
                - on_disk: Whether to store vectors on disk (default: False)
                - hnsw_config: Optional HNSW configuration
                - quantization_config: Optional quantization configuration
                
        Returns:
            bool: True if loading was successful, False otherwise.
            
        Raises:
            ConnectionError: If not connected to Qdrant.
        """
        if not self.client:
            raise ConnectionError("Not connected to Qdrant")
        
        if not data:
            logger.error("No data to load into Qdrant")
            return False
        
        collection_name = load_params.get("collection_name", "default_collection")
        recreate_collection = load_params.get("recreate_collection", False)
        batch_size = load_params.get("batch_size", 100)
        
        try:
            # Import models
            from qdrant_client import models
            
            # Get vector dimension from the first item
            vector_dim = len(data[0]["vector"])
            
            # Map string distance name to Qdrant Distance enum
            distance = load_params.get("distance", "Cosine")
            distance_map = {
                "cosine": models.Distance.COSINE,
                "euclid": models.Distance.EUCLID,
                "dot": models.Distance.DOT
            }
            distance_func = distance_map.get(distance.lower(), models.Distance.COSINE)
            
            # Set up vector params
            vectors_config = models.VectorParams(
                size=vector_dim,
                distance=distance_func,
                on_disk=load_params.get("on_disk", False)
            )
            
            # Handle HNSW config
            hnsw_config = load_params.get("hnsw_config")
            if hnsw_config:
                vectors_config.hnsw_config = models.HnswConfigDiff(**hnsw_config)
                
            # Handle quantization config
            quantization_config = load_params.get("quantization_config")
            if quantization_config:
                vectors_config.quantization_config = models.QuantizationConfig(**quantization_config)
            
            # Check if collection exists and recreate if needed
            try:
                self.client.get_collection(collection_name=collection_name)
                if recreate_collection:
                    logger.info(f"Collection {collection_name} already exists. Deleting...")
                    self.client.delete_collection(collection_name=collection_name)
                    # Recreate collection
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=vectors_config
                    )
                    logger.info(f"Collection {collection_name} recreated with vector dimension {vector_dim}")
            except Exception as e:
                if "not found" in str(e).lower() or "404" in str(e):
                    # Collection doesn't exist, create it
                    logger.info(f"Collection {collection_name} not found. Creating...")
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=vectors_config
                    )
                    logger.info(f"Collection {collection_name} created with vector dimension {vector_dim}")
                else:
                    # Other error
                    logger.error(f"Error checking/creating Qdrant collection: {e}")
                    return False
            
            # Convert data to Qdrant points and insert in batches
            batch_count = 0
            points_to_upsert = []
            
            for item in data:
                point_id = item["id"]
                # Convert point_id to int if it's a string containing only digits
                if isinstance(point_id, str) and point_id.isdigit():
                    point_id = int(point_id)
                
                points_to_upsert.append(
                    models.PointStruct(
                        id=point_id,
                        vector=item["vector"],
                        payload=item["metadata"]
                    )
                )
                
                # If batch is full, insert and clear
                if len(points_to_upsert) >= batch_size:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=points_to_upsert
                    )
                    batch_count += 1
                    logger.debug(f"Inserted batch {batch_count} ({len(points_to_upsert)} points)")
                    points_to_upsert = []
            
            # Insert any remaining points
            if points_to_upsert:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points_to_upsert
                )
                batch_count += 1
                logger.debug(f"Inserted final batch {batch_count} ({len(points_to_upsert)} points)")
            
            logger.info(f"Successfully loaded {len(data)} items into Qdrant collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading data into Qdrant: {e}")
            return False
    
    def get_schema_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Get information about the collection schema.
        
        Args:
            collection_name: Name of the collection to inspect (default: "default_collection")
            
        Returns:
            Dict[str, Any]: Schema information.
            
        Raises:
            ConnectionError: If not connected to Qdrant.
        """
        if not self.client:
            raise ConnectionError("Not connected to Qdrant")
        
        collection_name = collection_name or "default_collection"
        
        try:
            # Get collection info
            collection_info = self.client.get_collection(collection_name=collection_name)
            
            # Get a sample point to analyze payload structure
            sample_payload = {}
            try:
                sample = self.client.scroll(
                    collection_name=collection_name,
                    limit=1,
                    with_payload=True
                )
                if sample and sample[0]:
                    sample_point = sample[0][0]
                    sample_payload = sample_point.payload
            except Exception as e:
                logger.warning(f"Could not get sample point from collection: {e}")
            
            # Extract vector configuration
            config = collection_info.config
            vector_config = {}
            
            if hasattr(config, 'params') and hasattr(config.params, 'vectors'):
                vectors = config.params.vectors
                # Handle both single and multi-vector configs
                if hasattr(vectors, 'size'):
                    # Single vector config
                    vector_config = {
                        "size": vectors.size,
                        "distance": str(vectors.distance) if hasattr(vectors, 'distance') else None,
                        "on_disk": vectors.on_disk if hasattr(vectors, 'on_disk') else False
                    }
                else:
                    # Multi-vector config
                    vector_config = {name: {
                        "size": config.size,
                        "distance": str(config.distance) if hasattr(config, 'distance') else None,
                        "on_disk": config.on_disk if hasattr(config, 'on_disk') else False
                    } for name, config in vectors.items()}
            
            return {
                "collection_name": collection_name,
                "vector_config": vector_config,
                "points_count": collection_info.vectors_count,
                "payload_sample": sample_payload
            }
        except Exception as e:
            logger.error(f"Error getting schema info from Qdrant: {e}")
            return {}