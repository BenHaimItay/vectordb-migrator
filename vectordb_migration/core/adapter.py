"""
Vector DB Migration Core Functionality

This module provides the core abstractions and interfaces for the vector database
migration framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Tuple


class VectorDBAdapter(ABC):
    """Abstract base class for vector database adapters."""
    
    @abstractmethod
    def connect(self, **connection_params) -> bool:
        """Connect to the database using provided parameters.
        
        Args:
            **connection_params: Connection parameters specific to the database.
            
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close the database connection."""
        pass
    
    @abstractmethod
    def extract_data(self, **query_params) -> List[Dict[str, Any]]:
        """Extract data from the database, returning a list of items with ID, vectors, and metadata.
        
        Args:
            **query_params: Parameters controlling what data to extract.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries with 'id', 'vector', and 'metadata' keys.
        """
        pass
    
    @abstractmethod
    def load_data(self, data: List[Dict[str, Any]], **load_params) -> bool:
        """Load data into the database.
        
        Args:
            data: A list of dictionaries with 'id', 'vector', and 'metadata' keys.
            **load_params: Parameters controlling how data is loaded.
            
        Returns:
            bool: True if loading was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_schema_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Get information about the database schema including vector dimensions.
        
        Args:
            collection_name: Optional name of the specific collection/table to inspect.
            
        Returns:
            Dict[str, Any]: Schema information including vector dimensions.
        """
        pass