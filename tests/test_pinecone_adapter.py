#!/usr/bin/env python
"""
Simple test script for the Pinecone adapter.

This script tests basic functionality of the Pinecone adapter by:
1. Installing the pinecone-client if not already installed
2. Creating a simple connection to Pinecone
3. Validating the adapter can be initialized properly

Usage:
    python test_pinecone_adapter.py --api-key YOUR_API_KEY --environment YOUR_ENV
"""

import argparse
import logging
import sys
from typing import Any, Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pinecone_test")

def parse_args():
    parser = argparse.ArgumentParser(description="Test Pinecone adapter")
    parser.add_argument("--api-key", required=True, help="Pinecone API key")
    parser.add_argument("--environment", required=True, help="Pinecone environment")
    parser.add_argument("--index-name", default="test-index", help="Name of index to use for testing")
    parser.add_argument("--dimension", type=int, default=4, help="Vector dimension for test")
    return parser.parse_args()

def test_pinecone_connection(api_key: str, environment: str):
    """Test that we can connect to Pinecone"""
    try:
        import pinecone
        logger.info("Pinecone client is installed")

        # Initialize Pinecone client
        pc = pinecone.Pinecone(api_key=api_key, environment=environment)
        
        # List indexes to verify connection
        indexes = pc.list_indexes()
        logger.info(f"Successfully connected to Pinecone. Available indexes: {[idx.name for idx in indexes]}")
        return True
    except ImportError:
        logger.error("Pinecone client is not installed. Run 'pip install pinecone-client'")
        return False
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")
        return False

def test_adapter_initialization(api_key: str, environment: str, index_name: str):
    """Test that the adapter can be initialized correctly"""
    try:
        from vectordb_migration.adapters import get_adapter

        # Get the adapter class
        PineconeAdapter = get_adapter("pinecone")
        if not PineconeAdapter:
            logger.error("Could not get Pinecone adapter class")
            return False
            
        # Create an instance of the adapter
        adapter = PineconeAdapter()
        
        # Test the connect method
        connection_successful = adapter.connect(
            api_key=api_key,
            environment=environment
        )
        
        if not connection_successful:
            logger.error("Failed to connect to Pinecone")
            return False
            
        logger.info("Successfully initialized Pinecone adapter")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone adapter: {e}")
        return False

def test_adapter_methods(api_key: str, environment: str, index_name: str, dimension: int):
    """Test basic adapter functionality"""
    try:
        from vectordb_migration.adapters import get_adapter

        # Get the adapter class
        PineconeAdapter = get_adapter("pinecone")
        if not PineconeAdapter:
            logger.error("Could not get Pinecone adapter class")
            return False
            
        # Create an instance of the adapter
        adapter = PineconeAdapter()
        
        # Connect to Pinecone
        connection_successful = adapter.connect(
            api_key=api_key,
            environment=environment
        )
        
        if not connection_successful:
            logger.error("Failed to connect to Pinecone")
            return False
        
        # Check the methods exist
        if not hasattr(adapter, "extract_data"):
            logger.error("Adapter is missing 'extract_data' method")
            return False
            
        if not hasattr(adapter, "load_data"):
            logger.error("Adapter is missing 'load_data' method")
            return False
            
        if not hasattr(adapter, "get_schema_info"):
            logger.error("Adapter is missing 'get_schema_info' method")
            return False
            
        logger.info("Adapter has all required methods")
        
        # Disconnect when done
        adapter.disconnect()
        
        return True
    except Exception as e:
        logger.error(f"Error testing adapter methods: {e}")
        return False

def main():
    args = parse_args()
    
    # Test connection to Pinecone
    if not test_pinecone_connection(args.api_key, args.environment):
        logger.error("Pinecone connection test failed")
        return 1
        
    # Test adapter initialization
    if not test_adapter_initialization(args.api_key, args.environment, args.index_name):
        logger.error("Adapter initialization test failed")
        return 1
        
    # Test adapter methods
    if not test_adapter_methods(args.api_key, args.environment, args.index_name, args.dimension):
        logger.error("Adapter methods test failed")
        return 1
    
    logger.info("All tests passed! Pinecone adapter appears to be working correctly.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
