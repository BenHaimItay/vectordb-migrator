"""
Command-line interface for vector database migration

This module provides a CLI for the vector database migration tool, allowing users to configure
and execute migrations between different vector database systems.
"""

import os
import sys
import json
import logging
import argparse
import importlib.util
from typing import Dict, Any, List, Callable, Optional

from vectordb_migration.adapters import ADAPTERS
from vectordb_migration.core.migrator import DBMigrator


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_config(config_file: str) -> Dict[str, Any]:
    """Load and validate the migration configuration from a JSON file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Dict[str, Any]: The configuration
        
    Raises:
        ValueError: If the configuration is invalid
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Basic validation
        required_keys = ["source", "target"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in config file")
            
            if "type" not in config[key]:
                raise ValueError(f"Missing 'type' in {key} configuration")
            
            db_type = config[key]["type"]
            if db_type not in ADAPTERS:
                valid_types = ", ".join(ADAPTERS.keys())
                raise ValueError(f"Unsupported {key} database type: {db_type}. Valid types: {valid_types}")
        
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing config file: {e}")
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {config_file}")


def load_transform_function(transform_module_path: str) -> Optional[Callable]:
    """Load a transformation function from a Python module.
    
    Args:
        transform_module_path: Path to the Python module containing a transform function
        
    Returns:
        Optional[Callable]: The transform function or None if not found
    """
    if not transform_module_path:
        return None
    
    try:
        module_name = os.path.basename(transform_module_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, transform_module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for a function named 'transform' in the module
        if hasattr(module, 'transform') and callable(module.transform):
            return module.transform
        else:
            logger.warning(f"No 'transform' function found in {transform_module_path}. No transformation will be applied.")
            return None
    except Exception as e:
        logger.error(f"Error loading transform module: {e}")
        logger.warning("No transformation will be applied.")
        return None


def run_migration(config_file: str, transform_file: Optional[str] = None, verbose: bool = False) -> bool:
    """Run a vector database migration using the specified config.
    
    Args:
        config_file: Path to the configuration file
        transform_file: Optional path to a transform module
        verbose: Whether to enable verbose logging
        
    Returns:
        bool: True if migration was successful, False otherwise
    """
    # Set up logging level
    if verbose:
        logging.getLogger('vectordb_migration').setLevel(logging.DEBUG)
    
    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from {config_file}")
        config = load_config(config_file)
        
        # Load transform function if provided
        transform_func = None
        if transform_file:
            logger.info(f"Loading transform function from {transform_file}")
            transform_func = load_transform_function(transform_file)
        
        # Create and run the migrator
        logger.info(f"Starting migration from {config['source']['type']} to {config['target']['type']}")
        migrator = DBMigrator(
            adapters_registry=ADAPTERS,
            source_type=config["source"]["type"],
            target_type=config["target"]["type"]
        )
        
        success = migrator.migrate(
            source_params=config["source"],
            target_params=config["target"],
            transform_func=transform_func
        )
        
        if success:
            logger.info("Migration completed successfully!")
        else:
            logger.error("Migration failed. See above logs for details.")
        
        return success
    
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return False
    except Exception as e:
        logger.error(f"Migration failed with unexpected error: {e}", exc_info=True)
        return False


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Vector Database Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic migration with config file
  python -m vectordb_migration --config path/to/config.json
  
  # Migration with custom transformation
  python -m vectordb_migration --config path/to/config.json --transform path/to/transform.py
  
  # With verbose logging
  python -m vectordb_migration --config path/to/config.json --verbose
  
Config file format:
  {
    "source": {
      "type": "pgvector",
      "connection": {
        "host": "localhost",
        "dbname": "vectordb",
        "user": "postgres",
        "password": "password"
      },
      "query": {
        "table_name": "items",
        "id_column": "id",
        "vector_column": "embedding",
        "metadata_columns": ["name", "description"]
      }
    },
    "target": {
      "type": "qdrant",
      "connection": {
        "host": "localhost",
        "port": 6333
      },
      "load": {
        "collection_name": "items_collection",
        "recreate_collection": true,
        "distance": "Cosine"
      }
    }
  }
        """
    )
    
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to the migration configuration JSON file"
    )
    parser.add_argument(
        "--transform", 
        help="Path to a Python module with a 'transform' function for customizing data"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--version", 
        action="store_true", 
        help="Print version information and exit"
    )
    
    args = parser.parse_args()
    
    if args.version:
        from vectordb_migration import __version__
        print(f"vectordb_migration version {__version__}")
        return 0
    
    success = run_migration(
        config_file=args.config,
        transform_file=args.transform,
        verbose=args.verbose
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())