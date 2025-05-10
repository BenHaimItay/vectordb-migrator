"""
Entry point for the vectordb_migration package when run as a module.

Usage:
    python -m vectordb_migration --config path/to/config.json [--transform path/to/transform.py] [--verbose]
"""

from vectordb_migration.cli import main

if __name__ == "__main__":
    main()