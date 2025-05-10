"""
Command-line interface package for vector database migration

This package provides the command-line tools for the vectordb_migration package.
"""

from vectordb_migration.cli.migrate import main, run_migration

__all__ = ['main', 'run_migration']