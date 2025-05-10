"""
Core functionality for vectordb_migration

This module provides the core classes and functions for the vector database migration framework.
"""

from vectordb_migration.core.adapter import VectorDBAdapter
from vectordb_migration.core.migrator import DBMigrator

__all__ = ['VectorDBAdapter', 'DBMigrator']