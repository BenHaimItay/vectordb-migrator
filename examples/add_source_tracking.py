"""
Example transformation module for vectordb_migration

This module demonstrates how to create a custom transformation function
that can be used with the vectordb_migration tool to modify data
during the migration process.

Usage:
    vectordb-migrate --config config.json --transform examples/add_source_tracking.py
"""

def transform(data):
    """
    Add source tracking information to each item's metadata.
    
    This transformation adds two fields to each item's metadata:
    - source_db: The name of the source database
    - migration_timestamp: When the migration occurred
    
    Args:
        data: A list of items with id, vector, and metadata fields
        
    Returns:
        The transformed data with additional metadata fields
    """
    import datetime
    
    source_db = "custom_source"  # You might get this from elsewhere
    timestamp = datetime.datetime.now().isoformat()
    
    # Process each item
    for item in data:
        # Ensure metadata dict exists
        if "metadata" not in item:
            item["metadata"] = {}
            
        # Add tracking information
        item["metadata"]["source_db"] = source_db
        item["metadata"]["migration_timestamp"] = timestamp
        
        # You can perform other transformations here, such as:
        # - Renaming fields
        # - Filtering out certain items
        # - Modifying vector values
        # - Adding computed fields
        
    return data


# If this module is run directly, demonstrate the transformation
if __name__ == "__main__":
    # Sample data
    sample_data = [
        {"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": {"name": "Item One"}},
        {"id": 2, "vector": [0.4, 0.5, 0.6], "metadata": {"name": "Item Two"}},
    ]
    
    # Apply transformation
    transformed = transform(sample_data)
    
    # Print results
    import json
    print(json.dumps(transformed, indent=2))