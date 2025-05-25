-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table to store vector data
CREATE TABLE IF NOT EXISTS vector_items (
    id SERIAL PRIMARY KEY,
    embedding vector(3), -- Example: 3-dimensional vectors
    metadata JSONB
);

-- Insert some sample data
INSERT INTO vector_items (embedding, metadata) VALUES
(ARRAY[0.1, 0.2, 0.3], '{"source": "pgvector", "category": "A", "info": "First item"}'),
(ARRAY[0.4, 0.5, 0.6], '{"source": "pgvector", "category": "B", "info": "Second item"}'),
(ARRAY[0.7, 0.8, 0.9], '{"source": "pgvector", "category": "A", "info": "Third item"}'),
(ARRAY[0.1, 0.1, 0.1], '{"source": "pgvector", "category": "C", "info": "Fourth item, different category"}');

-- Optional: Log that the script has run
DO $$
BEGIN
  RAISE NOTICE 'pgvector init.sql script executed successfully.';
END $$;
