import os
import duckdb

def get_db_path() -> str:
    """Resolve database path: checks environment, sample db, or full db."""
    env_path = os.getenv("DUCKDB_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Check sample db first if on serverless/vercel
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "pharma_sales_sample.duckdb")
    if os.getenv("VERCEL") and os.path.exists(sample_path):
        return sample_path
        
    full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "pharma_sales.duckdb")
    if os.path.exists(full_path):
        return full_path
    elif os.path.exists(sample_path):
        return sample_path
    elif os.path.exists("data/pharma_sales.duckdb"):
        return "data/pharma_sales.duckdb"
    elif os.path.exists("data/pharma_sales_sample.duckdb"):
        return "data/pharma_sales_sample.duckdb"
    return "data/pharma_sales.duckdb"

def get_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with read-only default and optimized config."""
    db_path = get_db_path()
    return duckdb.connect(db_path, read_only=read_only)
