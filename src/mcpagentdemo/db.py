import os
import sys
import duckdb
import urllib.request
import logging

logger = logging.getLogger("mcpagentdemo.db")

REMOTE_DB_URL = "https://media.githubusercontent.com/media/KedarDamale/MCPAgentDemo/main/data/pharma_sales.duckdb"

def get_db_path() -> str:
    """Resolve database path: checks environment, local data/, or downloads to /tmp on serverless."""
    env_path = os.getenv("DUCKDB_PATH")
    if env_path and os.path.exists(env_path) and os.path.getsize(env_path) > 1000:
        return env_path
    
    # Check local repository path
    full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "pharma_sales.duckdb")
    if os.path.exists(full_path) and os.path.getsize(full_path) > 1000:
        return full_path
    elif os.path.exists("data/pharma_sales.duckdb") and os.path.getsize("data/pharma_sales.duckdb") > 1000:
        return "data/pharma_sales.duckdb"

    # Serverless fallback: download to writable /tmp
    tmp_path = "/tmp/pharma_sales.duckdb"
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1000:
        logger.info(f"Downloading DuckDB database to {tmp_path} from GitHub...")
        urllib.request.urlretrieve(REMOTE_DB_URL, tmp_path)
    return tmp_path

def get_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Return a read-only DuckDB connection with external file access disabled."""
    db_path = get_db_path()
    # MCP callers can submit analytical SQL. Prevent DuckDB table functions such
    # as read_csv/read_parquet from reaching files or network locations outside
    # the bundled database, in addition to opening the database read-only.
    return duckdb.connect(
        db_path,
        read_only=read_only,
        config={"enable_external_access": "false"},
    )
