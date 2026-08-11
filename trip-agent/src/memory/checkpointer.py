from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

from ..config import settings


# Conversation memory is stored per thread in Postgres via LangGraph's checkpointer.
# This is distinct from user memory, which is durable and queryable through dedicated Supabase tables.
def _configure_connection(conn: Connection) -> None:
    conn.autocommit = True
    conn.row_factory = dict_row


connection_pool = ConnectionPool(
    settings.supabase_db_url,
    connection_class=Connection,
    configure=_configure_connection,
    min_size=settings.postgres_pool_min,
    max_size=settings.postgres_pool_max,
    timeout=settings.postgres_pool_timeout,
)
conversation_checkpointer = PostgresSaver(connection_pool)

# Ensure checkpoint tables exist before starting request handling.
conversation_checkpointer.setup()
