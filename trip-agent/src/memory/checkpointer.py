from psycopg import Connection
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

from ..config import settings


# Conversation memory is stored per thread in Postgres via LangGraph's checkpointer.
# This is distinct from user memory, which is durable and queryable through dedicated Supabase tables.
connection = Connection.connect(
    settings.supabase_db_url,
    autocommit=True,
    prepare_threshold=0,
    row_factory=dict_row,
)
conversation_checkpointer = PostgresSaver(connection)
