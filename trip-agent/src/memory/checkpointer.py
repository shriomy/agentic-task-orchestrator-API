from langgraph.checkpointer.postgres import PostgresCheckpointer
from ..config import settings


# Conversation memory is stored per thread in Postgres via LangGraph's checkpointer.
# This is distinct from user memory, which is durable and queryable through dedicated Supabase tables.
conversation_checkpointer = PostgresCheckpointer(
    dsn=settings.supabase_db_url,
    table_name="conversation_checkpoints",
)
