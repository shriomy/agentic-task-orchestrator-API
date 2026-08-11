from supabase import create_client
from ..config import settings


supabase_client = create_client(
    settings.supabase_url,
    settings.supabase_anon_key,
)

# Service role client may be required for privileged inserts or cross-table operations.
supabase_service_client = create_client(
    settings.supabase_url,
    settings.supabase_service_role_key,
)
