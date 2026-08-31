/*
# Per-message token/cost usage

1. Table
- `agent_message_usage` : one row per assistant-visible message (a completed
                          reply or a pause-for-selection), recording the
                          token breakdown across context, memory, system
                          prompt, tools and other, plus an estimated USD cost.
                          Linked to `agent_messages` via `message_id` and to
                          `agent_conversations` via `conversation_id` (kept
                          denormalised on the row so RLS doesn't need a join
                          through a nullable FK).

2. Security
- Same pattern as `agent_selection_log` in 001_agent_tables.sql: RLS scoped
  TO authenticated, ownership inherited through the parent conversation. The
  backend writes via the service role key (bypasses RLS); this policy is what
  lets the frontend's anon client read usage back when reloading a thread.
*/

-- ---------------------------------------------------------------------------
-- Per-message usage
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_message_usage (
  id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id       uuid NOT NULL REFERENCES agent_conversations(id) ON DELETE CASCADE,
  message_id            uuid REFERENCES agent_messages(id) ON DELETE CASCADE,
  turn_index            integer NOT NULL DEFAULT 0,
  model                 text,
  context_tokens        integer NOT NULL DEFAULT 0,
  memory_tokens         integer NOT NULL DEFAULT 0,
  system_prompt_tokens  integer NOT NULL DEFAULT 0,
  tools_tokens          integer NOT NULL DEFAULT 0,
  other_tokens          integer NOT NULL DEFAULT 0,
  total_tokens          integer NOT NULL DEFAULT 0,
  cost_usd              numeric(12,6),
  created_at            timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Row level security
-- ---------------------------------------------------------------------------
ALTER TABLE agent_message_usage ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS agent_message_usage_select ON agent_message_usage;
CREATE POLICY agent_message_usage_select ON agent_message_usage FOR SELECT
  TO authenticated USING (EXISTS (
    SELECT 1 FROM agent_conversations c
    WHERE c.id = agent_message_usage.conversation_id AND c.user_id = auth.uid()));

DROP POLICY IF EXISTS agent_message_usage_insert ON agent_message_usage;
CREATE POLICY agent_message_usage_insert ON agent_message_usage FOR INSERT
  TO authenticated WITH CHECK (EXISTS (
    SELECT 1 FROM agent_conversations c
    WHERE c.id = agent_message_usage.conversation_id AND c.user_id = auth.uid()));

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_agent_message_usage_conversation_created
  ON agent_message_usage (conversation_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_agent_message_usage_message
  ON agent_message_usage (message_id);
