"""Chat model construction.

One place decides which provider/base URL/key to use, so the agent, the
guardrail classifiers and the summarizer all stay consistent.
"""

from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model

from .config import settings


def _resolve_provider() -> tuple[str, str | None, str | None, bool]:
    """Decide provider, key and base URL.

    Returns `(provider, api_key, base_url, via_openrouter)`. OpenRouter is used
    when it is named explicitly, or when it has a key and the direct LLM key is
    blank — which is how the checked-in .env is set up.
    """
    provider = (settings.llm_provider or "openai").lower()
    api_key = settings.llm_api_key or None
    base_url = settings.llm_api_base or None

    use_openrouter = bool(settings.openrouter_api_key) and (provider == "openrouter" or not api_key)
    if use_openrouter:
        # OpenRouter is OpenAI-compatible, so it is driven through that adapter.
        return (
            "openai",
            settings.openrouter_api_key,
            settings.openrouter_api_base or "https://openrouter.ai/api/v1",
            True,
        )
    return provider, api_key, base_url, False


@lru_cache(maxsize=8)
def build_model(model_name: str | None = None, *, temperature: float = 0.2, fast: bool = False) -> Any:
    """Return a chat model. Cached, because rebuilding per node call is wasteful."""
    provider, api_key, base_url, via_openrouter = _resolve_provider()

    if not model_name:
        if fast:
            model_name = settings.llm_fast_model
        elif via_openrouter:
            model_name = settings.openrouter_default_model or settings.llm_model
        else:
            model_name = settings.llm_model

    if via_openrouter and "/" not in model_name:
        # OpenRouter identifies models as "provider/model"; a bare "gpt-4o-mini"
        # is rejected. This matters for the fast model, which is configured
        # without a prefix so it also works when talking to OpenAI directly.
        model_name = f"openai/{model_name}"

    kwargs: dict[str, Any] = {"temperature": temperature}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return init_chat_model(model=model_name, model_provider=provider, **kwargs)


def fast_model() -> Any:
    """Cheap model for the classifier guardrails and summarization."""
    return build_model(fast=True, temperature=0.0)
