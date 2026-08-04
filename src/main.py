from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent import Agent, AgentError, build_tool_definitions
from src.config import settings
from src.llm import ChatCompletionsClient
from src.state import AgentState


app = FastAPI(title="Agent From Scratch")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str


def build_agent() -> Agent:
    tools, tool_functions = build_tool_definitions()
    llm_client = ChatCompletionsClient(
        api_key=settings.llm_api_key,
        api_base=settings.llm_api_base,
        model=settings.llm_model,
        timeout_seconds=settings.request_timeout_seconds,
    )
    return Agent(llm_client=llm_client, tools=tools, tool_functions=tool_functions, max_iterations=settings.max_iterations)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    agent = build_agent()
    try:
        result = agent.run(request.message, state=AgentState())
    except AgentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ChatResponse(answer=result.answer)
