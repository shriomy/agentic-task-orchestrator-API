from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.types import Command
from langsmith import trace

from .graph.graph import graph
from .config import settings

app = FastAPI()


class ChatRequest(BaseModel):
    thread_id: str
    user_id: str
    message: str
    resume_token: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    user_id: str
    response: str
    resume_token: str | None = None


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        command = Command(
            update={
                "thread_id": request.thread_id,
                "user_id": request.user_id,
                "message": request.message,
            },
            resume=request.resume_token,
        )

        with trace(
            "travel_discovery_chat",
            inputs={
                "thread_id": request.thread_id,
                "user_id": request.user_id,
                "message": request.message,
            },
            project_name=settings.langsmith_project_name,
        ):
            result = graph.invoke(
                command,
                config={"configurable": {"thread_id": request.thread_id}},
            )

        response_text = str(
            result.get("last_agent_output")
            or result.get("bot_response")
            or result.get("tool_result")
            or ""
        )

        return ChatResponse(
            thread_id=request.thread_id,
            user_id=request.user_id,
            response=response_text,
            resume_token=request.resume_token,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
