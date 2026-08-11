from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph import Command
from .graph.graph import graph
from .memory.checkpointer import conversation_checkpointer

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
        command = Command(input={"message": request.message}, resume=request.resume_token)
        result = graph.run(
            state={"thread_id": request.thread_id, "user_id": request.user_id},
            command=command,
            checkpointer=conversation_checkpointer,
        )
        return ChatResponse(
            thread_id=request.thread_id,
            user_id=request.user_id,
            response=str(result.output),
            resume_token=result.resume_token,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
