from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length = 1,
        max_length=10000,
        description="The user's message to the agent"
    ),

    thread_id: str = Field(
        default="default",
        description="Conversation thread ID"
    )

class ChatResponse(BaseModel):
    response: str
    model_used: str
    cached: bool=False
    processing_time_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc))