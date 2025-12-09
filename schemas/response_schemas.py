from typing import Generic, TypeVar, Optional
from pydantic import BaseModel, Field

T = TypeVar("T")


class BaseResponse(BaseModel, Generic[T]):
    status: str
    message: Optional[str] = None  # Describes the status or any additional information
    data: Optional[T] = None  # Generic field for response data 


class AgentMessage(BaseModel):
    speaker_id: str
    text: str
    audio_base64: str


class AgentItem(BaseModel):
    agent_id: str
    model_name: str


class AgentList(BaseModel):
    total: int
    agents: list[AgentItem]


class ModelItem(BaseModel):
    model_name: str
    provider: str = Field(..., pattern=r'^(gemini|hf_serverless)$')
