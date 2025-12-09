from pydantic import BaseModel, Field
from typing import Optional


class ConversationHistoryItem(BaseModel):
    role: str = Field(pattern=r'^(user|model|system)$') # 'user' or 'model'
    content: str  # the message itself

# Define the request body for the frontend
class ConversationRequest(BaseModel):
    # The entire history of the chat
    conversation_history: list[ConversationHistoryItem] 
    # The agent that is expected to speak next
    next_speaker_id: str 
    user_prompt: str


# Request body to create a new agent
class AgentCreateRequest(BaseModel):
    agent_id: str
    model_name: str 


class AgentUpdateRequest(BaseModel):
    model_name: Optional[str] = None