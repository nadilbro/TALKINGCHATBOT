from pydantic import BaseModel
from typing import Optional 
from datetime import datetime
class ChatRequest(BaseModel):
    site_id: str
    message: str

class ChatMessageStructure(BaseModel):   
    context: str
    userQ: str
    
class PaymentStructure(BaseModel):
    site_id: Optional[str]
    status: Optional[str]
    trial_end: Optional[str]
    current_end: Optional[str]

class ChatBotEdits(BaseModel):
    site_id: str
    chatbot_name: Optional[str] = None
    personality: Optional[str] = None
    tone: Optional[str] = None
    resp_length: Optional[str] = None
    temperature: Optional[float] = None
    greeting: Optional[str] = None
    fallback: Optional[str] = None
    widget_color: Optional[str] = None
    widget_size: Optional[str] = None
    border_radius: Optional[str] = None
    updated_at: Optional[str] = None

class ClientListSetUp(BaseModel):
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    address: Optional[str]
    country: Optional[str]
    subscription: str
    account_id: str
    subscription_end: datetime | None = None
    created_at: datetime | None = None  # Make this optional


class SiteID(BaseModel):
    site_id: str

class AvatarStructure(BaseModel):
    site_id: str
    AvatarUrl: Optional[str]
    updated_at: str

class ChatResponse(BaseModel):
    site_id: str
    reply: str

class AddDataRequest(BaseModel):
    site_id: str
    source: str
    text: str
class DeleteEmbeddingRequest(BaseModel):
    site_id: str
    chunk_index: int
class EmbeddingRow(BaseModel):
    id: int
    site_id: str
    source: Optional[str] = None
    chunk_index: int
    data: str
    created_at: Optional[str] = None
    
class GetDataRequest(BaseModel):
    site_id: str
class StatusResponse(BaseModel):
    status: str

class allowed_websites(BaseModel):
    site_id: str
    domain: str