from pydantic import BaseModel
from typing import Optional 

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
    chabot_name: Optional[str]
    personality: Optional[str]
    tone: Optional[str]
    resp_length: Optional[str]
    temperature: Optional[float]
    greeting: Optional[str]
    fallback: Optional[str]
    updated_at: Optional[str] = None

class WidgetAppearance(BaseModel):
    site_id: str
    widget_color: Optional[str]
    widget_size: Optional[str]
    border_radius: Optional[int]
    updated_at: Optional[str] = None

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

class DeleteDataRequest(BaseModel):
    site_id: Optional[str]
    source: Optional[str]

class StatusResponse(BaseModel):
    status: str

class allowed_websites(BaseModel):
    site_id: str
    domain: str