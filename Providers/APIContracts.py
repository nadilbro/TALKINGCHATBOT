from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime


class TextIngestionRequest(BaseModel):
    api_key: str
    content: str
    title: str = "Manual Entry"

    
class ChatRequest(BaseModel):
    user_id: str
    message: str
    pastMessages: Optional[List[str]] = None
    pastAnswers: Optional[List[str]] = None

class ChatMessageStructure(BaseModel):   
    user_id: str
    context: str
    userQ: str

class ChatBotEdits(BaseModel):
    user_id: str
    name: Optional[str] = None 
    email: Optional[str] = None
    phone: Optional[str] = None
    monthly_token_limit: Optional[str] = None
    monthly_token_used: Optional[str] = None 

class AccountInit(BaseModel):
    user_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    created_at: Optional[str] = None
    subscription_status: Optional[str] = None
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    monthly_token_limit: Optional[str] = None
    monthly_token_used: Optional[str] = None
    billing_cycle_start: Optional[str] = None

class SessionInit(BaseModel):
    userID: str
    chat_id: str

class AccountCreate(BaseModel):
    user_id: str
    email: str | None = None
    name: str | None = None
    phone: str | None = None


class SessionCreate(BaseModel):
    user_id: str
    title: Optional[str] = None
    avatar_name: Optional[str] = None

class SessionDelete(BaseModel):
    id: str
    user_id: str

class SessionBase(BaseModel):
    id: str
    user_id: str
    created_at: Optional[str] = None
    rive_avatar: Optional[str] = None
    last_message: Optional[str] = None
    welcome_message: Optional[str] = None
    status: Optional[str] = 'Open'
    summary: Optional[str] = None
    title: Optional[str] = None



class VoiceChat(BaseModel):
    # init response fields
    site_id: Optional[str] = None
    rive_url: Optional[str] = None
    voice_name: Optional[str] = None
    primary_color: Optional[str] = None
    welcome_message: Optional[str] = None

    # chat response fields
    text: Optional[str] = None
    audio_bytes: Optional[str] = None   # base64 wav
    visemes: Optional[List[Dict]] = None

class DiagramInit(BaseModel):
    chat_id: str

class ClientListSetUp(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    country: Optional[str] = None
    subscription: str
    account_id: str
    subscription_end: datetime | None = None
    created_at: datetime | None = None  # Make this optional


class UserID(BaseModel):
    user_id: str



class toggleDiagram(BaseModel):
    toggle: bool