from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime


class ChatRequest(BaseModel):
    site_id: str
    message: str
    pastMessages: List[str] 
    pastAnswers: List[str] 

class ChatMessageStructure(BaseModel):   
    context: str
    userQ: str

class ChatBotEdits(BaseModel):
    user_id: str
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    monthly_token_limit = Optional[str]
    monthly_token_used = Optional[str]  

class AccountInit(BaseModel):
    user_id: str
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    created_at = Optional[str]
    subscription_status = Optional[str]
    stripe_customer_id = Optional[str]
    stripe_subscription_id = Optional[str]
    monthly_token_limit = Optional[str]
    monthly_token_used = Optional[str]  
    billing_cycle_start = Optional[str]

class SessionInit(BaseModel):
    userID: str
    chat_id: str

class SessionCreate(BaseModel):
    id: str
    user_id: str
    created_at: Optional[str]
    rive_avatar: Optional[str]
    last_message: Optional[str]
    welcome_message: Optional[str]
    status: Optional[str] = 'Open'
    summary: Optional[str]
    title: Optional[str]



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


class UserID(BaseModel):
    user_id: str

