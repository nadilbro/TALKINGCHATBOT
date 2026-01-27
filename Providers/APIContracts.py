from pydantic import BaseModel
from typing import Optional 

class ChatRequest(BaseModel):
    site_id: str
    message: str

class ChatMessageStructure(BaseModel):   
    context: str
    userQ: str
    


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