from typing import AsyncIterator
from Providers.open_ai import OpenAIProvider
from Providers.gemeni import GeminiProvider
from SQL.RAG import VectorRAGService
import uuid
from Providers.APIContracts import ChatBotEdits
from urllib.parse import urlparse
from fastapi import Request, HTTPException


class StartUp:

    def __init__(self):
        self.rag = VectorRAGService()
    
    def init_chatbot_settings(self, settings: ChatBotEdits):
        pass

    def init_subscriptions(self):
        pass
    