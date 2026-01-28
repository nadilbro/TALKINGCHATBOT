from typing import AsyncIterator
from Providers.open_ai import OpenAIProvider
from Providers.gemeni import GeminiProvider
from SQL.RAG import VectorRAGService
import uuid
from Providers.APIContracts import ChatBotEdits

class StartUp:

    def __init__(self):
        self.rag = VectorRAGService()

    def init_site_id(self, username: str) -> str:
        site_id = uuid.uuid4().hex
        #Check if it already exists 
        while self.rag.check_exists("site_id", site_id, "client_list"):
            site_id = uuid.uuid4().hex #Makes sure there is no duplicate site_id
        
    def init_chatbot_settings(self, settings: ChatBotEdits):
        pass

    def init_subscriptions(self):
        pass