from tavily import TavilyClient
import os
import base64
import re
import httpx
import asyncio
from typing import List, Dict, Any, Tuple

class TavilyProvider: 
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        self.tavily_client = TavilyClient(api_key=api_key)

    def web_search(self, question: str, max_results: int = 3):
        try: 
            response = self.client.search(
                query=question,
                search_depth="basic",
                max_results=max_results
            )
        except Exception as e:
            print(f"Tavily search failed: {e}")
            return ""
        
        lines = []
        for i, r in enumerate(response, 1):
            title = r.get("title", "No title")
            content = r.get("content", "No content")
            lines.append(f"{i}. {title} — {content}")
 
        lines.append("")
        lines.append("Use these results to inform your answer if relevant. If the results are not useful, ignore them and answer normally.")
 
        return "\n".join(lines)
    
    def inject_search_context(system_prompt: str, search_context: str) -> str:
        """
        Adds search results to the end of Mia's system prompt.
        If search_context is empty, returns the prompt unchanged.
        """
        if not search_context:
            return system_prompt
    
        return f"{system_prompt}\n\n## Web context\n\n{search_context}"