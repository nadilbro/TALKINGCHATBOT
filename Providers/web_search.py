from tavily import TavilyClient
import os
from typing import List, Dict, Any, Tuple

class TavilyProvider: 
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        self.client = TavilyClient(api_key=api_key)

    def web_search(self, question: str, max_results: int = 5) -> str:
        try: 
            response = self.client.search(
                query=question,
                search_depth="basic",
                max_results=max_results
            )
        except Exception as e:
            print(f"Tavily search failed: {e}")
            return ""
        
        results = response.get("results", [])
        if not results:
            return ""

        lines = [f'Web search results for "{question}":']
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            content = r.get("content", "No content")
            lines.append(f"{i}. {title} — {content}")
 
        lines.append("")
        lines.append("Use these results to inform your answer if relevant. If the results are not useful, ignore them and answer normally.")
 
        return "\n".join(lines)
    
        