import asyncio
import httpx
from typing import List, Dict, Any
from app.core.config import settings

class TavilyClientError(Exception):
    """Raised when Tavily API key is missing or request fails."""
    pass

class TavilyResearchClient:
    def __init__(self):
        self.api_key = settings.TAVILY_API_KEY
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
            timeout = httpx.Timeout(20.0, connect=5.0)
            self._client = httpx.AsyncClient(limits=limits, timeout=timeout)
        return self._client

    async def search(self, query: str, max_results: int = 5, search_depth: str = "advanced") -> List[Dict[str, Any]]:
        self.api_key = settings.TAVILY_API_KEY
        if not self.api_key:
            print("[TavilyClient Warning] TAVILY_API_KEY is not set.")
            return []

        # Try using tavily-python SDK first via threadpool to avoid blocking FastAPI event loop
        try:
            from tavily import TavilyClient
            t_client = TavilyClient(api_key=self.api_key)
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    t_client.search,
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth
                ),
                timeout=20.0
            )
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0)
                })
            if results:
                return results
        except ImportError:
            pass
        except Exception as e:
            print(f"[TavilyClient SDK Warning] SDK call failed ({e}). Falling back to REST API.")

        # REST API fallback
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth
        }

        max_retries = 2
        backoff = 0.5
        for attempt in range(max_retries + 1):
            try:
                client = self._get_client()
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    results = []
                    for item in data.get("results", []):
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "content": item.get("content", ""),
                            "score": item.get("score", 0.0)
                        })
                    return results
                else:
                    print(f"[TavilyClient REST Error] Attempt {attempt+1} Status {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"[TavilyClient REST Exception] Attempt {attempt+1} Request failed: {e}")

            if attempt < max_retries:
                import asyncio
                await asyncio.sleep(backoff)
                backoff *= 2.0

        return []

tavily_client = TavilyResearchClient()
