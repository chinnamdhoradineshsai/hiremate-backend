from typing import List, Dict, Any
from app.services.research.tavily_client import tavily_client

class SearchProvider:
    """
    Unified Web Search Provider for HireMate.
    Powered exclusively by Tavily Deep Research Gateway.
    """
    async def search(
        self,
        query: str,
        max_results: int = 6,
        deep_research: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Executes search using Tavily as the web research layer.
        Cleans and deduplicates results by URL.
        """
        results: List[Dict[str, Any]] = []
        seen_urls = set()

        search_depth = "advanced" if deep_research else "basic"
        try:
            tavily_results = await tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth
            )
        except Exception as e:
            print(f"[SearchProvider Warning] Tavily search failed or timed out: {e}")
            tavily_results = []

        for item in tavily_results:
            url = item.get("url", "").strip()
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "snippet": item.get("content", "") or item.get("snippet", ""),
                    "engine": "tavily"
                })

        return results[:max_results]

search_provider = SearchProvider()


