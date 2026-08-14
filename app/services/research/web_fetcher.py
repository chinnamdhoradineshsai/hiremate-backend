import httpx
from bs4 import BeautifulSoup

class WebFetcher:
    async def fetch_page_content(self, url: str, max_chars: int = 2500) -> str:
        """
        Fetches webpage HTML and extracts readable main body text content.
        Safely limits output size and ignores irrelevant tags (script, style, nav).
        """
        if not url or not url.startswith("http"):
            return ""

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        try:
            async with httpx.AsyncClient(timeout=8.0, headers=headers, follow_redirects=True) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return ""

                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Remove non-content elements
                for element in soup(["script", "style", "nav", "footer", "header", "svg", "form"]):
                    element.extract()

                paragraphs = [p.get_text(strip=True) for p in soup.find_all(["p", "h1", "h2", "h3", "li"]) if len(p.get_text(strip=True)) > 20]
                text = "\n".join(paragraphs)
                
                return text[:max_chars]
        except Exception as e:
            print(f"[WebFetcher Warning] Could not fetch {url}: {e}")
            return ""

web_fetcher = WebFetcher()
