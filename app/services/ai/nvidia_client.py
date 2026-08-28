````python
import httpx
import json
from app.core.config import settings


class AIServiceError(Exception):
    """Raised when NVIDIA AI service encounters an unrecoverable error or missing credentials."""
    pass


class NVIDIAClient:
    def __init__(self):
        self.api_key = settings.NVIDIA_API_KEY
        self.model = settings.NVIDIA_MODEL or "nvidia/nemotron-3-ultra-550b-a55b"
        self._client: httpx.AsyncClient | None = None

    @property
    def endpoint(self) -> str:
        base_url = (
            settings.NVIDIA_API_BASE_URL
            or "https://integrate.api.nvidia.com/v1"
        ).rstrip("/")
        return f"{base_url}/chat/completions"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(
                max_keepalive_connections=50,
                max_connections=200
            )
            timeout = httpx.Timeout(
                120.0,
                connect=10.0
            )
            self._client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout
            )
        return self._client

    async def generate_completion(
        self,
        prompt: str,
        system_prompt: str = "",
        reasoning_budget: int = None,
        max_tokens: int = 16384
    ) -> str:

        self.api_key = settings.NVIDIA_API_KEY
        self.model = (
            settings.NVIDIA_MODEL
            or "nvidia/nemotron-3-ultra-550b-a55b"
        )

        if not self.api_key:
            print("[NVIDIAClient Warning] NVIDIA_API_KEY is not set.")
            return ""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # Standard NVIDIA chat completion payload.
        # Nemotron 3 Ultra V2 does not support
        # reasoning_budget / thinking_token_budget.
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": max_tokens
        }

        max_retries = 2
        backoff = 0.5
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                client = self._get_client()

                resp = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload
                )

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])

                    if choices:
                        content = (
                            choices[0]
                            .get("message", {})
                            .get("content", "")
                            .strip()
                        )

                        # Strip internal reasoning trace tags if present
                        if "<think>" in content and "</think>" in content:
                            import re

                            content = re.sub(
                                r"<think>.*?</think>",
                                "",
                                content,
                                flags=re.DOTALL
                            ).strip()

                        return content

                    return ""

                else:
                    print(
                        f"[NVIDIAClient Error] "
                        f"Attempt {attempt + 1} "
                        f"Status {resp.status_code}: {resp.text}"
                    )

                    last_exception = Exception(
                        f"HTTP {resp.status_code}: {resp.text}"
                    )

            except Exception as e:
                print(
                    f"[NVIDIAClient Exception] "
                    f"Attempt {attempt + 1} failed: "
                    f"{type(e).__name__} - {e}"
                )

                last_exception = e

            if attempt < max_retries:
                import asyncio

                await asyncio.sleep(backoff)
                backoff *= 2.0

        print(
            f"[NVIDIAClient Failure] "
            f"Exhausted retries: "
            f"{type(last_exception).__name__} - "
            f"{last_exception}"
        )

        return ""

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        reasoning_budget: int = None,
        max_tokens: int = 16384
    ) -> dict | list:

        json_system = (
            system_prompt
            + "\nCRITICAL: Return strictly valid JSON array or object ONLY. "
            "Do NOT use markdown code fences, headers, or conversational text."
        )

        raw_text = await self.generate_completion(
            prompt,
            json_system,
            reasoning_budget=reasoning_budget,
            max_tokens=max_tokens
        )

        clean_text = raw_text.strip()

        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]

        if clean_text.startswith("```"):
            clean_text = clean_text[3:]

        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]

        clean_text = clean_text.strip()

        if not clean_text:
            return {}

        try:
            return json.loads(clean_text)

        except Exception as e:
            print(
                f"[NVIDIAClient JSON Parse Warning]: "
                f"{e}. Attempting JSON repair..."
            )

            # Attempt repairing truncated JSON string
            try:
                import re

                in_quote = (
                    clean_text.count('"')
                    - clean_text.count('\\"')
                ) % 2 == 1

                repaired = (
                    clean_text + '"'
                    if in_quote
                    else clean_text
                )

                repaired = re.sub(
                    r",\s*$",
                    "",
                    repaired
                )

                open_braces = (
                    repaired.count("{")
                    - repaired.count("}")
                )

                open_brackets = (
                    repaired.count("[")
                    - repaired.count("]")
                )

                if open_brackets > 0:
                    repaired += "]" * open_brackets

                if open_braces > 0:
                    repaired += "}" * open_braces

                return json.loads(repaired)

            except Exception as repair_err:
                print(
                    f"[NVIDIAClient JSON Repair Error]: "
                    f"{repair_err}. "
                    f"Raw output snippet: "
                    f"{clean_text[:150]}"
                )

                return {}


nvidia_client = NVIDIAClient()
````


