import asyncio
import json
import re
from typing import Optional

import httpx

from app.core.config import settings


class AIServiceError(Exception):
    """Raised when NVIDIA AI service encounters an unrecoverable error."""
    pass


class NVIDIAClient:
    def __init__(self):
        self.api_key = settings.NVIDIA_API_KEY
        self.model = (
            settings.NVIDIA_MODEL
            or "nvidia/nemotron-3-ultra-550b-a55b"
        )
        self._client: Optional[httpx.AsyncClient] = None

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
                max_keepalive_connections=20,
                max_connections=50,
            )

            timeout = httpx.Timeout(
                timeout=90.0,
                connect=10.0,
                read=90.0,
                write=30.0,
                pool=10.0,
            )

            self._client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
            )

        return self._client

    async def generate_completion(
        self,
        prompt: str,
        system_prompt: str = "",
        reasoning_budget: int = None,
        max_tokens: int = 8192,
    ) -> str:

        self.api_key = settings.NVIDIA_API_KEY

        self.model = (
            settings.NVIDIA_MODEL
            or "nvidia/nemotron-3-ultra-550b-a55b"
        )

        if not self.api_key:
            print("[NVIDIAClient Error] NVIDIA_API_KEY is not configured.")
            return ""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        # IMPORTANT:
        # Nemotron 3 Ultra should be called with the standard
        # chat-completions payload. Do not send reasoning_budget
        # or thinking_token_budget parameters.
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": max_tokens,
        }

        client = self._get_client()

        # 503 = NVIDIA service temporarily overloaded.
        # Retry a small number of times with increasing delays.
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):

            try:
                print(
                    f"[NVIDIAClient] Request attempt "
                    f"{attempt}/{max_attempts} "
                    f"using model: {self.model}"
                )

                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                )

                if response.status_code == 200:
                    data = response.json()

                    choices = data.get("choices", [])

                    if not choices:
                        print(
                            "[NVIDIAClient Error] NVIDIA returned "
                            "200 but no choices were present."
                        )
                        return ""

                    message = choices[0].get("message", {})

                    content = message.get("content", "")

                    if content is None:
                        content = ""

                    content = str(content).strip()

                    # Remove reasoning tags if the provider returns them.
                    if "<think>" in content and "</think>" in content:
                        content = re.sub(
                            r"<think>.*?</think>",
                            "",
                            content,
                            flags=re.DOTALL,
                        ).strip()

                    print(
                        f"[NVIDIAClient] Success. "
                        f"Response length: {len(content)}"
                    )

                    return content

                # NVIDIA temporarily overloaded.
                if response.status_code == 503:
                    print(
                        f"[NVIDIAClient] NVIDIA service overloaded "
                        f"(503), attempt {attempt}/{max_attempts}."
                    )

                    if attempt < max_attempts:
                        wait_time = 3 * attempt
                        print(
                            f"[NVIDIAClient] Waiting {wait_time} seconds "
                            f"before retry..."
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    print(
                        "[NVIDIAClient Error] NVIDIA remained overloaded "
                        "after all retry attempts."
                    )
                    return ""

                # Rate limit.
                if response.status_code == 429:
                    print(
                        f"[NVIDIAClient] Rate limited (429), "
                        f"attempt {attempt}/{max_attempts}."
                    )

                    if attempt < max_attempts:
                        wait_time = 5 * attempt
                        await asyncio.sleep(wait_time)
                        continue

                    return ""

                # Authentication/configuration problem.
                if response.status_code in (401, 403):
                    print(
                        f"[NVIDIAClient Error] NVIDIA authentication "
                        f"failed. Status {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                    return ""

                # Bad request.
                if response.status_code == 400:
                    print(
                        f"[NVIDIAClient Error] NVIDIA rejected the "
                        f"request (400): {response.text[:1000]}"
                    )
                    return ""

                # Other server errors.
                print(
                    f"[NVIDIAClient Error] Attempt {attempt} "
                    f"Status {response.status_code}: "
                    f"{response.text[:1000]}"
                )

                if attempt < max_attempts:
                    await asyncio.sleep(2 * attempt)
                    continue

                return ""

            except httpx.ReadTimeout:
                print(
                    f"[NVIDIAClient Timeout] NVIDIA did not respond "
                    f"within the timeout on attempt "
                    f"{attempt}/{max_attempts}."
                )

                if attempt < max_attempts:
                    wait_time = 3 * attempt
                    await asyncio.sleep(wait_time)
                    continue

                print(
                    "[NVIDIAClient Failure] NVIDIA request timed out "
                    "after all retry attempts."
                )
                return ""

            except httpx.ConnectTimeout:
                print(
                    f"[NVIDIAClient Connect Timeout] "
                    f"Could not connect to NVIDIA on attempt "
                    f"{attempt}/{max_attempts}."
                )

                if attempt < max_attempts:
                    await asyncio.sleep(2 * attempt)
                    continue

                return ""

            except httpx.HTTPError as e:
                print(
                    f"[NVIDIAClient HTTP Error] "
                    f"{type(e).__name__}: {e}"
                )

                if attempt < max_attempts:
                    await asyncio.sleep(2 * attempt)
                    continue

                return ""

            except Exception as e:
                print(
                    f"[NVIDIAClient Exception] "
                    f"Attempt {attempt} failed: "
                    f"{type(e).__name__} - {e}"
                )

                if attempt < max_attempts:
                    await asyncio.sleep(2 * attempt)
                    continue

                return ""

        return ""

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        reasoning_budget: int = None,
        max_tokens: int = 8192,
    ) -> dict | list:

        json_system = (
            system_prompt
            + "\n\n"
            "CRITICAL INSTRUCTION: "
            "Return ONLY valid JSON. "
            "The response must be either a JSON object or JSON array. "
            "Do NOT use markdown code fences. "
            "Do NOT add explanations before or after the JSON."
        )

        raw_text = await self.generate_completion(
            prompt=prompt,
            system_prompt=json_system,
            reasoning_budget=reasoning_budget,
            max_tokens=max_tokens,
        )

        clean_text = raw_text.strip()

        if not clean_text:
            print(
                "[NVIDIAClient JSON] NVIDIA returned an empty response."
            )
            return {}

        # Remove markdown fences if NVIDIA accidentally returns them.
        clean_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            clean_text,
            flags=re.IGNORECASE,
        )

        clean_text = re.sub(
            r"\s*```$",
            "",
            clean_text,
        )

        clean_text = clean_text.strip()

        try:
            result = json.loads(clean_text)

            if isinstance(result, (dict, list)):
                return result

            print(
                "[NVIDIAClient JSON Warning] "
                "Valid JSON was returned but it was not an object/array."
            )
            return {}

        except json.JSONDecodeError as e:
            print(
                f"[NVIDIAClient JSON Parse Warning]: {e}. "
                "Attempting to repair response."
            )

        # Try extracting the JSON object/array from surrounding text.
        try:
            first_object = clean_text.find("{")
            first_array = clean_text.find("[")

            positions = [
                p for p in (first_object, first_array)
                if p >= 0
            ]

            if positions:
                start = min(positions)

                last_object = clean_text.rfind("}")
                last_array = clean_text.rfind("]")

                end = max(last_object, last_array)

                if end > start:
                    extracted = clean_text[start:end + 1]

                    result = json.loads(extracted)

                    if isinstance(result, (dict, list)):
                        print(
                            "[NVIDIAClient JSON] "
                            "Successfully extracted JSON "
                            "from surrounding text."
                        )
                        return result

        except Exception as extraction_error:
            print(
                f"[NVIDIAClient JSON Extraction Error]: "
                f"{extraction_error}"
            )

        # Basic repair for truncated JSON.
        try:
            repaired = clean_text

            # Remove trailing comma.
            repaired = re.sub(
                r",\s*([}\]])",
                r"\1",
                repaired,
            )

            # Close missing brackets/braces.
            open_braces = repaired.count("{") - repaired.count("}")
            open_brackets = repaired.count("[") - repaired.count("]")

            if open_brackets > 0:
                repaired += "]" * open_brackets

            if open_braces > 0:
                repaired += "}" * open_braces

            result = json.loads(repaired)

            if isinstance(result, (dict, list)):
                print(
                    "[NVIDIAClient JSON] "
                    "JSON repair succeeded."
                )
                return result

        except Exception as repair_error:
            print(
                f"[NVIDIAClient JSON Repair Error]: "
                f"{repair_error}. "
                f"Raw output: {clean_text[:500]}"
            )

        return {}


nvidia_client = NVIDIAClient()
