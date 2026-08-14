from app.services.ai.nvidia_client import nvidia_client

class AIRouter:
    """
    AI Router for HireMate.
    Routes AI reasoning tasks to NVIDIA Nemotron 3 Ultra as the primary reasoning engine.
    """
    async def execute_task(self, task_type: str, prompt: str, system_prompt: str = "", reasoning_budget: int = None, max_tokens: int = 16384) -> str:
        """
        Executes an AI text completion task using NVIDIA Nemotron 3 Ultra.
        """
        res = await nvidia_client.generate_completion(prompt, system_prompt, reasoning_budget=reasoning_budget, max_tokens=max_tokens)
        return res or ""

    async def execute_json_task(self, task_type: str, prompt: str, system_prompt: str = "", reasoning_budget: int = None, max_tokens: int = 16384) -> dict | list:
        """
        Executes an AI JSON generation task using NVIDIA Nemotron 3 Ultra.
        """
        res = await nvidia_client.generate_json(prompt, system_prompt, reasoning_budget=reasoning_budget, max_tokens=max_tokens)
        return res if res is not None else {}

ai_router = AIRouter()



