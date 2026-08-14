import time
import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Any, Optional
from fastapi import HTTPException, Request, status

class RateLimiterBackend(ABC):
    @abstractmethod
    async def is_allowed(self, identifier: str, max_requests: int, window_seconds: int) -> bool:
        pass

class InMemoryRateLimiterBackend(RateLimiterBackend):
    def __init__(self):
        self._history: Dict[str, List[float]] = defaultdict(list)

    async def is_allowed(self, identifier: str, max_requests: int, window_seconds: int) -> bool:
        now = time.time()
        window_start = now - window_seconds
        timestamps = [t for t in self._history[identifier] if t > window_start]
        if len(timestamps) >= max_requests:
            return False
        timestamps.append(now)
        self._history[identifier] = timestamps
        return True

class RedisRateLimiterBackend(RateLimiterBackend):
    """Pluggable Redis backend for distributed 1,000-user deployments."""
    def __init__(self, redis_client: Optional[Any] = None):
        self.redis = redis_client

    async def is_allowed(self, identifier: str, max_requests: int, window_seconds: int) -> bool:
        if not self.redis:
            # Fallback if Redis instance not configured
            return True
        key = f"ratelimit:{identifier}"
        try:
            current = await self.redis.incr(key)
            if current == 1:
                await self.redis.expire(key, window_seconds)
            return current <= max_requests
        except Exception:
            return True

class PluggableRateLimiter:
    def __init__(self, requests_per_minute: int = 60, backend: Optional[RateLimiterBackend] = None):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        self.backend = backend or InMemoryRateLimiterBackend()

    async def check_rate_limit(self, identifier: str):
        allowed = await self.backend.is_allowed(identifier, self.requests_per_minute, self.window_seconds)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Too many requests. Please wait before retrying."
            )

limiter = PluggableRateLimiter(requests_per_minute=60)
ai_limiter = PluggableRateLimiter(requests_per_minute=20)

async def rate_limit_dependency(request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    await limiter.check_rate_limit(client_ip)

async def rate_limit_expensive_ai(request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    await ai_limiter.check_rate_limit(f"ai:{client_ip}")

class RequestDeduplicator:
    """
    Deduplicates expensive parallel AI generation & Tavily research requests in flight.
    If multiple requests for the same key arrive simultaneously, only 1 external call is made.
    """
    def __init__(self):
        self._in_flight: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def execute_or_await(self, key: str, coroutine_fn):
        is_leader = False
        async with self._lock:
            if key in self._in_flight:
                fut = self._in_flight[key]
            else:
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                self._in_flight[key] = fut
                is_leader = True

        if not is_leader:
            return await fut

        try:
            result = await coroutine_fn()
            if not fut.done():
                fut.set_result(result)
            return result
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
            raise e
        finally:
            async with self._lock:
                self._in_flight.pop(key, None)

request_deduplicator = RequestDeduplicator()

