"""Per-user Token Bucket rate limiter (in-process; swap for Redis when horizontally scaled)."""
import asyncio
import time
from dataclasses import dataclass


class RateLimited(Exception):
    def __init__(self, retry_after: float):
        super().__init__(f"Rate limit exceeded. Retry in {retry_after:.1f}s")
        self.retry_after = retry_after


@dataclass
class TokenBucket:
    capacity: float
    refill_per_sec: float
    tokens: float = 0.0
    updated: float = 0.0

    def __post_init__(self):
        self.tokens = self.capacity
        self.updated = time.monotonic()

    def take(self, cost: float = 1.0) -> float:
        """Returns 0 if allowed, otherwise seconds until `cost` tokens are available."""
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill_per_sec)
        self.updated = now
        if self.tokens >= cost:
            self.tokens -= cost
            return 0.0
        return (cost - self.tokens) / self.refill_per_sec


class RateLimiter:
    def __init__(self, capacity_by_role: dict[str, int], refill_per_sec: float):
        self.capacity_by_role = capacity_by_role
        self.refill = refill_per_sec
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, username: str, role: str, cost: float = 1.0) -> None:
        async with self._lock:
            bucket = self._buckets.setdefault(
                username, TokenBucket(self.capacity_by_role.get(role, 5), self.refill)
            )
            wait = bucket.take(cost)
        if wait > 0:
            raise RateLimited(wait)
