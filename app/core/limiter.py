import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests=5, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def allow(self, ip: str, now: float) -> bool:
        self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window_seconds]
        return len(self.requests[ip]) < self.max_requests

    def record(self, ip: str, now: float):
        self.requests[ip].append(now)

rate_limiter = RateLimiter()
