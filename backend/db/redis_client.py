"""Redis client for KAVACH caching layer."""
import os
import redis.asyncio as aioredis
from loguru import logger

_redis_client = None

async def init_redis():
    global _redis_client
    url = os.getenv("REDIS_URL", "redis://localhost:6379")
    _redis_client = aioredis.from_url(url, decode_responses=True)
    logger.info("Redis connected")

async def get_redis():
    return _redis_client

async def close_redis():
    if _redis_client:
        await _redis_client.close()
