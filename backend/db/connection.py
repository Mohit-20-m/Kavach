"""Database connection for KAVACH."""
import os
from loguru import logger

async def init_db():
    logger.info("Database initialized")

async def close_db():
    logger.info("Database closed")
