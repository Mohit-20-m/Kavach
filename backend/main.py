"""
KAVACH — कवच
Main FastAPI Application Entry Point
"""

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import make_asgi_app, Counter, Histogram
from loguru import logger

from api.routes import scan, dashboard, cicd, health
from core.orchestrator import KavachOrchestrator
from db.connection import init_db, close_db
from db.redis_client import init_redis, close_redis
from utils.model_loader import ModelLoader


# ─── Prometheus Metrics ──────────────────────────────────────────────────────
SCAN_REQUESTS = Counter("kavach_scan_requests_total", "Total package scan requests")
SCAN_LATENCY = Histogram("kavach_scan_latency_seconds", "Package scan latency")
BLOCKED_PACKAGES = Counter("kavach_blocked_packages_total", "Total packages blocked")


# ─── App Lifespan ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager."""
    logger.info("🛡️  KAVACH starting up...")

    # Initialize database connections
    await init_db()
    await init_redis()

    # Pre-load all ML models into memory (eliminates per-request load latency)
    logger.info("Loading ML models into memory...")
    app.state.model_loader = ModelLoader()
    await app.state.model_loader.load_all()

    # Initialize the main orchestrator and load all agent models
    app.state.orchestrator = KavachOrchestrator(app.state.model_loader)
    await app.state.orchestrator.initialize()

    logger.info("✅ KAVACH ready — all agents armed")
    yield

    # Shutdown
    logger.info("KAVACH shutting down...")
    await close_db()
    await close_redis()


# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="KAVACH — कवच",
    description="Agentic Behavioral Shield for Open Source Supply Chain Security",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ─── Middleware ───────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "vscode-webview://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ─── Routes ──────────────────────────────────────────────────────────────────
app.include_router(scan.router, prefix="/api/v1/scan", tags=["Scan"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
app.include_router(cicd.router, prefix="/api/v1/cicd", tags=["CI/CD"])
app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])


# ─── WebSocket for Real-time Updates ─────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws/scan-updates")
async def websocket_scan_updates(websocket: WebSocket):
    """
    WebSocket endpoint for real-time scan progress updates.
    Frontend connects here to receive live agent results as they complete.
    """
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Make manager accessible to routes
app.state.ws_manager = manager
