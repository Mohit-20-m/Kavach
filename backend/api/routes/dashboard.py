"""
KAVACH Dashboard API Routes
============================
Provides REST endpoints for the real-time dashboard.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# ─── In-memory scan history (last 200 scans) ─────────────────────────────────
# In production this would be a database query.
# This module-level store is populated by the scan route after every scan.

_scan_history: deque = deque(maxlen=200)
_stats = {
    "total_scans": 0,
    "blocked_count": 0,
    "safe_count": 0,
    "caution_count": 0,
    "high_count": 0,
    "critical_count": 0,
}


def record_scan(result: dict):
    """
    Called by the scan route after every completed scan.
    Stores the result in the in-memory history and updates stats.
    """
    entry = {
        "id": int(time.time() * 1000),
        "package_name": result.get("package_name", "unknown"),
        "ecosystem": result.get("ecosystem", "npm"),
        "risk_score": result.get("risk_score", 0.0),
        "risk_tier": result.get("risk_tier", "SAFE"),
        "install_blocked": result.get("install_blocked", False),
        "timestamp": datetime.utcnow().isoformat(),
        "agent_scores": result.get("agent_scores", {}),
        "execution_time_ms": result.get("execution_time_ms", 0),
    }
    _scan_history.appendleft(entry)
    _stats["total_scans"] += 1

    tier = result.get("risk_tier", "SAFE")
    if result.get("install_blocked"):
        _stats["blocked_count"] += 1
    if tier == "SAFE":
        _stats["safe_count"] += 1
    elif tier == "CAUTION":
        _stats["caution_count"] += 1
    elif tier == "HIGH":
        _stats["high_count"] += 1
    elif tier == "CRITICAL":
        _stats["critical_count"] += 1


# ─── Response models ──────────────────────────────────────────────────────────

class ScanEntry(BaseModel):
    id: int
    package_name: str
    ecosystem: str
    risk_score: float
    risk_tier: str
    install_blocked: bool
    timestamp: str
    agent_scores: dict
    execution_time_ms: float


class DashboardResponse(BaseModel):
    scans: list[ScanEntry]
    total_scans: int
    blocked_count: int
    safe_count: int
    caution_count: int
    high_count: int
    critical_count: int
    uptime_seconds: float


_start_time = time.time()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/recent-scans", response_model=DashboardResponse)
async def get_recent_scans(limit: int = 50):
    """
    Return recent scan history for dashboard initialization.
    The dashboard fetches this on mount, then subscribes to WebSocket
    for live updates.
    """
    scans = list(_scan_history)[:limit]
    return DashboardResponse(
        scans=scans,
        total_scans=_stats["total_scans"],
        blocked_count=_stats["blocked_count"],
        safe_count=_stats["safe_count"],
        caution_count=_stats["caution_count"],
        high_count=_stats["high_count"],
        critical_count=_stats["critical_count"],
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/stats")
async def get_stats():
    """Quick stats endpoint — called by dashboard every 30s as a heartbeat."""
    return {
        **_stats,
        "uptime_seconds": round(time.time() - _start_time, 1),
        "history_count": len(_scan_history),
    }


@router.delete("/history")
async def clear_history():
    """Clear scan history — useful for demo resets."""
    _scan_history.clear()
    for k in _stats:
        _stats[k] = 0
    return {"message": "History cleared"}