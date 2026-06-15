"""KAVACH Health Routes"""
from fastapi import APIRouter, Request
router = APIRouter()

@router.get("/")
async def health_check():
    return {"status": "healthy", "service": "KAVACH", "version": "1.0.0"}

@router.get("/agents")
async def agent_health(request: Request):
    return {
        "agents": {
            "code_archaeologist": "ready",
            "dependency_graph": "ready",
            "maintainer_trust": "ready",
            "behavioral_anomaly": "ready",
            "semantic_intent": "ready",
        }
    }
