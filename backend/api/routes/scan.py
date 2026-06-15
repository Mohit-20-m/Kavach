"""
KAVACH API — Scan Routes
"""

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
from loguru import logger
from api.routes.dashboard import record_scan

router = APIRouter()


class ScanRequest(BaseModel):
    package_name: str
    ecosystem: str = "npm"
    version: str = None
    source: str = "cli"  # cli, vscode, cicd, api


class ScanResponse(BaseModel):
    package_name: str
    ecosystem: str
    risk_score: float
    risk_tier: str
    confidence: float
    install_blocked: bool
    plain_english_explanation: str
    evidence_summary: list[dict]
    safe_alternatives: list[dict]
    causal_explanation: dict
    similar_attacks: list[dict]
    execution_time_ms: float
    agent_scores: dict


@router.post("/", response_model=ScanResponse)
async def scan_package(request_data: ScanRequest, request: Request):
    """
    Main scan endpoint — analyze a package for supply chain threats.
    Called by CLI wrapper and VS Code extension.
    """
    orchestrator = request.app.state.orchestrator
    ws_manager = request.app.state.ws_manager

    logger.info(
        f"Scan request: {request_data.package_name} "
        f"({request_data.ecosystem}) from {request_data.source}"
    )

    try:
        verdict = await orchestrator.scan(
            package_name=request_data.package_name,
            ecosystem=request_data.ecosystem,
            ws_manager=ws_manager,
        )

        # Extract agent scores for response
        agent_scores = {}
        ar = verdict.agent_results
        for name, result in [
            ("code_archaeologist", ar.code_archaeologist),
            ("dependency_graph", ar.dependency_graph),
            ("maintainer_trust", ar.maintainer_trust),
            ("behavioral_anomaly", ar.behavioral_anomaly),
            ("semantic_intent", ar.semantic_intent),
        ]:
            if result:
                agent_scores[name] = {
                    "risk_score": result.risk_score,
                    "confidence": result.confidence,
                    "execution_time_ms": result.execution_time_ms,
                }
        record_scan({
            "package_name": verdict.package_name,
            "ecosystem": verdict.ecosystem,
            "risk_score": verdict.risk_score,
            "risk_tier": verdict.risk_tier.value,
            "install_blocked": verdict.install_blocked,
            "execution_time_ms": verdict.execution_time_ms,
            "agent_scores": agent_scores,
        })

        return ScanResponse(
            package_name=verdict.package_name,
            ecosystem=verdict.ecosystem,
            risk_score=verdict.risk_score,
            risk_tier=verdict.risk_tier.value,
            confidence=verdict.confidence,
            install_blocked=verdict.install_blocked,
            plain_english_explanation=verdict.plain_english_explanation,
            evidence_summary=verdict.evidence_summary,
            safe_alternatives=verdict.safe_alternatives,
            causal_explanation=verdict.causal_explanation,
            similar_attacks=verdict.similar_attacks,
            execution_time_ms=verdict.execution_time_ms,
            agent_scores=agent_scores,
        )

    except Exception as e:
        logger.error(f"Scan error for {request_data.package_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_scan_history(request: Request, limit: int = 50, ecosystem: str = None):
    """Get recent scan history for the organization dashboard."""
    # TODO: Fetch from PostgreSQL
    return {"history": [], "total": 0}


@router.get("/stats")
async def get_scan_stats(request: Request):
    """Get aggregate statistics for dashboard."""
    return {
        "total_scans": 0,
        "blocked_packages": 0,
        "critical_threats": 0,
        "safe_packages": 0,
    }
