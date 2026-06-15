"""KAVACH CI/CD Pipeline Routes"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
router = APIRouter()

class CICDScanRequest(BaseModel):
    packages: list[dict]  # [{"name": "pkg", "version": "1.0.0", "ecosystem": "npm"}]
    fail_on_critical: bool = True
    fail_on_high: bool = False

@router.post("/scan-lockfile")
async def scan_lockfile(request_data: CICDScanRequest, request: Request):
    """Scan all packages in a lockfile — used in CI/CD pipelines."""
    orchestrator = request.app.state.orchestrator
    results = []
    blocked = []

    for pkg in request_data.packages[:50]:  # Cap at 50 per CI run
        verdict = await orchestrator.scan(
            package_name=pkg["name"],
            ecosystem=pkg.get("ecosystem", "npm"),
        )
        results.append({
            "package": pkg["name"],
            "risk_tier": verdict.risk_tier.value,
            "risk_score": verdict.risk_score,
            "blocked": verdict.install_blocked,
        })
        if verdict.install_blocked:
            blocked.append(pkg["name"])

    pipeline_failed = bool(blocked) and request_data.fail_on_critical

    return {
        "pipeline_status": "FAILED" if pipeline_failed else "PASSED",
        "total_scanned": len(results),
        "blocked_packages": blocked,
        "results": results,
    }
