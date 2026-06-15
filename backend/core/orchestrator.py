"""
KAVACH Core Orchestrator
========================
LangGraph-based orchestrator that runs all 5 agents in parallel,
aggregates results with a trained meta-learner, and produces
a final risk verdict with causal explanation.
"""

import asyncio
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import joblib
import numpy as np
from loguru import logger

from agents.agent1_code_archaeologist import CodeArchaeologist, CodeArchaeologistResult
from agents.agent2_dependency_graph import DependencyGraphAnalyst, DependencyGraphResult
from agents.agent3_maintainer_trust import MaintainerTrustProfiler, MaintainerTrustResult
from agents.agent4_behavioral_anomaly import BehavioralAnomalyDetector, BehavioralAnomalyResult
from agents.agent5_semantic_intent import SemanticIntentAnalyzer, SemanticIntentResult
from core.explainability import ExplainabilityEngine
from core.rag_engine import RAGEngine
from db.redis_client import get_redis


class RiskTier(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class AgentResults:
    code_archaeologist: Optional[CodeArchaeologistResult] = None
    dependency_graph: Optional[DependencyGraphResult] = None
    maintainer_trust: Optional[MaintainerTrustResult] = None
    behavioral_anomaly: Optional[BehavioralAnomalyResult] = None
    semantic_intent: Optional[SemanticIntentResult] = None


@dataclass
class KavachVerdict:
    package_name: str
    ecosystem: str
    risk_score: float
    risk_tier: RiskTier
    confidence: float
    agent_results: AgentResults
    causal_explanation: dict
    plain_english_explanation: str
    similar_attacks: list[dict]
    safe_alternatives: list[dict]
    execution_time_ms: float
    install_blocked: bool
    evidence_summary: list[dict] = field(default_factory=list)


class KavachOrchestrator:
    """
    Central orchestrator — runs all 5 agents in parallel via asyncio,
    aggregates with meta-learner, generates causal explanation.
    """

    # Risk tier thresholds — calibrated to the trained meta-learner's output space.
    # The meta-learner (logistic regression) outputs probabilities where:
    #   - Legitimate packages land in the 0.001–0.10 range
    #   - Borderline packages land in 0.10–0.50 range
    #   - Clearly malicious packages land in 0.75–0.999 range
    SAFE_THRESHOLD = 0.20
    CAUTION_THRESHOLD = 0.50
    HIGH_THRESHOLD = 0.75

    # Cache TTL seconds
    CACHE_TTL = 3600 * 6  # 6 hours

    def __init__(self, model_loader=None):
        self.model_loader = model_loader

        # Initialize agents
        self.agent1 = CodeArchaeologist()
        self.agent2 = DependencyGraphAnalyst()
        self.agent3 = MaintainerTrustProfiler(
            github_token=os.getenv("GITHUB_TOKEN")
        )
        self.agent4 = BehavioralAnomalyDetector()
        self.agent5 = SemanticIntentAnalyzer()

        # Initialize support systems
        self.explainability = ExplainabilityEngine()
        self.rag_engine = RAGEngine()

        # Trained meta-learner (loaded by _load_meta_learner).
        # Logistic regression trained on 11,287 real packages.
        # Coefficient magnitudes: code_arch +3.79, dep_graph +3.73,
        # maint_trust -0.36, behav_anom +0.09, semantic +0.99.
        # The near-zero weight on behavioral_anomaly correctly discounts
        # its noisy LSTM/IF outputs that produce high false-positive rates.
        self._meta_learner = None
        self._meta_scaler  = None

        # Fallback manual weights — used only when meta-learner files are absent.
        # Mirrors the logistic regression coefficient sign/magnitude.
        self._meta_weights = {
            "code_archaeologist": 0.35,
            "dependency_graph":   0.35,
            "maintainer_trust":   0.05,
            "behavioral_anomaly": 0.10,
            "semantic_intent":    0.15,
        }

        self._confidence_floor = 0.1

        # Load meta-learner immediately (sync — just joblib.load)
        # so it's available even when initialize() is never awaited.
        self._load_meta_learner()

    async def initialize(self):
        """Load all models into memory."""
        logger.info("Loading all agent models...")
        self.agent1.load_model()
        self.agent2.load_model()
        self.agent3.load_model()
        self.agent4.load_model()
        self.agent5.load_model()
        self._load_meta_learner()
        await self.rag_engine.initialize()
        logger.info("All agents armed and ready")

    def _load_meta_learner(self):
        """Load trained meta-learner and its feature scaler.

        Tries multiple path strategies so the orchestrator works in:
        - Docker (volume-mounted: /app/models/)
        - Local dev with symlink (backend/models/ → ../data/models/)
        - Local dev without symlink (project-root/data/models/)
        """
        # Candidate paths — first match wins
        core_dir    = os.path.dirname(os.path.abspath(__file__))   # backend/core/
        backend_dir = os.path.dirname(core_dir)                     # backend/
        project_dir = os.path.dirname(backend_dir)                  # project root

        candidates = [
            os.path.join("models", "meta_learner.pkl"),             # Docker / symlink (cwd-relative)
            os.path.join(backend_dir, "models", "meta_learner.pkl"),# backend/models/
            os.path.join(project_dir, "data", "models", "meta_learner.pkl"),  # data/models/
        ]
        scaler_candidates = [c.replace("meta_learner.pkl", "meta_learner_scaler.pkl") for c in candidates]

        ml_path = next((p for p in candidates        if os.path.exists(p)), None)
        sc_path = next((p for p in scaler_candidates if os.path.exists(p)), None)

        if ml_path and sc_path:
            try:
                self._meta_learner = joblib.load(ml_path)
                self._meta_scaler  = joblib.load(sc_path)
                logger.info(f"Trained meta-learner loaded from {ml_path}")
            except Exception as exc:
                logger.warning(f"Meta-learner load failed ({exc}) — using fallback weights")
        else:
            logger.warning("Meta-learner files not found — using fallback weighted average")

    async def scan(
        self, package_name: str, ecosystem: str = "npm",
        ws_manager=None
    ) -> KavachVerdict:
        """
        Main scan entry point.
        Implements tiered analysis for latency optimization.
        """
        start = time.time()

        # ── Tier 0: Cache Check ──────────────────────────────────────────────
        cached = await self._check_cache(package_name, ecosystem)
        if cached:
            logger.info(f"Cache hit for {package_name}")
            return cached

        # ── Broadcast scan started ───────────────────────────────────────────
        if ws_manager:
            await ws_manager.broadcast({
                "type": "scan_started",
                "package": package_name,
                "ecosystem": ecosystem,
            })

        # ── Tier 1: Fast check (API-only agents) ─────────────────────────────
        tier1_results = await self._tier1_fast_check(package_name, ecosystem)

        # If tier1 is clearly safe with high confidence, return early
        if self._is_clearly_safe(tier1_results):
            verdict = await self._build_verdict(
                package_name, ecosystem, tier1_results,
                AgentResults(), start, fast_mode=True
            )
            await self._cache_result(package_name, ecosystem, verdict)
            return verdict

        # ── Broadcast tier1 results ───────────────────────────────────────────
        if ws_manager:
            await ws_manager.broadcast({
                "type": "tier1_complete",
                "package": package_name,
                "preliminary_risk": tier1_results.get("preliminary_score", 0),
            })

        # ── Tier 2: Full parallel analysis ───────────────────────────────────
        agent_results = await self._run_all_agents_parallel(
            package_name, ecosystem, ws_manager
        )

        # ── Build final verdict ───────────────────────────────────────────────
        verdict = await self._build_verdict(
            package_name, ecosystem, tier1_results,
            agent_results, start, fast_mode=False
        )

        # ── Cache result ─────────────────────────────────────────────────────
        await self._cache_result(package_name, ecosystem, verdict)

        # ── Broadcast final verdict ───────────────────────────────────────────
        if ws_manager:
            await ws_manager.broadcast({
                "type": "scan_complete",
                "package": package_name,
                "risk_tier": verdict.risk_tier,
                "risk_score": verdict.risk_score,
                "install_blocked": verdict.install_blocked,
            })

        return verdict

    async def _tier1_fast_check(
        self, package_name: str, ecosystem: str
    ) -> dict:
        """
        Fast tier-1 check using only API-based agents (no code download).
        Returns within ~500ms.
        """
        # Run agents 3 and 4 in parallel (fastest — API calls only)
        results3, results4 = await asyncio.gather(
            self.agent3.analyze(package_name, ecosystem),
            self.agent4.analyze(package_name, ecosystem),
            return_exceptions=True,
        )

        scores = []
        if isinstance(results3, MaintainerTrustResult):
            scores.append(results3.risk_score * self._meta_weights["maintainer_trust"])
        if isinstance(results4, BehavioralAnomalyResult):
            scores.append(results4.risk_score * self._meta_weights["behavioral_anomaly"])

        preliminary_score = sum(scores) / sum(
            w for k, w in self._meta_weights.items()
            if k in ("maintainer_trust", "behavioral_anomaly")
        ) if scores else 0.3

        return {
            "preliminary_score": preliminary_score,
            "results3": results3,
            "results4": results4,
        }

    def _is_clearly_safe(self, tier1_results: dict) -> bool:
        """
        Return True only if both fast agents agree the package is safe
        with high confidence. Conservative — prefer full scan.
        """
        score = tier1_results.get("preliminary_score", 1.0)
        r3 = tier1_results.get("results3")
        r4 = tier1_results.get("results4")

        # Both agents must agree AND have high confidence
        if score > 0.25:
            return False
        if isinstance(r3, MaintainerTrustResult) and r3.confidence < 0.7:
            return False
        if isinstance(r4, BehavioralAnomalyResult) and r4.confidence < 0.7:
            return False

        return True

    async def _run_all_agents_parallel(
        self, package_name: str, ecosystem: str, ws_manager=None
    ) -> AgentResults:
        """
        Run all 5 agents simultaneously using asyncio.gather.
        Total time = slowest single agent, not sum of all.
        """
        logger.info(f"Running all 5 agents in parallel for {package_name}")

        # Run agent1 first so its downloaded source_files can be passed to agent5
        agent1_result = await self.agent1.analyze(package_name, ecosystem)
        source_files = getattr(agent1_result, "source_files", {}) or {}

        # Run remaining 4 agents in parallel, passing source_files to agent5
        remaining = await asyncio.gather(
            self.agent2.analyze(package_name, ecosystem),
            self.agent3.analyze(package_name, ecosystem),
            self.agent4.analyze(package_name, ecosystem),
            self.agent5.analyze(package_name, ecosystem, source_files=source_files),
            return_exceptions=True,
        )

        # Reconstruct results in original agent order
        results = [agent1_result] + list(remaining)

        agent_results = AgentResults()

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {i+1} failed: {result}")
                continue

            if i == 0 and isinstance(result, CodeArchaeologistResult):
                agent_results.code_archaeologist = result
                if ws_manager:
                    await ws_manager.broadcast({
                        "type": "agent_complete",
                        "agent": "Code Archaeologist",
                        "risk_score": result.risk_score,
                        "confidence": result.confidence,
                        "features": result.features or {},
                        "evidence": [
                            {"severity": e.get("severity",""), "description": e.get("description", e.get("msg",""))}
                            for e in (result.evidence or [])[:5]
                        ],
                    })
            elif i == 1 and isinstance(result, DependencyGraphResult):
                agent_results.dependency_graph = result
                if ws_manager:
                    await ws_manager.broadcast({
                        "type": "agent_complete",
                        "agent": "Dependency Graph Analyst",
                        "risk_score": result.risk_score,
                        "confidence": result.confidence,
                        "vulnerabilities": [
                            {"id": v.get("id",""), "severity": v.get("severity",""), "summary": v.get("summary","")}
                            for v in (getattr(result, "vulnerabilities", None) or [])[:5]
                        ],
                        "evidence": [
                            {"severity": e.get("severity",""), "description": e.get("description", e.get("msg",""))}
                            for e in (result.evidence or [])[:5]
                        ],
                    })
            elif i == 2 and isinstance(result, MaintainerTrustResult):
                agent_results.maintainer_trust = result
                if ws_manager:
                    await ws_manager.broadcast({
                        "type": "agent_complete",
                        "agent": "Maintainer Trust Profiler",
                        "risk_score": result.risk_score,
                        "confidence": result.confidence,
                        "maintainer_profiles": (result.maintainer_profiles or [])[:3],
                        "evidence": [
                            {"severity": e.get("severity",""), "description": e.get("description", e.get("msg",""))}
                            for e in (result.evidence or [])[:5]
                        ],
                    })
            elif i == 3 and isinstance(result, BehavioralAnomalyResult):
                agent_results.behavioral_anomaly = result
                if ws_manager:
                    await ws_manager.broadcast({
                        "type": "agent_complete",
                        "agent": "Behavioral Anomaly Detector",
                        "risk_score": result.risk_score,
                        "confidence": result.confidence,
                        "anomaly_score": result.anomaly_score,
                        "evidence": [
                            {"severity": e.get("severity",""), "description": e.get("description", e.get("msg",""))}
                            for e in (result.evidence or [])[:5]
                        ],
                    })
            elif i == 4 and isinstance(result, SemanticIntentResult):
                agent_results.semantic_intent = result
                if ws_manager:
                    await ws_manager.broadcast({
                        "type": "agent_complete",
                        "agent": "Semantic Intent Analyzer",
                        "risk_score": result.risk_score,
                        "confidence": result.confidence,
                        "top_kb_match": result.top_kb_match or "",
                        "top_kb_similarity": result.top_kb_similarity or 0.0,
                        "evidence": [
                            {"severity": e.get("severity",""), "description": e.get("description", e.get("msg",""))}
                            for e in (result.evidence or [])[:5]
                        ],
                    })

        return agent_results

    def _meta_aggregate(self, agent_results: AgentResults) -> tuple[float, float]:
        """
        Meta-aggregate all agent scores using the trained logistic regression
        meta-learner when available, otherwise fall back to a conservative
        weighted average.

        The trained meta-learner was fit on 11,287 real packages (5104 malicious,
        6183 benign) and learned that:
          - Code Archaeologist (+3.79) and Dependency Graph (+3.73) are the
            primary reliable signals.
          - Behavioral Anomaly (+0.09) is nearly uninformative due to the
            LSTM model's sub-optimal training quality.
          - Maintainer Trust (-0.36) slightly reduces risk when it fires high
            (correlated with well-established packages).
          - Semantic Intent (+0.99) provides a useful secondary signal.

        By using the trained model we avoid false positives caused by the
        previous exponential amplification (alpha=4) which let Agent 4's
        noisy 0.85 score for react/lodash/axios dominate the final verdict.
        """
        # Collect agent scores in training order:
        # [code_archaeologist, dependency_graph, maintainer_trust,
        #  behavioral_anomaly, semantic_intent]
        default_neutral = 0.35  # score to use when an agent result is missing

        def _score(result) -> float:
            return result.risk_score if result is not None else default_neutral

        feature_vector = [
            _score(agent_results.code_archaeologist),
            _score(agent_results.dependency_graph),
            _score(agent_results.maintainer_trust),
            _score(agent_results.behavioral_anomaly),
            _score(agent_results.semantic_intent),
        ]

        confidences = [
            r.confidence for r in [
                agent_results.code_archaeologist,
                agent_results.dependency_graph,
                agent_results.maintainer_trust,
                agent_results.behavioral_anomaly,
                agent_results.semantic_intent,
            ] if r is not None
        ]
        mean_confidence = float(np.mean(confidences)) if confidences else 0.3

        # ── Primary path: trained meta-learner ─────────────────────────────
        if self._meta_learner is not None and self._meta_scaler is not None:
            try:
                X = self._meta_scaler.transform([feature_vector])
                proba = self._meta_learner.predict_proba(X)[0]
                # proba[1] = P(malicious) — already well-calibrated by the LR
                malicious_proba = float(proba[1])
                # Use meta-learner's own certainty as confidence signal
                ml_confidence = float(max(proba))
                # Blend with per-agent mean confidence so the UI still shows
                # agent-level certainty information
                blended_confidence = 0.7 * ml_confidence + 0.3 * mean_confidence
                return malicious_proba, blended_confidence
            except Exception as exc:
                logger.warning(f"Meta-learner predict failed, using fallback: {exc}")

        # ── Fallback: conservative weighted average ─────────────────────────
        # Used only when the meta-learner .pkl files are missing.
        # Weights mirror the logistic regression coefficient magnitudes.
        total_score = 0.0
        total_weight = 0.0
        for agent_name, result in {
            "code_archaeologist": agent_results.code_archaeologist,
            "dependency_graph":   agent_results.dependency_graph,
            "maintainer_trust":   agent_results.maintainer_trust,
            "behavioral_anomaly": agent_results.behavioral_anomaly,
            "semantic_intent":    agent_results.semantic_intent,
        }.items():
            if result is None:
                continue
            w = self._meta_weights[agent_name]
            total_score  += result.risk_score * w
            total_weight += w

        if total_weight == 0:
            return 0.3, 0.2

        final_score = total_score / total_weight
        return min(final_score, 1.0), mean_confidence

    def _score_to_tier(self, score: float) -> RiskTier:
        """Convert numeric score to risk tier."""
        if score < self.SAFE_THRESHOLD:
            return RiskTier.SAFE
        elif score < self.CAUTION_THRESHOLD:
            return RiskTier.CAUTION
        elif score < self.HIGH_THRESHOLD:
            return RiskTier.HIGH
        else:
            return RiskTier.CRITICAL

    async def _build_verdict(
        self, package_name: str, ecosystem: str,
        tier1_results: dict, agent_results: AgentResults,
        start: float, fast_mode: bool = False
    ) -> KavachVerdict:
        """Build the final verdict object."""

        # Use tier1 results if fast mode — still run through the meta-learner
        # with partial data so the output is on the same probability scale.
        if fast_mode:
            # Build partial AgentResults from tier1 data
            partial = AgentResults()
            r3 = tier1_results.get("results3")
            r4 = tier1_results.get("results4")
            if isinstance(r3, MaintainerTrustResult):
                partial.maintainer_trust = r3
            if isinstance(r4, BehavioralAnomalyResult):
                partial.behavioral_anomaly = r4
            risk_score, confidence = self._meta_aggregate(partial)
        else:
            risk_score, confidence = self._meta_aggregate(agent_results)

        risk_tier = self._score_to_tier(risk_score)
        install_blocked = risk_tier in (RiskTier.HIGH, RiskTier.CRITICAL)

        # Collect all evidence
        evidence_summary = self._collect_evidence(agent_results)

        # Generate causal explanation (SHAP + DoWhy)
        causal_explanation = {}
        if not fast_mode:
            try:
                causal_explanation = self.explainability.explain(
                    agent_results, risk_score
                )
            except Exception as e:
                logger.error(f"Explainability error: {e}")

        # RAG-based plain English explanation
        plain_english = ""
        similar_attacks = []
        try:
            plain_english, similar_attacks = await self.rag_engine.generate_explanation(
                package_name, risk_score, risk_tier, evidence_summary
            )
        except Exception as e:
            logger.error(f"RAG explanation error: {e}")
            plain_english = self._fallback_explanation(
                package_name, risk_score, risk_tier
            )

        # Safe alternatives
        safe_alternatives = await self._get_safe_alternatives(
            package_name, ecosystem, agent_results
        )

        exec_time = (time.time() - start) * 1000

        return KavachVerdict(
            package_name=package_name,
            ecosystem=ecosystem,
            risk_score=round(risk_score, 4),
            risk_tier=risk_tier,
            confidence=round(confidence, 4),
            agent_results=agent_results,
            causal_explanation=causal_explanation,
            plain_english_explanation=plain_english,
            similar_attacks=similar_attacks,
            safe_alternatives=safe_alternatives,
            execution_time_ms=round(exec_time, 2),
            install_blocked=install_blocked,
            evidence_summary=evidence_summary[:15],
        )

    def _collect_evidence(self, agent_results: AgentResults) -> list[dict]:
        """Collect and deduplicate evidence from all agents."""
        all_evidence = []

        for result in [
            agent_results.code_archaeologist,
            agent_results.dependency_graph,
            agent_results.maintainer_trust,
            agent_results.behavioral_anomaly,
            agent_results.semantic_intent,
        ]:
            if result and hasattr(result, "evidence"):
                for ev in result.evidence:
                    ev["source_agent"] = result.agent_name
                    all_evidence.append(ev)

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_evidence.sort(
            key=lambda x: severity_order.get(x.get("severity", "low"), 3)
        )

        return all_evidence

    async def _get_safe_alternatives(
        self, package_name: str, ecosystem: str, agent_results: AgentResults
    ) -> list[dict]:
        """Suggest safer alternatives for the flagged package."""
        alternatives_map = {
            "npm": {
                "pdf": [
                    {"name": "pdf-parse", "description": "Lightweight PDF text extraction", "score": 8},
                    {"name": "pdfjs-dist", "description": "Mozilla's PDF.js — gold standard", "score": 5},
                    {"name": "pdf2json", "description": "PDF to JSON converter", "score": 12},
                ],
                "http": [
                    {"name": "axios", "description": "Promise-based HTTP client", "score": 6},
                    {"name": "node-fetch", "description": "Lightweight fetch implementation", "score": 8},
                ],
                "crypto": [
                    {"name": "bcrypt", "description": "Password hashing library", "score": 4},
                    {"name": "crypto-js", "description": "Standard crypto algorithms", "score": 10},
                ],
            },
            "pypi": {
                "pdf": [
                    {"name": "PyPDF2", "description": "Pure Python PDF library", "score": 7},
                    {"name": "pdfminer.six", "description": "PDF text extraction", "score": 9},
                ],
                "http": [
                    {"name": "requests", "description": "HTTP for humans", "score": 3},
                    {"name": "httpx", "description": "Async HTTP client", "score": 5},
                ],
            },
        }

        # Simple keyword matching against package name
        pkg_lower = package_name.lower()
        eco_alts = alternatives_map.get(ecosystem, {})

        for keyword, alts in eco_alts.items():
            if keyword in pkg_lower:
                return alts

        return []

    async def _check_cache(self, package_name: str, ecosystem: str) -> Optional[KavachVerdict]:
        """Check Redis cache for existing scan result."""
        try:
            redis = await get_redis()
            cache_key = f"kavach:{ecosystem}:{package_name}"
            cached = await redis.get(cache_key)
            if cached:
                import json
                data = json.loads(cached)
                # Reconstruct verdict from cache
                # (simplified — full implementation would serialize/deserialize properly)
                return None  # TODO: Full deserialization
        except Exception:
            pass
        return None

    async def _cache_result(
        self, package_name: str, ecosystem: str, verdict: KavachVerdict
    ):
        """Cache scan result in Redis."""
        try:
            redis = await get_redis()
            cache_key = f"kavach:{ecosystem}:{package_name}"
            import json
            data = {
                "risk_score": verdict.risk_score,
                "risk_tier": verdict.risk_tier,
                "confidence": verdict.confidence,
                "install_blocked": verdict.install_blocked,
            }
            await redis.setex(cache_key, self.CACHE_TTL, json.dumps(data))
        except Exception as e:
            logger.debug(f"Cache write error: {e}")

    def _fallback_explanation(
        self, package_name: str, score: float, tier: RiskTier
    ) -> str:
        """Fallback explanation when RAG is unavailable."""
        if tier == RiskTier.CRITICAL:
            return (
                f"KAVACH has flagged {package_name} as CRITICAL risk (score: {score:.2f}). "
                "Multiple behavioral signals indicate this package may be malicious. "
                "Installation has been blocked. Review the agent findings for details."
            )
        elif tier == RiskTier.HIGH:
            return (
                f"{package_name} shows concerning behavioral patterns (score: {score:.2f}). "
                "Review the detailed findings before proceeding with installation."
            )
        elif tier == RiskTier.CAUTION:
            return (
                f"{package_name} has some anomalous signals (score: {score:.2f}). "
                "Proceed with caution and review the evidence."
            )
        else:
            return f"{package_name} passed all behavioral checks (score: {score:.2f}). Safe to install."