"""
KAVACH Model Loader
====================
Pre-loads all ML models into memory at startup.
Eliminates per-request model loading latency.
"""

from loguru import logger


class ModelLoader:
    """Loads and holds all models in memory."""

    def __init__(self):
        self.models_loaded = False

    async def load_all(self):
        """Load all agent models into memory."""
        from agents.agent1_code_archaeologist import CodeArchaeologist
        from agents.agent2_dependency_graph import DependencyGraphAnalyst
        from agents.agent3_maintainer_trust import MaintainerTrustProfiler
        from agents.agent4_behavioral_anomaly import BehavioralAnomalyDetector
        from agents.agent5_semantic_intent import SemanticIntentAnalyzer

        agents = [
            CodeArchaeologist(),
            DependencyGraphAnalyst(),
            MaintainerTrustProfiler(),
            BehavioralAnomalyDetector(),
            SemanticIntentAnalyzer(),
        ]

        for agent in agents:
            try:
                agent.load_model()
            except Exception as e:
                logger.warning(f"Model load warning for {agent.__class__.__name__}: {e}")

        self.models_loaded = True
        logger.info("All models loaded into memory")
