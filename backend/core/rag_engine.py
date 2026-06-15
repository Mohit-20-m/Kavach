"""
KAVACH RAG Engine
==================
Retrieves similar historical attacks from ChromaDB knowledge base
and generates plain English explanations via Gemini.
"""

from loguru import logger

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

import os


# Historical attack database — ingested into ChromaDB
HISTORICAL_ATTACKS = [
    {
        "id": "xz-utils-2024",
        "title": "XZ Utils Backdoor (2024)",
        "description": (
            "A sophisticated multi-year supply chain attack where attacker Jia Tan "
            "spent 2.5 years building trust in the XZ Utils open source project before "
            "inserting a backdoor in versions 5.6.0 and 5.6.1. The backdoor allowed "
            "unauthorized SSH access to affected Linux systems. Caught by accident when "
            "a Microsoft engineer noticed 500ms SSH latency."
        ),
        "signals": "maintainer_takeover new_maintainer dormant_reactivation obfuscated_code",
        "severity": "critical",
        "year": 2024,
    },
    {
        "id": "event-stream-2018",
        "title": "event-stream npm Attack (2018)",
        "description": (
            "Original maintainer Dominic Tarr transferred ownership of the popular "
            "event-stream package to an unknown attacker. The attacker added a malicious "
            "dependency (flatmap-stream) that targeted Bitcoin wallet theft. "
            "Affected millions of applications. Classic ownership transfer attack."
        ),
        "signals": "maintainer_takeover malicious_dependency bitcoin_theft npm ownership_transfer",
        "severity": "critical",
        "year": 2018,
    },
    {
        "id": "ua-parser-js-2021",
        "title": "ua-parser-js Hijacking (2021)",
        "description": (
            "The ua-parser-js npm package was hijacked to install cryptomining malware "
            "and password-stealing trojans. The package had 8 million weekly downloads. "
            "Malicious versions were available for ~4 hours before removal."
        ),
        "signals": "account_hijack cryptominer password_stealer postinstall_script",
        "severity": "critical",
        "year": 2021,
    },
    {
        "id": "node-ipc-2022",
        "title": "node-ipc Protestware (2022)",
        "description": (
            "The legitimate maintainer of node-ipc intentionally added destructive "
            "code that overwrote files on systems with Russian or Belarusian IP addresses. "
            "Demonstrates insider threat — malicious code from legitimate maintainer."
        ),
        "signals": "legitimate_maintainer malicious_update geolocation destructive_code",
        "severity": "high",
        "year": 2022,
    },
    {
        "id": "colors-faker-2022",
        "title": "colors.js & faker.js Sabotage (2022)",
        "description": (
            "Author Marak Squires intentionally broke colors.js and faker.js "
            "affecting millions of dependent packages. Published versions with "
            "infinite loops and gibberish output."
        ),
        "signals": "maintainer_sabotage infinite_loop package_corruption",
        "severity": "high",
        "year": 2022,
    },
    {
        "id": "ctx-pypi-2022",
        "title": "ctx PyPI Package (2022)",
        "description": (
            "The ctx package on PyPI was hijacked to steal environment variables "
            "including AWS credentials. The attack was via an expired domain — "
            "attacker registered the maintainer's email domain to take over the package."
        ),
        "signals": "env_harvesting aws_credentials domain_hijack pypi",
        "severity": "critical",
        "year": 2022,
    },
    {
        "id": "solarwinds-2020",
        "title": "SolarWinds SUNBURST (2020)",
        "description": (
            "Nation-state actors (Russia/APT29) compromised SolarWinds' build pipeline "
            "to insert SUNBURST backdoor into Orion software updates. "
            "Affected 18,000+ organizations including US government agencies. "
            "Supply chain attack via build system compromise."
        ),
        "signals": "build_pipeline_compromise nation_state_actor long_term_persistence",
        "severity": "critical",
        "year": 2020,
    },
]


class RAGEngine:
    """
    Retrieval-Augmented Generation for attack context and explanations.
    Ingests historical attack database and generates contextual explanations.
    """

    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self.gemini_model = None

    async def initialize(self):
        """Initialize ChromaDB and Gemini."""
        await self._init_chromadb()
        self._init_gemini()

    async def _init_chromadb(self):
        """Initialize ChromaDB and ingest historical attacks."""
        if not CHROMA_AVAILABLE:
            logger.warning("ChromaDB not available")
            return

        try:
            self.chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            self.collection = self.chroma_client.get_or_create_collection(
                name="historical_attacks",
                metadata={"hnsw:space": "cosine"},
            )

            # Check if already ingested
            existing = self.collection.count()
            if existing < len(HISTORICAL_ATTACKS):
                await self._ingest_attacks()

            logger.info(f"RAG knowledge base ready: {self.collection.count()} attacks indexed")

        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}")

    async def _ingest_attacks(self):
        """Ingest historical attack data into ChromaDB."""
        documents = [a["description"] + " " + a["signals"] for a in HISTORICAL_ATTACKS]
        embeddings = self.embedding_model.encode(documents).tolist()
        ids = [a["id"] for a in HISTORICAL_ATTACKS]
        metadatas = [
            {
                "title": a["title"],
                "severity": a["severity"],
                "year": a["year"],
                "signals": a["signals"],
            }
            for a in HISTORICAL_ATTACKS
        ]

        self.collection.upsert(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info(f"Ingested {len(HISTORICAL_ATTACKS)} historical attacks into ChromaDB")

    def _init_gemini(self):
        """Initialize Gemini for explanation generation."""
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini not available")
            return

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set")
            return

        try:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel("gemini-pro")
            logger.info("Gemini initialized for RAG explanations")
        except Exception as e:
            logger.warning(f"Gemini init failed: {e}")

    async def generate_explanation(
        self, package_name: str, risk_score: float,
        risk_tier, evidence_summary: list[dict]
    ) -> tuple[str, list[dict]]:
        """
        Retrieve similar attacks and generate plain English explanation.
        Returns (explanation_text, similar_attacks_list).
        """
        # Step 1: Retrieve similar historical attacks
        similar_attacks = await self._retrieve_similar_attacks(
            evidence_summary, risk_score
        )

        # Step 2: Generate explanation
        if self.gemini_model:
            explanation = await self._gemini_explain(
                package_name, risk_score, risk_tier,
                evidence_summary, similar_attacks
            )
        else:
            explanation = self._template_explanation(
                package_name, risk_score, risk_tier,
                evidence_summary, similar_attacks
            )

        return explanation, similar_attacks

    async def _retrieve_similar_attacks(
        self, evidence_summary: list[dict], risk_score: float
    ) -> list[dict]:
        """Query ChromaDB for historically similar attack patterns."""
        if not self.collection or not self.embedding_model:
            return self._fallback_similar_attacks(evidence_summary)

        try:
            # Build query from evidence
            query_text = " ".join([
                e.get("type", "") + " " + e.get("description", "")[:100]
                for e in evidence_summary[:5]
            ])

            if not query_text.strip():
                return []

            query_embedding = self.embedding_model.encode([query_text]).tolist()

            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=3,
                include=["documents", "metadatas", "distances"],
            )

            similar = []
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                similarity = 1 - dist  # Convert distance to similarity
                if similarity > 0.3:  # Only include reasonably similar attacks
                    similar.append({
                        "title": meta.get("title", ""),
                        "year": meta.get("year", ""),
                        "severity": meta.get("severity", ""),
                        "similarity": round(similarity, 3),
                        "description": doc[:300],
                    })

            return similar

        except Exception as e:
            logger.error(f"ChromaDB query error: {e}")
            return self._fallback_similar_attacks(evidence_summary)

    def _fallback_similar_attacks(self, evidence_summary: list[dict]) -> list[dict]:
        """Return relevant attacks based on evidence types when ChromaDB unavailable."""
        evidence_types = {e.get("type", "") for e in evidence_summary}

        relevant = []
        if "ownership_transfer_pattern" in evidence_types or "maintainer_anomaly" in evidence_types:
            relevant.append({
                "title": "XZ Utils Backdoor (2024)",
                "year": 2024,
                "severity": "critical",
                "similarity": 0.85,
                "description": HISTORICAL_ATTACKS[0]["description"][:300],
            })
            relevant.append({
                "title": "event-stream npm Attack (2018)",
                "year": 2018,
                "severity": "critical",
                "similarity": 0.78,
                "description": HISTORICAL_ATTACKS[1]["description"][:300],
            })

        if "env_harvesting" in evidence_types or "base64_decode" in evidence_types:
            relevant.append({
                "title": "ctx PyPI Package (2022)",
                "year": 2022,
                "severity": "critical",
                "similarity": 0.72,
                "description": HISTORICAL_ATTACKS[5]["description"][:300],
            })

        return relevant[:3]

    async def _gemini_explain(
        self, package_name: str, risk_score: float, risk_tier,
        evidence_summary: list[dict], similar_attacks: list[dict]
    ) -> str:
        """Generate plain English explanation using Gemini."""
        # Build context for Gemini
        evidence_text = "\n".join([
            f"- [{e.get('severity', '').upper()}] {e.get('description', '')}"
            for e in evidence_summary[:8]
        ])

        attacks_text = "\n".join([
            f"- {a['title']} ({a['year']}): {a['description'][:150]}"
            for a in similar_attacks[:2]
        ]) if similar_attacks else "No similar historical attacks found in database."

        prompt = f"""You are KAVACH, an AI security analyst. Write a clear, professional 3-4 sentence explanation of why the package "{package_name}" was flagged.

Risk Score: {risk_score:.2f}/1.0 ({risk_tier})

Evidence Found:
{evidence_text}

Similar Historical Attacks:
{attacks_text}

Write the explanation for a developer who needs to understand:
1. What specifically makes this package suspicious
2. What kind of attack this resembles
3. What could happen if they install it

Be specific, factual, and avoid being alarmist. Use plain English without excessive technical jargon."""

        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return self._template_explanation(
                package_name, risk_score, risk_tier, evidence_summary, similar_attacks
            )

    def _template_explanation(
        self, package_name: str, risk_score: float, risk_tier,
        evidence_summary: list[dict], similar_attacks: list[dict]
    ) -> str:
        """Template-based fallback explanation."""
        tier_str = risk_tier if isinstance(risk_tier, str) else risk_tier.value

        # SAFE packages get a clean, reassuring message
        if tier_str == "SAFE":
            return (
                f"{package_name} passed all behavioral checks (score: {risk_score:.2f}). "
                f"No significant supply chain risk signals were detected. "
                f"Safe to install."
            )

        critical_evidence = [
            e for e in evidence_summary if e.get("severity") == "critical"
        ]
        primary = critical_evidence[0] if critical_evidence else evidence_summary[0] if evidence_summary else None

        attack_ref = ""
        if similar_attacks:
            attack_ref = (
                f" This pattern closely resembles the {similar_attacks[0]['title']}, "
                f"where {similar_attacks[0]['description'][:150]}"
            )

        primary_desc = primary.get("description", "multiple suspicious signals") if primary else "suspicious behavioral patterns"

        if tier_str == "CAUTION":
            return (
                f"KAVACH found some anomalous signals in {package_name} (score: {risk_score:.2f}). "
                f"The primary signal is: {primary_desc}.{attack_ref} "
                f"Review the evidence carefully before installing."
            )

        return (
            f"KAVACH flagged {package_name} with a {tier_str} risk score of {risk_score:.2f}. "
            f"The primary concern is: {primary_desc}.{attack_ref} "
            f"If installed, this package could compromise your development environment, "
            f"steal credentials, or introduce malicious code into your application."
        )
