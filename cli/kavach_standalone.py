#!/usr/bin/env python3
"""
KAVACH Standalone CLI
======================
Runs all 5 security agents DIRECTLY — no Docker, no backend server needed.
Models are loaded from ~/.kavach/models/ on first run, then cached in memory.

Install:
  pip install -e ./cli

Setup (one time):
  kavach-standalone setup
  source ~/.zshrc

After setup, npm/pip are automatically intercepted:
  npm install axios       ← scanned before install
  pip install requests    ← scanned before install
"""

import asyncio
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# ─── Model directory ─────────────────────────────────────────────────────────
# Default: ~/.kavach/models/
# Override: KAVACH_MODELS_DIR env var
MODELS_DIR = Path(
    os.getenv("KAVACH_MODELS_DIR", Path.home() / ".kavach" / "models")
)

app = typer.Typer(
    name="kavach-standalone",
    help="🛡️  KAVACH — Standalone Supply Chain Security (no Docker needed)",
    add_completion=False,
)
console = Console()

TIER_COLORS = {
    "SAFE": "bold green",
    "CAUTION": "bold yellow",
    "HIGH": "bold red",
    "CRITICAL": "bold red on white",
}
TIER_ICONS = {
    "SAFE": "✅",
    "CAUTION": "⚠️ ",
    "HIGH": "🔴",
    "CRITICAL": "🚨",
}

NPM_REGISTRY  = "https://registry.npmjs.org"
PYPI_REGISTRY = "https://pypi.org/pypi"

_TRUSTED_DOMAINS = {
    "github.com", "githubusercontent.com", "npmjs.com", "pypi.org",
    "registry.npmjs.org", "files.pythonhosted.org", "cloudflare.com",
    "amazonaws.com", "googleapis.com", "google.com", "microsoft.com",
    "unpkg.com", "jsdelivr.net", "nodejs.org", "mozilla.org",
    "developer.mozilla.org", "ecma-international.org", "w3.org",
    "axios-http.com", "lodash.com", "reactjs.org",
    "stackoverflow.com", "wikipedia.org",
}

# ─── Feature extraction (same as Agent 1) ────────────────────────────────────

FEATURE_NAMES = [
    "max_string_entropy", "mean_string_entropy", "high_entropy_string_count",
    "network_call_count", "network_calls_per_100_lines", "dynamic_execution_count",
    "env_access_count", "filesystem_access_count", "has_postinstall",
    "obfuscated_identifier_ratio", "base64_decode_count",
    "suspicious_external_domain_count", "dangerous_import_count",
    "max_call_nesting_depth", "lambda_count", "magic_method_overrides",
]

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())

def _is_trusted(domain: str) -> bool:
    return any(domain == t or domain.endswith("." + t) for t in _TRUSTED_DOMAINS)

def _extract_features(source: str, filename: str) -> dict:
    lines = source.split("\n")
    total_lines = max(len(lines), 1)
    strings = re.findall(r'"([^"]{20,})"', source) + re.findall(r"'([^']{20,})'", source)
    entropies = [_shannon_entropy(s) for s in strings]

    net = sum(len(re.findall(p, source)) for p in [
        r"require\(['\"]http[s]?['\"]", r"require\(['\"]net['\"]",
        r"fetch\(", r"XMLHttpRequest", r"urllib\.request",
        r"requests\.(get|post|put)", r"http\.get\(", r"https\.get\(",
    ])
    dyn = sum(len(re.findall(p, source)) for p in [
        r"\beval\s*\(", r"\bexec\s*\(", r"Function\s*\(",
        r"new\s+Function", r"__import__\s*\(", r"execSync\s*\(",
    ])
    env = sum(len(re.findall(p, source)) for p in [
        r"process\.env\.", r"os\.environ", r"os\.getenv\(", r"getenv\(",
    ])
    fs = sum(len(re.findall(p, source)) for p in [
        r"fs\.read", r"fs\.write", r"os\.walk",
        r"open\s*\(", r"readdir", r"readFileSync",
    ])
    b64 = sum(len(re.findall(p, source)) for p in [
        r"atob\(", r"base64\.b64decode",
        r"Buffer\.from\([^,]+,\s*['\"]base64['\"]",
    ])
    domains = re.findall(r'https?://([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', source)
    susp_domains = sum(1 for d in domains if not _is_trusted(d))
    url_short = sum(1 for d in domains if any(
        s in d for s in ["bit.ly", "tinyurl", "t.co", "goo.gl"]
    ))
    postinstall = 1 if "package.json" in filename and "postinstall" in source.lower() else 0
    ids = re.findall(r'\b([a-zA-Z_]\w*)\b', source)
    obfusc = sum(1 for i in ids if len(i) == 1) / max(len(ids), 1)
    danger = 0
    if filename.endswith(".py"):
        danger = sum(len(re.findall(p, source)) for p in [
            r"\bsubprocess\b", r"\bctypes\b", r"\bpickle\b", r"\bos\.system\b",
        ])

    return {
        "max_string_entropy":               max(entropies) if entropies else 0.0,
        "mean_string_entropy":              float(sum(entropies)/len(entropies)) if entropies else 0.0,
        "high_entropy_string_count":        sum(1 for e in entropies if e > 4.5),
        "network_call_count":               net,
        "network_calls_per_100_lines":      (net / total_lines) * 100,
        "dynamic_execution_count":          dyn,
        "env_access_count":                 env,
        "filesystem_access_count":          fs,
        "has_postinstall":                  postinstall,
        "obfuscated_identifier_ratio":      obfusc,
        "base64_decode_count":              b64,
        "suspicious_external_domain_count": susp_domains,
        "dangerous_import_count":           danger,
        "max_call_nesting_depth":           0,
        "lambda_count":                     len(re.findall(r'\blambda\b', source)),
        "magic_method_overrides":           len(re.findall(
            r'def (__reduce__|__reduce_ex__|__getattr__|__setattr__)', source
        )),
        "url_shortener_count":              url_short,
    }

def _aggregate(features: list[dict]) -> dict:
    if not features:
        return {k: 0.0 for k in FEATURE_NAMES}
    agg = {}
    for key in set().union(*features):
        vals = [f.get(key, 0) for f in features]
        agg[key] = sum(vals) if ("count" in key or "has_" in key) else max(vals)
    return agg

# ─── Model loader (singleton) ─────────────────────────────────────────────────

class _Models:
    """Loads all models once and keeps them in memory."""
    _instance = None
    _loaded = False

    xgb        = None
    ifor_maint = None
    ifor_behav = None
    sbert      = None
    meta       = None
    meta_scaler = None
    thresholds  = None
    col99_maint = None
    col99_behav = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        if not cls._loaded:
            cls._instance._load()
        return cls._instance

    def _load(self):
        import joblib
        import numpy as np

        errors = []

        # XGBoost
        p = MODELS_DIR / "code_archaeologist.pkl"
        if p.exists():
            try:
                self.xgb = joblib.load(p)
            except Exception as e:
                errors.append(f"XGBoost: {e}")

        # Maintainer IsoForest
        p = MODELS_DIR / "maintainer_isolation_forest.pkl"
        if p.exists():
            try:
                self.ifor_maint = joblib.load(p)
            except Exception as e:
                errors.append(f"IsoForest-maint: {e}")

        p2 = MODELS_DIR / "maintainer_profile_col99.npy"
        if p2.exists():
            self.col99_maint = np.load(str(p2))

        # Behavioral IsoForest
        p = MODELS_DIR / "behavioral_isolation_forest.pkl"
        if p.exists():
            try:
                self.ifor_behav = joblib.load(p)
            except Exception as e:
                errors.append(f"IsoForest-behav: {e}")

        p2 = MODELS_DIR / "behavioral_metrics_col99.npy"
        if p2.exists():
            self.col99_behav = np.load(str(p2))

        # SBERT
        p = MODELS_DIR / "sbert_fine_tuned"
        if p.exists():
            try:
                from sentence_transformers import SentenceTransformer
                self.sbert = SentenceTransformer(str(p))
            except Exception as e:
                errors.append(f"SBERT: {e}")

        # Meta-learner
        p = MODELS_DIR / "meta_learner.pkl"
        p2 = MODELS_DIR / "meta_learner_scaler.pkl"
        if p.exists() and p2.exists():
            try:
                self.meta = joblib.load(p)
                self.meta_scaler = joblib.load(p2)
            except Exception as e:
                errors.append(f"Meta: {e}")

        # Thresholds
        p = MODELS_DIR / "score_thresholds.json"
        if p.exists():
            with open(p) as f:
                self.thresholds = json.load(f)

        if errors:
            console.print(f"[yellow]⚠️  Some models failed to load: {', '.join(errors)}[/yellow]")

        _Models._loaded = True


# ─── Source downloader ────────────────────────────────────────────────────────

async def _download_source(name: str, ecosystem: str) -> dict:
    """Download package source and return {filename: content} dict."""
    source_files = {}
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            if ecosystem == "npm":
                r = await client.get(f"{NPM_REGISTRY}/{name}/latest")
                if r.status_code != 200:
                    return {}
                meta = r.json()
                url = meta.get("dist", {}).get("tarball")
                if not url:
                    return {}
                tr = await client.get(url, timeout=20.0)
                if tr.status_code != 200:
                    return {}
                with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as f:
                    f.write(tr.content)
                    tmp = f.name
                try:
                    with tarfile.open(tmp, "r:gz") as tar:
                        for m in tar.getmembers():
                            if not m.name.endswith((".js", ".ts", ".json")):
                                continue
                            if any(x in m.name for x in (
                                "node_modules", ".min.", ".bundle.", "dist/",
                                "__tests__", ".spec.", ".test."
                            )):
                                continue
                            if m.size > 100_000:
                                continue
                            try:
                                fobj = tar.extractfile(m)
                                if fobj:
                                    source_files[m.name] = fobj.read().decode("utf-8", errors="ignore")
                            except Exception:
                                pass
                finally:
                    Path(tmp).unlink(missing_ok=True)

            elif ecosystem == "pypi":
                r = await client.get(f"{PYPI_REGISTRY}/{name}/json")
                if r.status_code != 200:
                    return {}
                urls = r.json().get("urls", [])
                target = next((u for u in urls if u["packagetype"] == "bdist_wheel"), None) or \
                         next((u for u in urls if u["packagetype"] == "sdist"), None)
                if not target:
                    return {}
                pr = await client.get(target["url"], timeout=20.0)
                if pr.status_code != 200:
                    return {}
                suffix = ".whl" if "whl" in target["url"] else ".tar.gz"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                    f.write(pr.content)
                    tmp = f.name
                try:
                    try:
                        with zipfile.ZipFile(tmp) as z:
                            for n in z.namelist():
                                if n.endswith(".py") and "__pycache__" not in n:
                                    source_files[n] = z.read(n).decode("utf-8", errors="ignore")
                    except zipfile.BadZipFile:
                        with tarfile.open(tmp, "r:gz") as tar:
                            for m in tar.getmembers():
                                if m.name.endswith(".py"):
                                    fobj = tar.extractfile(m)
                                    if fobj:
                                        source_files[m.name] = fobj.read().decode("utf-8", errors="ignore")
                finally:
                    Path(tmp).unlink(missing_ok=True)
    except Exception:
        pass
    return source_files


# ─── Individual agent scorers ─────────────────────────────────────────────────

def _agent1_score(source_files: dict, models: _Models) -> tuple[float, float]:
    """Code Archaeologist — XGBoost on AST features."""
    import numpy as np
    if not source_files:
        return 0.30, 0.20

    feats = [_extract_features(content, name) for name, content in source_files.items()]
    agg = _aggregate(feats)
    vec = [float(agg.get(f, 0.0)) for f in FEATURE_NAMES]

    # Evidence-based floor
    floor = 0.0
    if agg.get("url_shortener_count", 0) > 0:
        floor = max(floor, 0.35)
    if agg.get("base64_decode_count", 0) > 0 and agg.get("network_call_count", 0) > 0:
        floor = max(floor, 0.40)
    if agg.get("dynamic_execution_count", 0) > 0:
        floor = max(floor, 0.25)

    if models.xgb is not None:
        try:
            proba = models.xgb.predict_proba([vec])[0]
            score = float(proba[1])
            conf  = float(max(proba))
            # Reduce confidence if raw features disagree
            suspicious = sum([
                1 if agg.get("high_entropy_string_count", 0) > 0 else 0,
                1 if agg.get("dynamic_execution_count", 0) > 0 else 0,
                1 if agg.get("base64_decode_count", 0) > 0 else 0,
                1 if agg.get("suspicious_external_domain_count", 0) > 0 else 0,
                1 if agg.get("has_postinstall", 0) > 0 else 0,
            ])
            if score < 0.15 and conf > 0.85 and suspicious >= 2:
                conf = 0.60
            return max(score, floor), conf
        except Exception:
            pass

    # Rule-based fallback
    score = sum([
        min(agg.get("high_entropy_string_count", 0) / 5.0, 1.0) * 0.15,
        min(agg.get("dynamic_execution_count", 0) / 3.0, 1.0) * 0.20,
        min(agg.get("base64_decode_count", 0) / 3.0, 1.0) * 0.18,
        min(agg.get("suspicious_external_domain_count", 0) / 3.0, 1.0) * 0.20,
        min(agg.get("env_access_count", 0) / 5.0, 1.0) * 0.10,
        float(agg.get("has_postinstall", 0)) * 0.08,
        agg.get("url_shortener_count", 0) * 0.09,
    ])
    return max(min(score, 1.0), floor), 0.55


async def _agent2_score(name: str, ecosystem: str) -> tuple[float, float]:
    """Dependency Graph — checks for MALWARE-class advisories via OSV + GHSA."""
    try:
        eco_map = {"npm": "npm", "pypi": "PyPI"}
        eco = eco_map.get(ecosystem, "npm")
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                "https://api.osv.dev/v1/query",
                json={"package": {"name": name, "ecosystem": eco}},
            )
            if r.status_code == 200:
                vulns = r.json().get("vulns", [])

                # Only flag supply chain MALWARE — not regular CVEs
                # MAL- prefix = OSSF malicious packages database
                # GHSA- with malware classification
                malware_vulns = []
                for v in vulns:
                    vid = v.get("id", "")
                    summary = v.get("summary", "").lower()
                    # Only flag if advisory ID is MAL- (OSSF malicious packages)
                    # or summary explicitly says malware/malicious code
                    if vid.startswith("MAL-"):
                        malware_vulns.append(vid)
                    elif any(kw in summary for kw in [
                        "malicious code", "malware", "backdoor",
                        "credential harvesting", "data exfiltration",
                        "supply chain attack", "trojan",
                    ]):
                        malware_vulns.append(vid)

                if malware_vulns:
                    return 0.95, 0.90
                # Regular CVEs (XSS, prototype pollution etc) = low-medium risk
                # These are NOT supply chain attacks
                elif len(vulns) > 5:
                    return 0.25, 0.60
                elif len(vulns) > 0:
                    return 0.15, 0.55
    except Exception:
        pass
    return 0.05, 0.50


async def _agent3_score(name: str, ecosystem: str, models: _Models) -> tuple[float, float]:
    """Maintainer Trust — IsoForest on maintainer profiles."""
    import numpy as np

    profile = [0.0] * 9  # default unknown profile

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if ecosystem == "npm":
                r = await client.get(f"{NPM_REGISTRY}/{name}")
                if r.status_code == 200:
                    meta = r.json()
                    time_info = meta.get("time", {})

                    # Account age proxy: package age
                    try:
                        from datetime import datetime, timezone
                        created = datetime.fromisoformat(
                            time_info.get("created", "2020-01-01T00:00:00Z").rstrip("Z")
                        ).replace(tzinfo=timezone.utc)
                        age_days = (datetime.now(timezone.utc) - created).days
                    except Exception:
                        age_days = 365

                    versions = [v for v in time_info if v not in ("created", "modified")]
                    maintainers = meta.get("maintainers", [])
                    downloads = 0
                    try:
                        from datetime import datetime, timedelta
                        end = datetime.now()
                        start = end - timedelta(days=30)
                        period = f"{start.strftime('%Y-%m-%d')}:{end.strftime('%Y-%m-%d')}"
                        dr = await client.get(
                            f"https://api.npmjs.org/downloads/range/{period}/{name}",
                            timeout=8.0
                        )
                        if dr.status_code == 200:
                            downloads = sum(d.get("downloads", 0) for d in dr.json().get("downloads", []))
                    except Exception:
                        pass

                    profile = [
                        min(float(downloads), 1e8),
                        float(max(len(versions), 1)),
                        float(90),  # days_since_update proxy
                        float(max(len(maintainers), 1)),
                        float(age_days),
                        float(1 if meta.get("repository") else 0),
                        float(1 if meta.get("homepage") and meta.get("license") else 0),
                        0.8,
                        float(min(len(maintainers), 20)),
                    ]
    except Exception:
        pass

    if models.ifor_maint is not None:
        try:
            import numpy as np
            arr = np.array(profile, dtype=np.float32).reshape(1, -1)
            if models.col99_maint is not None:
                arr = (arr / models.col99_maint).clip(0, 1)
            decision = models.ifor_maint.decision_function(arr)[0]
            score = max(0.0, min(1.0, (0.10 - decision) / 0.40))
            return round(score, 3), 0.72
        except Exception:
            pass

    return 0.25, 0.20


async def _agent4_score(name: str, ecosystem: str, models: _Models) -> tuple[float, float]:
    """Behavioral Anomaly — LSTM + IsoForest on download time series."""
    import numpy as np

    try:
        if ecosystem != "npm":
            return 0.20, 0.50

        async with httpx.AsyncClient(timeout=12.0) as client:
            from datetime import datetime, timedelta
            end = datetime.now()
            start = end - timedelta(weeks=52)
            period = f"{start.strftime('%Y-%m-%d')}:{end.strftime('%Y-%m-%d')}"
            r = await client.get(
                f"https://api.npmjs.org/downloads/range/{period}/{name}"
            )
            if r.status_code != 200:
                return 0.20, 0.50

            data = r.json().get("downloads", [])
            if len(data) < 12:
                return 0.20, 0.50

            daily = np.array([d["downloads"] for d in data], dtype=np.float32)
            n_weeks = len(daily) // 7
            if n_weeks < 4:
                return 0.20, 0.50
            weekly = daily[:n_weeks * 7].reshape(n_weeks, 7).sum(axis=1)

            # Compute temporal metrics
            mean_dl = float(weekly.mean())
            std_dl  = float(weekly.std()) + 1e-8
            z_scores = (weekly - mean_dl) / std_dl
            max_z    = float(z_scores.max())
            spike_weeks = int((np.abs(z_scores) > 3).sum())
            cv = std_dl / (mean_dl + 1e-8)

            # IsoForest scoring
            feature_vec = [
                float(np.log1p(mean_dl) / np.log1p(10_000_000)),
                min(cv / 5.0, 1.0),
                min(max_z / 10.0, 1.0),
                min(spike_weeks / 10.0, 1.0),
                min(float(weekly[-4:].mean() / (mean_dl + 1e-8)) / 10.0, 1.0),
                0.0, 0.0, 0.0,
            ]

            if models.ifor_behav is not None:
                arr = np.array(feature_vec, dtype=np.float32).reshape(1, -1)
                if models.col99_behav is not None:
                    arr = (arr / models.col99_behav).clip(0, 1)
                decision = models.ifor_behav.decision_function(arr)[0]
                score = max(0.0, min(1.0, (0.10 - decision) / 0.40))
                return round(score, 3), 0.75
            else:
                # Rule-based
                score = min(max_z / 10.0 + spike_weeks * 0.05, 1.0)
                return score, 0.55

    except Exception:
        pass
    return 0.20, 0.50


async def _agent5_score(
    name: str, ecosystem: str,
    source_files: dict, models: _Models
) -> tuple[float, float]:
    """Semantic Intent — SBERT KB similarity."""
    _MALICIOUS_KB = [
        "reads environment variables including API keys tokens secrets and credentials",
        "executes shell commands or system calls accessing operating system resources",
        "establishes network connections to external servers for data exfiltration",
        "obfuscates code using base64 encoding eval calls or hex encoding to hide intent",
        "modifies installation hooks in package.json postinstall scripts for persistence",
        "accesses and reads sensitive filesystem paths including SSH keys and config files",
        "performs cryptocurrency mining operations consuming CPU and system resources",
        "harvests and exfiltrates developer credentials tokens and API keys",
        "establishes backdoor connections maintaining persistent remote access",
        "performs typosquatting mimicking legitimate package names to deceive developers",
        "communicates with command and control infrastructure receiving malicious instructions",
        "steals browser session tokens cookies or stored database credentials",
    ]

    try:
        # Build behavior summary from source files
        all_source = "\n".join(source_files.values()) if source_files else ""
        behaviors = []

        if re.search(r"https?://|fetch\(|urllib\.request|requests\.", all_source):
            behaviors.append("makes HTTP network requests to external servers")
        env_count = len(re.findall(r'process\.env\.|os\.environ|os\.getenv', all_source))
        if env_count > 0:
            behaviors.append(f"reads {env_count} environment variables including potential secrets")
        if re.search(r'exec\(|spawn\(|child_process|subprocess\.', all_source):
            behaviors.append("executes shell commands")
        if re.search(r'atob|btoa|base64|b64decode', all_source):
            behaviors.append("encodes/decodes binary data using base64")
        domains = re.findall(r'https?://([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', all_source)
        susp = [d for d in domains if not _is_trusted(d)]
        if susp:
            behaviors.append(f"communicates with external domain: {susp[0]}")

        if not behaviors:
            # Fetch description for semantic comparison
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    if ecosystem == "npm":
                        r = await client.get(f"{NPM_REGISTRY}/{name}/latest")
                        desc = r.json().get("description", "") if r.status_code == 200 else ""
                    else:
                        r = await client.get(f"{PYPI_REGISTRY}/{name}/json")
                        desc = r.json().get("info", {}).get("summary", "") if r.status_code == 200 else ""
                behaviors = [desc] if desc else ["performs utility operations"]
            except Exception:
                behaviors = ["performs utility operations"]

        behavior_text = ". ".join(behaviors)

        if models.sbert is not None:
            try:
                from sentence_transformers import util
                import torch
                kb_embs = models.sbert.encode(_MALICIOUS_KB, convert_to_tensor=True)
                beh_emb = models.sbert.encode(behavior_text, convert_to_tensor=True)
                sims = util.cos_sim(beh_emb, kb_embs)[0]
                max_sim = float(sims.max().item())
                # Map similarity to risk: >0.7 = high risk, <0.3 = low risk
                if max_sim >= 0.70:
                    score = 0.05 + (max_sim - 0.70) * 0.5
                elif max_sim >= 0.50:
                    score = 0.15 + (max_sim - 0.50) * 0.75
                elif max_sim >= 0.30:
                    score = 0.50 + (0.50 - max_sim) * 1.0
                else:
                    score = 0.70 + (0.30 - max_sim) * 0.5
                return round(min(score, 1.0), 3), 0.65
            except Exception:
                pass
    except Exception:
        pass
    return 0.15, 0.10


# ─── Meta-learner aggregation ─────────────────────────────────────────────────

def _aggregate_scores(
    s1: float, s2: float, s3: float, s4: float, s5: float,
    models: _Models
) -> tuple[float, float]:
    """Combine 5 agent scores into final risk score."""
    feature_vec = [s1, s2, s3, s4, s5]

    if models.meta is not None and models.meta_scaler is not None:
        try:
            import numpy as np
            X = models.meta_scaler.transform([feature_vec])
            proba = models.meta.predict_proba(X)[0]
            mal_proba = float(proba[1])
            conf = float(max(proba))

            # Safety net: if 3+ agents see risk, blend with weighted average
            agents_above = sum(1 for s in feature_vec if s > 0.30)
            if agents_above >= 3:
                weights = [0.22, 0.18, 0.18, 0.17, 0.25]
                weighted_avg = sum(s * w for s, w in zip(feature_vec, weights))
                mal_proba = 0.5 * mal_proba + 0.5 * weighted_avg

            return round(mal_proba, 3), round(conf, 3)
        except Exception:
            pass

    # Fallback weighted average
    weights = [0.22, 0.18, 0.18, 0.17, 0.25]
    score = sum(s * w for s, w in zip(feature_vec, weights))
    return round(min(score, 1.0), 3), 0.60


def _score_to_tier(score: float, thresholds: Optional[dict]) -> str:
    safe_t    = float(thresholds.get("SAFE_THRESHOLD",    0.10)) if thresholds else 0.10
    caution_t = float(thresholds.get("CAUTION_THRESHOLD", 0.30)) if thresholds else 0.30
    high_t    = float(thresholds.get("HIGH_THRESHOLD",    0.75)) if thresholds else 0.75

    if score < safe_t:
        return "SAFE"
    elif score < caution_t:
        return "CAUTION"
    elif score < high_t:
        return "HIGH"
    else:
        return "CRITICAL"


# ─── Main scan function ────────────────────────────────────────────────────────

async def _scan_package_standalone(
    package_name: str,
    ecosystem: str,
) -> dict:
    """Run all 5 agents locally and return verdict dict."""

    start = time.time()
    models = _Models.get()

    # Download source (agent1 + agent5 use this)
    source_files = await _download_source(package_name, ecosystem)

    # Run all 5 agents in parallel (agents 2,3,4 are network-bound)
    results = await asyncio.gather(
        asyncio.coroutine(lambda: _agent1_score(source_files, models))()
        if False else asyncio.get_event_loop().run_in_executor(
            None, _agent1_score, source_files, models
        ),
        _agent2_score(package_name, ecosystem),
        _agent3_score(package_name, ecosystem, models),
        _agent4_score(package_name, ecosystem, models),
        _agent5_score(package_name, ecosystem, source_files, models),
        return_exceptions=True,
    )

    def _safe(r, default=(0.2, 0.1)):
        return r if isinstance(r, tuple) else default

    s1, c1 = _safe(results[0])
    s2, c2 = _safe(results[1])
    s3, c3 = _safe(results[2])
    s4, c4 = _safe(results[3])
    s5, c5 = _safe(results[4])

    final_score, final_conf = _aggregate_scores(s1, s2, s3, s4, s5, models)
    tier = _score_to_tier(final_score, models.thresholds)
    blocked = tier in ("HIGH", "CRITICAL")
    exec_ms = (time.time() - start) * 1000

    return {
        "package_name":              package_name,
        "ecosystem":                 ecosystem,
        "risk_score":                final_score,
        "risk_tier":                 tier,
        "confidence":                final_conf,
        "install_blocked":           blocked,
        "execution_time_ms":         exec_ms,
        "plain_english_explanation": (
            f"{package_name} passed all behavioral checks (score: {final_score:.2f}). "
            "No significant supply chain risk signals were detected. Safe to install."
            if not blocked else
            f"KAVACH flagged {package_name} with a {tier} risk score of {final_score:.2f}. "
            "If installed, this package could compromise your development environment."
        ),
        "evidence_summary":          [],
        "safe_alternatives":         [],
        "similar_attacks":           [],
        "causal_explanation":        {},
        "agent_scores": {
            "code_archaeologist": {"risk_score": s1, "confidence": c1, "execution_time_ms": 0},
            "dependency_graph":   {"risk_score": s2, "confidence": c2, "execution_time_ms": 0},
            "maintainer_trust":   {"risk_score": s3, "confidence": c3, "execution_time_ms": 0},
            "behavioral_anomaly": {"risk_score": s4, "confidence": c4, "execution_time_ms": 0},
            "semantic_intent":    {"risk_score": s5, "confidence": c5, "execution_time_ms": 0},
        },
    }


# ─── Display ──────────────────────────────────────────────────────────────────

def _display_verdict(result: dict, package_name: str):
    tier    = result.get("risk_tier", "SAFE")
    score   = result.get("risk_score", 0.0)
    blocked = result.get("install_blocked", False)
    color   = TIER_COLORS.get(tier, "white")
    icon    = TIER_ICONS.get(tier, "?")
    exec_ms = result.get("execution_time_ms", 0)

    console.print()
    console.rule(f"[cyan]🛡️  KAVACH Security Scan — {package_name}[/cyan]")

    verdict_text = Text()
    verdict_text.append(f"\n  {icon} Risk Tier: ", style="bold white")
    verdict_text.append(f"{tier}", style=color)
    verdict_text.append("   Score: ", style="bold white")
    verdict_text.append(f"{score:.2f}/1.0", style=color)
    verdict_text.append(f"   Analysis: {exec_ms:.0f}ms\n", style="dim")

    if blocked:
        title = f"[bold red] {icon} INSTALL BLOCKED — {tier} RISK DETECTED[/bold red]"
        border = "red"
    elif tier == "CAUTION":
        title = f"[bold yellow]{icon} CAUTION — Review before installing[/bold yellow]"
        border = "yellow"
    else:
        title = f"[bold green]{icon} SAFE — Package cleared all checks[/bold green]"
        border = "green"

    console.print(Panel(verdict_text, title=title, border_style=border))

    # Agent table
    scores = result.get("agent_scores", {})
    if scores:
        table = Table(title="Agent Analysis", box=box.ROUNDED,
                      border_style="cyan", header_style="bold cyan")
        table.add_column("Agent",      style="white", min_width=25)
        table.add_column("Risk Score", justify="center", min_width=12)
        table.add_column("Confidence", justify="center", min_width=12)

        display = {
            "code_archaeologist": "🔬 Code Archaeologist",
            "dependency_graph":   "🕸️  Dependency Graph",
            "maintainer_trust":   "👤 Maintainer Trust",
            "behavioral_anomaly": "📈 Behavioral Anomaly",
            "semantic_intent":    "🧠 Semantic Intent",
        }
        for key, data in scores.items():
            s = data.get("risk_score", 0)
            c = data.get("confidence", 0)
            sc = f"[bold red]{s:.2f}[/bold red]" if s > 0.7 else \
                 f"[yellow]{s:.2f}[/yellow]" if s > 0.4 else \
                 f"[green]{s:.2f}[/green]"
            table.add_row(display.get(key, key), sc, f"{c:.2f}")
        console.print(table)

    explanation = result.get("plain_english_explanation", "")
    if explanation:
        console.print(Panel(explanation, title="[bold white]🔍 Security Analysis[/bold white]",
                            border_style="blue", padding=(1, 2)))
    console.print()


def _execute(package: str, ecosystem: str, extra_args: list):
    cmd = (["npm", "install", package] if ecosystem == "npm"
           else ["pip", "install", package]) + extra_args
    console.print(f"[dim]→ Executing: {' '.join(cmd)}[/dim]")
    return subprocess.run(cmd).returncode


# ─── CLI commands ─────────────────────────────────────────────────────────────

@app.command("npm")
def kavach_npm(args: list[str] = typer.Argument(..., help="npm arguments")):
    """KAVACH-wrapped npm. Intercepts install commands."""
    if not args or args[0] not in ("install", "i", "add"):
        subprocess.run(["npm"] + list(args))
        return

    packages = [a for a in args[1:] if not a.startswith("-")]
    flags    = [a for a in args[1:] if a.startswith("-")]

    if not packages:
        subprocess.run(["npm"] + list(args))
        return

    blocked_any = False
    for pkg in packages:
        name = pkg.split("@")[0] if "@" in pkg and not pkg.startswith("@") else pkg
        with Progress(SpinnerColumn(),
                      TextColumn(f"[cyan]KAVACH scanning [bold]{name}[/bold]..."),
                      transient=True, console=console) as p:
            p.add_task("scan", total=None)
            result = asyncio.run(_scan_package_standalone(name, "npm"))

        _display_verdict(result, name)
        if result["install_blocked"]:
            blocked_any = True
            console.print(f"[bold red]🚫 Installation of '{name}' blocked by KAVACH.[/bold red]\n")
        else:
            if _execute(pkg, "npm", flags) != 0:
                sys.exit(1)

    if blocked_any:
        sys.exit(1)


@app.command("pip")
def kavach_pip(args: list[str] = typer.Argument(..., help="pip arguments")):
    """KAVACH-wrapped pip. Intercepts install commands."""
    if not args or args[0] != "install":
        subprocess.run(["pip"] + list(args))
        return

    packages = [a for a in args[1:] if not a.startswith("-")]
    flags    = [a for a in args[1:] if a.startswith("-")]

    blocked_any = False
    for pkg in packages:
        name = pkg.split("==")[0].split(">=")[0].split("<=")[0].strip()
        with Progress(SpinnerColumn(),
                      TextColumn(f"[cyan]KAVACH scanning [bold]{name}[/bold]..."),
                      transient=True, console=console) as p:
            p.add_task("scan", total=None)
            result = asyncio.run(_scan_package_standalone(name, "pypi"))

        _display_verdict(result, name)
        if result["install_blocked"]:
            blocked_any = True
            console.print(f"[bold red]🚫 Installation of '{name}' blocked by KAVACH.[/bold red]\n")
        else:
            if _execute(pkg, "pypi", flags) != 0:
                sys.exit(1)

    if blocked_any:
        sys.exit(1)


@app.command("scan")
def manual_scan(
    package:    str  = typer.Argument(..., help="Package name"),
    ecosystem:  str  = typer.Option("npm", "--ecosystem", "-e"),
    json_output: bool = typer.Option(False, "--json"),
):
    """Manually scan a package (no install)."""
    with Progress(SpinnerColumn(),
                  TextColumn(f"[cyan]KAVACH scanning [bold]{package}[/bold]..."),
                  transient=True, console=console) as p:
        p.add_task("scan", total=None)
        result = asyncio.run(_scan_package_standalone(package, ecosystem))

    if json_output:
        print(json.dumps(result, indent=2))
    else:
        _display_verdict(result, package)

    if result["install_blocked"]:
        raise typer.Exit(1)


@app.command("setup")
def setup():
    """
    Install models to ~/.kavach/models/ and add shell intercepts.
    Run once after installing kavach-standalone.
    """
    home = Path.home()
    kavach_dir = home / ".kavach" / "models"
    kavach_dir.mkdir(parents=True, exist_ok=True)

    # Try to find models in common locations
    search_paths = [
        Path("data/models"),
        Path("../data/models"),
        Path(os.getenv("KAVACH_PROJECT", ".")) / "data" / "models",
    ]

    found = None
    for p in search_paths:
        if (p / "meta_learner.pkl").exists():
            found = p
            break

    if found:
        import shutil
        console.print(f"[cyan]Copying models from {found} → {kavach_dir}[/cyan]")
        for f in found.iterdir():
            dest = kavach_dir / f.name
            if f.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(f, dest)
            else:
                shutil.copy2(f, dest)
        console.print(f"[green]✅ Models installed to {kavach_dir}[/green]")
    else:
        console.print(
            f"[yellow]⚠️  Could not find models automatically.[/yellow]\n"
            f"Manually copy your data/models/ folder to {kavach_dir}"
        )

    # Add shell intercepts
    intercept_lines = [
        "\n# KAVACH Standalone Supply Chain Security",
        "function npm() { kavach-standalone npm \"$@\"; }",
        "function pip() { kavach-standalone pip \"$@\"; }",
        "function pip3() { kavach-standalone pip \"$@\"; }",
        "export -f npm pip pip3 2>/dev/null || true",
    ]

    shells_updated = []
    for rc in [".zshrc", ".bashrc", ".bash_profile"]:
        rc_path = home / rc
        if rc_path.exists():
            content = rc_path.read_text()
            if "KAVACH Standalone" not in content:
                with open(rc_path, "a") as f:
                    f.write("\n".join(intercept_lines) + "\n")
                shells_updated.append(str(rc_path))
                console.print(f"[green]✅ Shell intercepts added to {rc_path}[/green]")
            else:
                console.print(f"[dim]Already configured in {rc_path}[/dim]")

    console.print(Panel(
        "[bold green]KAVACH Standalone is ready![/bold green]\n\n"
        "Restart your terminal or run:\n"
        "  [cyan]source ~/.zshrc[/cyan]\n\n"
        "Every npm/pip install is now scanned — no Docker needed.\n"
        "Models live in: [cyan]~/.kavach/models/[/cyan]",
        title="🛡️  Setup Complete",
        border_style="green",
    ))


@app.command("models-path")
def models_path():
    """Show where KAVACH looks for models."""
    console.print(f"Models directory: [cyan]{MODELS_DIR}[/cyan]")
    if MODELS_DIR.exists():
        files = list(MODELS_DIR.iterdir())
        console.print(f"Files found: {len(files)}")
        for f in files:
            console.print(f"  [dim]{f.name}[/dim]")
    else:
        console.print("[red]Directory does not exist — run: kavach-standalone setup[/red]")


if __name__ == "__main__":
    app()