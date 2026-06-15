"""
KAVACH Data Collection Script — OSSF Edition
==============================================
Reads real malicious package data directly from the cloned OSSF
malicious-packages repository (222,354 confirmed malicious packages).

Prerequisites:
  git clone https://github.com/ossf/malicious-packages data/ossf-malicious

Usage:
  python scripts/collect_training_data.py
"""

import asyncio
import json
import math
import os
import re
import sys
import tarfile
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import httpx
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DATA_DIR      = Path("data/raw");      DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path("data/processed"); PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OSSF_DIR      = Path("data/ossf-malicious/osv")

NPM_REGISTRY  = "https://registry.npmjs.org"
PYPI_REGISTRY = "https://pypi.org/pypi"


# ─── Feature extraction (identical to Agent 1 inference) ─────────────────────

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())

_TRUSTED_DOMAINS = {
    "github.com", "githubusercontent.com", "npmjs.com", "pypi.org",
    "registry.npmjs.org", "files.pythonhosted.org", "cloudflare.com",
    "amazonaws.com", "googleapis.com", "google.com", "microsoft.com",
    "unpkg.com", "jsdelivr.net", "nodejs.org", "mozilla.org",
    "httpbin.org", "example.com", "example.org", "axios-http.com",
    "lodash.com", "reactjs.org", "w3.org", "ecma-international.org",
    "developer.mozilla.org", "stackoverflow.com", "wikipedia.org",
}

def _is_trusted_domain(domain: str) -> bool:
    return any(domain == t or domain.endswith("." + t) for t in _TRUSTED_DOMAINS)

def _extract_code_features(source: str, filename: str) -> dict:
    lines = source.split("\n")
    total_lines = max(len(lines), 1)

    string_literals = re.findall(r'"([^"]{20,})"', source) + \
                      re.findall(r"'([^']{20,})'", source)
    entropies = [_shannon_entropy(s) for s in string_literals]

    network_count = sum(len(re.findall(p, source)) for p in [
        r"require\(['\"]http[s]?['\"]", r"require\(['\"]net['\"]",
        r"fetch\(", r"XMLHttpRequest", r"urllib\.request",
        r"requests\.(get|post|put)", r"socket\.connect", r"dns\.resolve",
        r"http\.get\(", r"http\.post\(", r"https\.get\(",
    ])

    eval_count = sum(len(re.findall(p, source)) for p in [
        r"\beval\s*\(", r"\bexec\s*\(", r"Function\s*\(",
        r"new\s+Function", r"__import__\s*\(", r"compile\s*\(",
        r"execSync\s*\(", r"spawnSync\s*\(",
    ])

    env_count = sum(len(re.findall(p, source)) for p in [
        r"process\.env\.", r"os\.environ", r"os\.getenv\(", r"getenv\(",
    ])

    fs_count = sum(len(re.findall(p, source)) for p in [
        r"fs\.read", r"fs\.write", r"os\.walk", r"glob\.glob",
        r"open\s*\(", r"readdir", r"readdirSync", r"readFileSync",
    ])

    has_postinstall = 1 if (
        "package.json" in filename and
        "postinstall" in source.lower()
    ) else 0

    identifiers = re.findall(r'\b([a-zA-Z_]\w*)\b', source)
    short_ids = sum(1 for i in identifiers if len(i) == 1)
    obfusc_ratio = short_ids / max(len(identifiers), 1)

    b64_count = sum(len(re.findall(p, source)) for p in [
        r"atob\(", r"base64\.b64decode",
        r"Buffer\.from\([^,]+,\s*['\"]base64['\"]",
        r"decode\(['\"]base64['\"]", r"\.toString\(['\"]base64['\"]",
    ])

    domains = re.findall(r'https?://([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', source)
    suspicious_domain_count = sum(
        1 for d in domains if not _is_trusted_domain(d)
    )

    url_shortener_count = sum(
        1 for d in domains
        if any(s in d for s in ["bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly"])
    )

    dangerous_imports = 0
    if filename.endswith(".py"):
        dangerous_imports = sum(len(re.findall(p, source)) for p in [
            r"\bsubprocess\b", r"\bctypes\b", r"\bpickle\b",
            r"\bmarshal\b", r"\bpty\b", r"\bos\.system\b",
        ])

    return {
        "max_string_entropy":               max(entropies) if entropies else 0.0,
        "mean_string_entropy":              float(sum(entropies) / len(entropies)) if entropies else 0.0,
        "high_entropy_string_count":        sum(1 for e in entropies if e > 4.5),
        "network_call_count":               network_count,
        "network_calls_per_100_lines":      (network_count / total_lines) * 100,
        "dynamic_execution_count":          eval_count,
        "env_access_count":                 env_count,
        "filesystem_access_count":          fs_count,
        "has_postinstall":                  has_postinstall,
        "obfuscated_identifier_ratio":      obfusc_ratio,
        "base64_decode_count":              b64_count,
        "suspicious_external_domain_count": suspicious_domain_count,
        "dangerous_import_count":           dangerous_imports,
        "max_call_nesting_depth":           0,
        "lambda_count":                     len(re.findall(r'\blambda\b', source)),
        "magic_method_overrides":           len(re.findall(
            r'def (__reduce__|__reduce_ex__|__getattr__|__setattr__)', source
        )),
        "url_shortener_count":              url_shortener_count,
    }

def _aggregate_features(feature_list: list[dict]) -> dict:
    if not feature_list:
        return {}
    aggregated = {}
    all_keys = set().union(*feature_list)
    for key in all_keys:
        values = [f.get(key, 0) for f in feature_list if key in f]
        if values:
            if "count" in key or "has_" in key:
                aggregated[key] = sum(values)
            else:
                aggregated[key] = max(values)
    return aggregated


# ─── OSSF Dataset Reader ──────────────────────────────────────────────────────

def read_ossf_malicious_packages(
    max_npm: int = 3000,
    max_pypi: int = 1000,
) -> list[dict]:
    """
    Read confirmed malicious packages from the cloned OSSF repository.
    Parses all JSON files from osv/ (excluding withdrawn).
    Returns list of {name, ecosystem, summary, vuln_id, label=1}.
    """
    if not OSSF_DIR.exists():
        logger.error(f"OSSF directory not found: {OSSF_DIR}")
        logger.error("Run: git clone https://github.com/ossf/malicious-packages data/ossf-malicious")
        return []

    logger.info(f"Reading OSSF malicious packages from {OSSF_DIR}...")

    packages = []
    seen = set()
    npm_count = 0
    pypi_count = 0
    skipped_withdrawn = 0
    parse_errors = 0

    # Walk all JSON files recursively
    for json_path in OSSF_DIR.rglob("*.json"):
        # Skip withdrawn packages — they were cleaned up and are no longer malicious
        if "withdrawn" in str(json_path):
            skipped_withdrawn += 1
            continue

        try:
            with open(json_path) as f:
                data = json.load(f)

            # Skip if explicitly marked withdrawn in content
            if data.get("withdrawn"):
                skipped_withdrawn += 1
                continue

            summary = data.get("summary", "") or data.get("details", "")[:300]

            for affected in data.get("affected", []):
                pkg = affected.get("package", {})
                name = pkg.get("name", "").strip()
                eco_raw = pkg.get("ecosystem", "").lower()

                if not name:
                    continue

                # Normalize ecosystem
                if eco_raw in ("npm",):
                    eco = "npm"
                elif eco_raw in ("pypi", "pip"):
                    eco = "pypi"
                else:
                    continue

                # Apply caps
                if eco == "npm" and npm_count >= max_npm:
                    continue
                if eco == "pypi" and pypi_count >= max_pypi:
                    continue

                key = f"{name}:{eco}"
                if key in seen:
                    continue
                seen.add(key)

                packages.append({
                    "name":      name,
                    "ecosystem": eco,
                    "vuln_id":   data.get("id", json_path.stem),
                    "summary":   summary[:400] if summary else f"Malicious {eco} package",
                    "severity":  "CRITICAL",
                    "label":     1,
                })

                if eco == "npm":
                    npm_count += 1
                else:
                    pypi_count += 1

        except Exception as e:
            parse_errors += 1
            logger.debug(f"Parse error {json_path}: {e}")

    logger.info(
        f"OSSF packages loaded: {npm_count} npm + {pypi_count} pypi = {len(packages)} total "
        f"(skipped {skipped_withdrawn} withdrawn, {parse_errors} parse errors)"
    )
    return packages


# ─── Package source downloader ────────────────────────────────────────────────

async def _download_and_extract_features(
    client: httpx.AsyncClient,
    package_name: str,
    ecosystem: str,
) -> dict | None:
    """Download package source and extract feature vector."""
    try:
        if ecosystem == "npm":
            resp = await client.get(
                f"{NPM_REGISTRY}/{package_name}/latest", timeout=10.0
            )
            if resp.status_code != 200:
                return None
            meta = resp.json()
            tarball_url = meta.get("dist", {}).get("tarball")
            if not tarball_url:
                return None

            tarball_resp = await client.get(tarball_url, timeout=20.0)
            if tarball_resp.status_code != 200:
                return None

            with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as f:
                f.write(tarball_resp.content)
                tmp_path = f.name

            feature_list = []
            try:
                with tarfile.open(tmp_path, "r:gz") as tar:
                    for member in tar.getmembers():
                        name = member.name
                        if not name.endswith((".js", ".ts", ".json")):
                            continue
                        if any(x in name for x in (
                            "node_modules", ".min.", ".bundle.",
                            "dist/", "__tests__", ".spec.", ".test."
                        )):
                            continue
                        if member.size > 150_000:
                            continue
                        try:
                            fobj = tar.extractfile(member)
                            if fobj:
                                content = fobj.read().decode("utf-8", errors="ignore")
                                feature_list.append(
                                    _extract_code_features(content, name)
                                )
                        except Exception:
                            pass
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            return _aggregate_features(feature_list) if feature_list else None

        elif ecosystem == "pypi":
            resp = await client.get(
                f"{PYPI_REGISTRY}/{package_name}/json", timeout=10.0
            )
            if resp.status_code != 200:
                return None
            urls = resp.json().get("urls", [])
            wheel  = next((u for u in urls if u["packagetype"] == "bdist_wheel"), None)
            sdist  = next((u for u in urls if u["packagetype"] == "sdist"), None)
            target = wheel or sdist
            if not target:
                return None

            pkg_resp = await client.get(target["url"], timeout=20.0)
            if pkg_resp.status_code != 200:
                return None

            suffix = ".whl" if wheel else ".tar.gz"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(pkg_resp.content)
                tmp_path = f.name

            feature_list = []
            try:
                try:
                    with zipfile.ZipFile(tmp_path) as z:
                        for name in z.namelist():
                            if not name.endswith(".py"):
                                continue
                            if any(x in name for x in ("__pycache__", ".pyc", "test_", "_test")):
                                continue
                            content = z.read(name).decode("utf-8", errors="ignore")
                            feature_list.append(_extract_code_features(content, name))
                except zipfile.BadZipFile:
                    with tarfile.open(tmp_path, "r:gz") as tar:
                        for member in tar.getmembers():
                            if not member.name.endswith(".py"):
                                continue
                            fobj = tar.extractfile(member)
                            if fobj:
                                content = fobj.read().decode("utf-8", errors="ignore")
                                feature_list.append(
                                    _extract_code_features(content, member.name)
                                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            return _aggregate_features(feature_list) if feature_list else None

    except Exception as e:
        logger.debug(f"Feature extraction failed for {package_name}/{ecosystem}: {e}")
        return None


async def extract_features_parallel(
    packages: list[dict],
    max_concurrent: int = 8,
) -> list[dict]:
    """Download and extract features for a batch of packages in parallel."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results   = []

    async def _one(pkg: dict) -> dict | None:
        async with semaphore:
            async with httpx.AsyncClient(
                timeout=20.0, follow_redirects=True
            ) as client:
                features = await _download_and_extract_features(
                    client, pkg["name"], pkg["ecosystem"]
                )
                if features is None:
                    return None
                return {
                    "name":      pkg["name"],
                    "ecosystem": pkg["ecosystem"],
                    "label":     pkg.get("label", 0),
                    "vuln_id":   pkg.get("vuln_id", ""),
                    "summary":   pkg.get("summary", ""),
                    "severity":  pkg.get("severity", ""),
                    **features,
                }

    tasks = [_one(pkg) for pkg in packages]
    total = len(tasks)

    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        if result is not None:
            results.append(result)
        if (i + 1) % 50 == 0 or (i + 1) == total:
            logger.info(
                f"  Feature extraction: {i+1}/{total} processed, "
                f"{len(results)} successful ({len(results)/(i+1)*100:.0f}%)"
            )

    return results


# ─── Benign packages ──────────────────────────────────────────────────────────

def get_benign_packages() -> list[dict]:
    """Well-known benign packages with long clean histories."""
    npm_benign = [
        "lodash", "express", "react", "react-dom", "axios", "moment",
        "chalk", "commander", "dotenv", "webpack", "typescript", "eslint",
        "jest", "mocha", "prettier", "nodemon", "cors", "body-parser",
        "jsonwebtoken", "bcryptjs", "mongoose", "uuid", "dayjs", "date-fns",
        "ramda", "rxjs", "zod", "yup", "joi", "ajv", "debug", "ms",
        "semver", "glob", "mkdirp", "rimraf", "cross-env", "classnames",
        "prop-types", "immer", "zustand", "mobx", "redux", "d3", "three",
        "nanoid", "crypto-js", "qs", "ws", "got", "node-fetch", "pino",
        "winston", "form-data", "fastify", "next", "vue", "svelte", "vite",
        "rollup", "inquirer", "ora", "boxen", "execa", "cross-spawn",
        "p-limit", "p-queue", "p-retry", "p-map", "async", "bluebird",
        "mime", "mime-types", "multer", "lru-cache", "keyv", "retry",
        "string-width", "wrap-ansi", "open", "conf", "http-errors",
        "strip-ansi", "minimatch", "micromatch", "fast-glob", "globby",
        "chokidar", "fs-extra", "graceful-fs", "path-to-regexp", "slash",
        "json5", "js-yaml", "csv-parse", "xml2js", "marked", "gray-matter",
        "luxon", "chrono-node", "validator", "express-validator",
        "cosmiconfig", "cuid", "ulid", "morgan", "log4js", "socket.io",
        "graphql", "graphql-tools", "sharp", "jimp", "pdfkit", "pdf-parse",
        "xlsx", "nodemailer", "bull", "bullmq", "agenda", "node-cron",
        "passport", "passport-jwt", "passport-local", "helmet",
        "compression", "cookie-parser", "express-session", "rate-limiter-flexible",
        "sequelize", "typeorm", "knex", "pg", "mysql2", "better-sqlite3",
        "ioredis", "redis", "mongodb", "connect-redis",
        "supertest", "nock", "sinon", "chai", "expect", "vitest",
        "@testing-library/react", "cypress", "playwright",
        "husky", "lint-staged", "standard", "xo",
        "tailwindcss", "postcss", "autoprefixer", "sass",
        "framer-motion", "react-spring", "recharts", "chart.js",
        "@reduxjs/toolkit", "redux-saga", "redux-thunk", "recoil",
        "socket.io-client", "eventsource", "apollo-server",
        "class-validator", "class-transformer",
        "aws-sdk", "@aws-sdk/client-s3", "firebase-admin",
        "dotenv-safe", "convict", "nconf",
        "winston-transport", "pino-pretty",
        "superagent", "undici", "bent",
        "archiver", "adm-zip", "tar",
        "qrcode", "jsbarcode", "canvas",
        "jsonschema", "tv4", "revalidator",
        "moment-timezone", "spacetime", "fecha",
        "lodash-es", "just-clone", "klona",
        "deepmerge", "merge-deep", "defaults-deep",
    ]

    pypi_benign = [
        "requests", "numpy", "pandas", "scipy", "matplotlib", "seaborn",
        "scikit-learn", "tensorflow", "torch", "keras", "transformers",
        "xgboost", "lightgbm", "shap", "lime", "statsmodels", "optuna",
        "ray", "dask", "polars", "pyarrow", "fastparquet",
        "nltk", "spacy", "gensim", "textblob", "sentence-transformers",
        "tokenizers", "sentencepiece", "pillow", "opencv-python-headless",
        "scikit-image", "imageio", "torchvision", "albumentations",
        "fastapi", "flask", "django", "tornado", "aiohttp", "starlette",
        "uvicorn", "gunicorn", "hypercorn", "waitress",
        "sqlalchemy", "alembic", "psycopg2-binary", "asyncpg",
        "pymongo", "motor", "redis", "aioredis", "peewee",
        "httpx", "urllib3", "certifi", "charset-normalizer",
        "cryptography", "bcrypt", "pyjwt", "paramiko", "pyopenssl",
        "python-jose", "passlib", "itsdangerous", "authlib",
        "click", "typer", "rich", "colorama", "prompt-toolkit", "tqdm",
        "python-dotenv", "pydantic", "pydantic-settings", "dynaconf",
        "loguru", "structlog", "python-json-logger",
        "pytest", "pytest-asyncio", "pytest-cov", "pytest-mock",
        "hypothesis", "factory-boy", "faker", "mock", "responses",
        "black", "isort", "flake8", "mypy", "pylint", "bandit", "ruff",
        "marshmallow", "attrs", "cattrs", "pyyaml", "ujson", "orjson",
        "msgpack", "protobuf", "grpcio",
        "celery", "rq", "dramatiq", "huey", "arq",
        "schedule", "apscheduler", "croniter",
        "openpyxl", "xlrd", "python-docx", "python-pptx",
        "pypdf", "pdfplumber", "reportlab",
        "arrow", "pendulum", "pytz", "dateutil",
        "boto3", "botocore", "google-cloud-storage",
        "docker", "kubernetes",
        "beautifulsoup4", "lxml", "scrapy", "selenium", "playwright",
        "plotly", "bokeh", "altair", "streamlit", "gradio", "dash",
        "networkx", "sympy", "numba", "cython",
        "humanize", "tabulate", "joblib", "boltons", "toolz",
        "cachetools", "tenacity", "backoff", "watchdog", "psutil",
        "validators", "email-validator", "phonenumbers",
        "python-dateutil", "isodate",
        "wrapt", "decorator", "typing-extensions",
        "sentry-sdk", "prometheus-client", "statsd",
        "jinja2", "mako", "pygments",
        "pathspec", "filelock", "portalocker",
        "chardet", "ftfy", "unidecode",
        "regex", "pyparsing", "lark",
        "python-slugify", "bleach",
        "webencodings", "idna",
        "shortuuid", "ulid-py",
    ]

    packages = []
    seen = set()
    for name in npm_benign:
        if name not in seen:
            seen.add(name)
            packages.append({"name": name, "ecosystem": "npm", "label": 0})
    for name in pypi_benign:
        if name not in seen:
            seen.add(name)
            packages.append({"name": name, "ecosystem": "pypi", "label": 0})

    return packages


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    logger.info("=" * 60)
    logger.info("KAVACH Data Collection — OSSF Edition")
    logger.info("=" * 60)
    start = time.time()

    # ── Step 1: Load malicious packages from OSSF dataset ────────────────────
    logger.info("\n[1/4] Loading malicious packages from OSSF dataset...")
    all_malicious = read_ossf_malicious_packages(max_npm=3000, max_pypi=1000)

    if not all_malicious:
        logger.error("No malicious packages loaded. Is data/ossf-malicious cloned?")
        sys.exit(1)

    # Shuffle so we get a mix of attack types in the capped download batch
    import random
    random.seed(42)
    random.shuffle(all_malicious)

    # Save raw list
    with open(DATA_DIR / "malicious_packages.json", "w") as f:
        json.dump(all_malicious, f, indent=2)
    logger.info(f"  Saved {len(all_malicious)} malicious package names to data/raw/")

    # ── Step 2: Load benign packages ──────────────────────────────────────────
    logger.info("\n[2/4] Loading benign package list...")
    all_benign = get_benign_packages()
    logger.info(f"  {len(all_benign)} benign packages")

    # ── Step 3: Download source + extract features ────────────────────────────
    # Cap malicious downloads to 1500 — registries delete most malware quickly
    # so we expect ~30-40% success rate = ~450-600 real feature vectors
    mal_to_download = all_malicious[:1500]
    logger.info(f"\n[3/4] Extracting features from malicious packages "
                f"({len(mal_to_download)} attempts, expect ~30-40% success)...")
    mal_with_features = await extract_features_parallel(
        mal_to_download, max_concurrent=8
    )
    logger.info(f"  ✅ Real malicious features extracted: {len(mal_with_features)}")

    logger.info(f"\n[4/4] Extracting features from benign packages "
                f"({len(all_benign)} packages)...")
    ben_with_features = await extract_features_parallel(
        all_benign, max_concurrent=8
    )
    logger.info(f"  ✅ Benign features extracted: {len(ben_with_features)}")

    # ── Step 4: Build balanced dataset ───────────────────────────────────────
    # Use all real malicious with features.
    # Cap benign to 2x malicious for a 33/67 split (slightly benign-heavy
    # is better than even — reduces false positives in production).
    n_mal = len(mal_with_features)
    random.shuffle(ben_with_features)
    ben_capped = ben_with_features[:n_mal * 2]

    combined = mal_with_features + ben_capped
    random.shuffle(combined)

    n_mal_final = sum(1 for r in combined if r["label"] == 1)
    n_ben_final = sum(1 for r in combined if r["label"] == 0)

    # Save datasets
    with open(PROCESSED_DIR / "malicious_with_features.json", "w") as f:
        json.dump(mal_with_features, f, indent=2)
    with open(PROCESSED_DIR / "benign_with_features.json", "w") as f:
        json.dump(ben_with_features, f, indent=2)
    with open(PROCESSED_DIR / "training_dataset.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Also save OSSF summaries for SBERT training
    ossf_summaries = [
        {"name": p["name"], "summary": p["summary"], "vuln_id": p["vuln_id"]}
        for p in all_malicious if p.get("summary")
    ]
    with open(PROCESSED_DIR / "ossf_summaries.json", "w") as f:
        json.dump(ossf_summaries, f, indent=2)

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"Collection complete in {elapsed:.0f}s")
    logger.info(f"  Real malicious samples: {n_mal_final}")
    logger.info(f"  Benign samples:         {n_ben_final}")
    logger.info(f"  Total dataset:          {len(combined)}")
    logger.info(f"  Malicious rate:         {n_mal_final/max(len(combined),1)*100:.1f}%")
    logger.info(f"  OSSF summaries saved:   {len(ossf_summaries)} (for SBERT training)")
    logger.info(f"  Saved to: {PROCESSED_DIR}/")
    logger.info(f"\nNext: python scripts/train_all_models.py")


if __name__ == "__main__":
    asyncio.run(main())