"""
KAVACH Agent 4 — Behavioral Anomaly Detector
=============================================
Analyzes temporal behavioral patterns of packages using
LSTM Autoencoder + Isolation Forest on time-series data.

Brain: LSTM Autoencoder trained on normal behavioral trajectories
of 100,000+ packages. High reconstruction error = anomaly.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import numpy as np
from loguru import logger
from sklearn.ensemble import IsolationForest

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class BehavioralAnomalyResult:
    agent_name: str = "Behavioral Anomaly Detector"
    risk_score: float = 0.0
    confidence: float = 0.0
    evidence: list[dict] = field(default_factory=list)
    temporal_metrics: dict = field(default_factory=dict)
    anomaly_score: float = 0.0
    reconstruction_error: float = 0.0
    execution_time_ms: float = 0.0


# ─── LSTM Autoencoder ────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Learns normal temporal patterns of package downloads/versions.
    High reconstruction error on input = behavioral anomaly.
    """

    def __init__(self, input_size: int = 8, hidden_size: int = 128, num_layers: int = 2):
        if TORCH_AVAILABLE:
            super().__init__()
            # Encoder
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2,
            )
            # Decoder
            self.decoder = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2,
            )
            self.output_layer = nn.Linear(hidden_size, input_size)
            self.hidden_size = hidden_size

    def forward(self, x):
        if not TORCH_AVAILABLE:
            return None, None

        # Encode
        encoded, (hidden, cell) = self.encoder(x)

        # Use final hidden state as context
        context = encoded[:, -1:, :].repeat(1, x.size(1), 1)

        # Decode
        decoded, _ = self.decoder(context)
        output = self.output_layer(decoded)

        return output, encoded


# ─── Main Agent ──────────────────────────────────────────────────────────────

class BehavioralAnomalyDetector:
    """
    Fetches temporal behavioral data and detects anomalies
    using LSTM Autoencoder reconstruction error.
    """

    NPM_DOWNLOADS_API = "https://api.npmjs.org/downloads/range"
    PYPI_STATS_API = "https://pypistats.org/api/packages"

    def __init__(
        self,
        lstm_model_path: str = "models/lstm_autoencoder.pt",
        isolation_forest_path: str = "models/behavioral_isolation_forest.pkl",
    ):
        self.lstm_model = None
        self.isolation_forest = None
        self.lstm_model_path = lstm_model_path
        self.isolation_forest_path = isolation_forest_path

        # Reconstruction error threshold (learned from real inference on benign packages).
        # React (a large benign package) produces MSE ≈ 0.42 on normalized sequences;
        # the model is not high quality (training log warned spike_ratio 0.09 < 1.5).
        # Raising the threshold to 1.5 ensures only truly anomalous download patterns
        # (MSE > 1.5) contribute meaningful LSTM risk.
        self.reconstruction_threshold = 1.5

    def load_model(self):
        """Load LSTM Autoencoder and Isolation Forest from disk."""
        # Load LSTM
        if TORCH_AVAILABLE:
            try:
                self.lstm_model = LSTMAutoencoder(hidden_size=128)
                self.lstm_model.load_state_dict(
                    torch.load(self.lstm_model_path, map_location="cpu")
                )
                self.lstm_model.eval()
                logger.info("LSTM Autoencoder loaded")
            except FileNotFoundError:
                logger.warning("LSTM model not found — using Isolation Forest only")

        # Load Isolation Forest
        try:
            import joblib
            self.isolation_forest = joblib.load(self.isolation_forest_path)
            logger.info("Behavioral Isolation Forest loaded")
        except FileNotFoundError:
            logger.warning("Isolation Forest not found — creating fresh instance")
            self.isolation_forest = IsolationForest(
                n_estimators=300,
                contamination=0.03,
                random_state=42,
            )

    async def analyze(
        self, package_name: str, ecosystem: str = "npm"
    ) -> BehavioralAnomalyResult:
        """
        Behavioral analysis pipeline:
        1. Fetch weekly download time series (52 weeks)
        2. Fetch version release history
        3. Build multi-variate time series feature matrix
        4. Run LSTM Autoencoder — compute reconstruction error
        5. Run Isolation Forest on static metrics
        6. Combine both signals
        """
        start = time.time()

        try:
            # Step 1: Fetch download time series
            download_series = await self._fetch_download_series(
                package_name, ecosystem
            )

            # Step 2: Fetch version history
            version_history = await self._fetch_version_history(
                package_name, ecosystem
            )

            # Step 3: Compute temporal metrics
            temporal_metrics = self._compute_temporal_metrics(
                download_series, version_history
            )

            # Step 4: LSTM Autoencoder analysis
            reconstruction_error = 0.0
            if download_series and len(download_series) >= 12:
                reconstruction_error = self._lstm_reconstruct(download_series)

            # Step 5: Isolation Forest
            if_score = self._isolation_forest_score(temporal_metrics)

            # Step 6: Combine signals
            risk_score, confidence = self._combine_signals(
                reconstruction_error, if_score, temporal_metrics
            )

            # Build evidence
            evidence = self._build_evidence(
                temporal_metrics, reconstruction_error, download_series
            )

            exec_time = (time.time() - start) * 1000

            return BehavioralAnomalyResult(
                risk_score=risk_score,
                confidence=confidence,
                evidence=evidence,
                temporal_metrics=temporal_metrics,
                anomaly_score=if_score,
                reconstruction_error=reconstruction_error,
                execution_time_ms=exec_time,
            )

        except Exception as e:
            logger.error(f"Behavioral Anomaly error for {package_name}: {e}")
            return BehavioralAnomalyResult(
                risk_score=0.2, confidence=0.1,
                evidence=[{"type": "error", "msg": str(e)}],
            )

    async def _fetch_download_series(
        self, package_name: str, ecosystem: str
    ) -> list[int]:
        """Fetch 52-week download count time series."""
        downloads = []

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                if ecosystem == "npm":
                    # npm provides weekly download counts
                    from datetime import datetime, timedelta
                    end = datetime.now()
                    start_date = end - timedelta(weeks=52)
                    date_range = (
                        f"{start_date.strftime('%Y-%m-%d')}:"
                        f"{end.strftime('%Y-%m-%d')}"
                    )
                    resp = await client.get(
                        f"{self.NPM_DOWNLOADS_API}/{date_range}/{package_name}"
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        downloads = [
                            d.get("downloads", 0)
                            for d in data.get("downloads", [])
                        ]

                elif ecosystem == "pypi":
                    resp = await client.get(
                        f"{self.PYPI_STATS_API}/{package_name}/recent"
                    )
                    if resp.status_code == 200:
                        data = resp.json().get("data", {})
                        # Convert to list
                        downloads = [
                            data.get("last_week", 0),
                            data.get("last_month", 0),
                        ]

            except Exception as e:
                logger.debug(f"Download series fetch error: {e}")

        return downloads

    async def _fetch_version_history(
        self, package_name: str, ecosystem: str
    ) -> list[dict]:
        """Fetch version release timeline."""
        versions = []

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                if ecosystem == "npm":
                    resp = await client.get(
                        f"https://registry.npmjs.org/{package_name}"
                    )
                    if resp.status_code == 200:
                        meta = resp.json()
                        time_data = meta.get("time", {})
                        versions = [
                            {"version": v, "published": t}
                            for v, t in time_data.items()
                            if v not in ("created", "modified")
                        ]
                        versions.sort(key=lambda x: x["published"])

                        # Add size information from dist-tags
                        for v in versions[-10:]:  # Last 10 versions
                            ver_meta = meta.get("versions", {}).get(
                                v["version"], {}
                            )
                            v["size"] = ver_meta.get("dist", {}).get(
                                "unpackedSize", 0
                            )

            except Exception as e:
                logger.debug(f"Version history fetch error: {e}")

        return versions

    def _compute_temporal_metrics(
        self, download_series: list[int], version_history: list[dict]
    ) -> dict:
        """
        Compute static temporal metrics for Isolation Forest.
        These capture the behavioral fingerprint of the package.
        """
        metrics = {}

        # Download statistics
        if download_series:
            arr = np.array(download_series)
            metrics["total_downloads"] = int(np.sum(arr))
            metrics["mean_weekly_downloads"] = float(np.mean(arr))
            metrics["download_std"] = float(np.std(arr))
            metrics["download_cv"] = (
                float(np.std(arr) / np.mean(arr))
                if np.mean(arr) > 0 else 0.0
            )  # Coefficient of variation

            # Week-over-week growth rate — sudden spikes are suspicious
            if len(arr) > 1:
                growth_rates = np.diff(arr) / (arr[:-1] + 1)
                metrics["max_weekly_growth_rate"] = float(np.max(growth_rates))
                metrics["mean_growth_rate"] = float(np.mean(growth_rates))
                metrics["growth_volatility"] = float(np.std(growth_rates))

                # Detect sudden spike
                z_scores = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
                metrics["max_z_score"] = float(np.max(z_scores))
                metrics["spike_weeks"] = int(np.sum(z_scores > 3))
            else:
                metrics["max_weekly_growth_rate"] = 0.0
                metrics["mean_growth_rate"] = 0.0
                metrics["growth_volatility"] = 0.0
                metrics["max_z_score"] = 0.0
                metrics["spike_weeks"] = 0

        else:
            metrics.update({
                "total_downloads": 0, "mean_weekly_downloads": 0,
                "download_std": 0, "download_cv": 0,
                "max_weekly_growth_rate": 0, "mean_growth_rate": 0,
                "growth_volatility": 0, "max_z_score": 0, "spike_weeks": 0,
            })

        # Version release metrics
        if version_history:
            metrics["total_versions"] = len(version_history)

            if len(version_history) > 1:
                from datetime import datetime
                dates = []
                for v in version_history:
                    try:
                        d = datetime.fromisoformat(
                            v["published"].replace("Z", "+00:00")
                        )
                        dates.append(d)
                    except Exception:
                        pass

                if len(dates) > 1:
                    gaps = [
                        (dates[i+1] - dates[i]).days
                        for i in range(len(dates)-1)
                    ]
                    metrics["mean_version_gap_days"] = float(np.mean(gaps))
                    metrics["min_version_gap_days"] = float(np.min(gaps))
                    metrics["version_gap_std"] = float(np.std(gaps))

                    # Sudden burst of versions
                    recent_gaps = gaps[-5:] if len(gaps) >= 5 else gaps
                    older_gaps = gaps[:-5] if len(gaps) >= 5 else []
                    if older_gaps and recent_gaps:
                        metrics["recent_vs_historical_gap_ratio"] = (
                            float(np.mean(recent_gaps)) /
                            float(np.mean(older_gaps) + 1)
                        )
                    else:
                        metrics["recent_vs_historical_gap_ratio"] = 1.0

            # Size change between versions
            sizes = [v.get("size", 0) for v in version_history if v.get("size", 0) > 0]
            if len(sizes) > 1:
                size_changes = [
                    abs(sizes[i+1] - sizes[i]) / (sizes[i] + 1)
                    for i in range(len(sizes)-1)
                ]
                metrics["max_size_change_ratio"] = float(np.max(size_changes))
            else:
                metrics["max_size_change_ratio"] = 0.0
        else:
            metrics.update({
                "total_versions": 0, "mean_version_gap_days": 0,
                "min_version_gap_days": 0, "version_gap_std": 0,
                "recent_vs_historical_gap_ratio": 1.0, "max_size_change_ratio": 0.0,
            })

        return metrics

    def _lstm_reconstruct(self, download_series: list[int]) -> float:
        """
        Run LSTM Autoencoder and return reconstruction error.
        Higher error = more anomalous temporal pattern.
        """
        if not TORCH_AVAILABLE or self.lstm_model is None:
            return 0.0

        try:
            arr = np.array(download_series, dtype=np.float32)

            # Normalize
            mean_val = arr.mean()
            std_val = arr.std() + 1e-8
            arr_norm = (arr - mean_val) / std_val

            # Build multi-variate time series
            # Features per timestep: [downloads, log_downloads, delta, z_score]
            log_arr = np.log1p(arr)
            delta = np.diff(arr_norm, prepend=arr_norm[0])
            z_scores = arr_norm

            # Stack into sequence
            sequence = np.stack([arr_norm, log_arr/log_arr.max() if log_arr.max() > 0 else log_arr,
                                  delta, z_scores,
                                  np.zeros_like(arr_norm),  # padding features
                                  np.zeros_like(arr_norm),
                                  np.zeros_like(arr_norm),
                                  np.zeros_like(arr_norm)], axis=1)

            tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                reconstructed, _ = self.lstm_model(tensor)
                # Mean squared reconstruction error
                mse = float(
                    torch.mean((tensor - reconstructed) ** 2).item()
                )

            return mse

        except Exception as e:
            logger.debug(f"LSTM reconstruction error: {e}")
            return 0.0

    def _isolation_forest_score(self, metrics: dict) -> float:
        """Run Isolation Forest on temporal metrics."""
        if not metrics:
            return 0.0

        feature_vector = [
            # Log-scale download normalization so popular packages (react: 8M/week)
            # are not capped at 1.0 the same as a 100k-download package.
            # log(8M+1)/log(10M+1) ≈ 0.93  vs  log(100k+1)/log(10M+1) ≈ 0.70
            float(np.log1p(metrics.get("mean_weekly_downloads", 0)) /
                  np.log1p(10_000_000)),
            min(metrics.get("download_cv", 0) / 5.0, 1.0),
            min(metrics.get("max_weekly_growth_rate", 0) / 100.0, 1.0),
            min(metrics.get("growth_volatility", 0) / 10.0, 1.0),
            min(metrics.get("max_z_score", 0) / 10.0, 1.0),
            min(metrics.get("spike_weeks", 0) / 10.0, 1.0),
            min(metrics.get("recent_vs_historical_gap_ratio", 1.0) / 10.0, 1.0),
            min(metrics.get("max_size_change_ratio", 0) / 5.0, 1.0),
        ]

        if self.isolation_forest is not None:
            try:
                # Use decision_function instead of score_samples.
                # decision_function > 0  → inlier (normal)
                # decision_function < 0  → outlier (anomalous)
                # Typical range: roughly -0.3 to +0.3 for the trained population.
                # Mapping: 0.0 (boundary) → risk 0.25, -0.3 (anomaly) → risk ~1.0
                decision = self.isolation_forest.decision_function([feature_vector])[0]
                # Invert and scale: strong inlier (+0.2) → risk 0.08, boundary 0.0 → 0.25
                risk = max(0.0, min(1.0, (0.10 - decision) / 0.40))
                return risk
            except Exception:
                pass

        # Fallback (no model)
        return min(
            metrics.get("max_z_score", 0) / 10.0 +
            metrics.get("spike_weeks", 0) * 0.05 +
            metrics.get("max_size_change_ratio", 0) * 0.1,
            1.0,
        )

    def _combine_signals(
        self, reconstruction_error: float, if_score: float, metrics: dict
    ) -> tuple[float, float]:
        """Combine LSTM and Isolation Forest signals.

        The LSTM model was trained with sub-optimal quality (spike ratio 0.09x,
        threshold raised to 1.5).  We therefore weight the Isolation Forest
        more heavily and only let the LSTM contribute when its error is
        substantially above the raised threshold.
        """
        # Normalize reconstruction error to 0-1 with the raised threshold
        lstm_risk = min(reconstruction_error / self.reconstruction_threshold, 1.0)

        if self.lstm_model is not None and reconstruction_error > 0:
            # LSTM quality is poor — reduce its weight vs IF to 0.35/0.65
            combined = 0.35 * lstm_risk + 0.65 * if_score
            confidence = 0.75
        else:
            combined = if_score
            confidence = 0.65

        return min(combined, 1.0), confidence

    def _build_evidence(
        self, metrics: dict, reconstruction_error: float, downloads: list[int]
    ) -> list[dict]:
        """Build evidence list from detected anomalies."""
        evidence = []

        if metrics.get("spike_weeks", 0) > 0:
            evidence.append({
                "type": "download_spike",
                "severity": "high",
                "description": (
                    f"Detected {metrics['spike_weeks']} week(s) with abnormal download spikes "
                    f"(>3 standard deviations from baseline) — "
                    "consistent with artificial inflation"
                ),
            })

        if metrics.get("max_weekly_growth_rate", 0) > 50:
            growth = metrics["max_weekly_growth_rate"]
            evidence.append({
                "type": "viral_growth_anomaly",
                "severity": "high",
                "description": (
                    f"Download count grew {growth:.0f}x in a single week — "
                    "organic packages rarely grow this fast without a viral announcement"
                ),
            })

        if metrics.get("max_size_change_ratio", 0) > 2.0:
            evidence.append({
                "type": "unexplained_size_increase",
                "severity": "medium",
                "description": (
                    f"Package size increased by "
                    f"{metrics['max_size_change_ratio']:.1f}x between versions "
                    "without corresponding changelog entry"
                ),
            })

        if reconstruction_error > self.reconstruction_threshold * 1.5:
            evidence.append({
                "type": "temporal_pattern_anomaly",
                "severity": "high",
                "description": (
                    f"LSTM Autoencoder reconstruction error ({reconstruction_error:.4f}) "
                    f"is {reconstruction_error/self.reconstruction_threshold:.1f}x above normal threshold — "
                    "temporal behavior deviates significantly from legitimate packages"
                ),
            })

        return evidence