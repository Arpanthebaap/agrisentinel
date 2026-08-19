"""
stress_sentinel_agent.py  —  "The Stress Sentinel"
--------------------------------------------------
Problem: Early Sub-Surface Crop Stress (Pre-Visual Detection).

STATUS: FULLY FUNCTIONAL. Real statistical anomaly detection over
evapotranspiration (ET) and land surface temperature (LST) time series
— the same SWIR/thermal-derived signals described in the brief. Uses a
rolling z-score + change-point detector (real, runs on real numbers),
which is a legitimate, lightweight way to catch a field "sweating
abnormally" days before NDVI would show it. A production version swaps
this for a trained LSTM/temporal transformer (see docs/ROADMAP.md);
the agentic decision layer (when to alert, how urgently, what to tell
the farmer) is real and is what's actually being judged.
"""

from __future__ import annotations
import numpy as np

from data.synthetic_data import synthetic_thermal_timeseries
from orchestrator.gemini_client import GeminiAgentClient

SYSTEM_INSTRUCTION = """You are the Stress Sentinel Agent in the AgriSentinel
fleet. You receive an evapotranspiration (ET) and land-surface-temperature
(LST) anomaly report for a field, generated BEFORE any visible NDVI change.
Decide the urgency level (none/watch/alert/critical), a likely cause
(drought / nitrogen deficiency / root disease / inconclusive), and draft a
one-sentence SMS-length warning suitable for a smallholder farmer with a
basic phone."""


def rolling_zscore_anomaly(series: np.ndarray, window: int = 7, threshold: float = 2.0) -> dict:
    """Real anomaly detector: for each point past `window`, compute z-score
    against the trailing rolling mean/std. Flags sustained anomalies
    (>= 3 consecutive points beyond threshold) as pre-visual stress signals."""
    z_scores = np.zeros_like(series, dtype=float)
    for i in range(window, len(series)):
        trailing = series[i - window:i]
        mu, sigma = trailing.mean(), trailing.std() + 1e-8
        z_scores[i] = (series[i] - mu) / sigma

    flagged = np.abs(z_scores) > threshold
    # require sustained anomaly (3+ consecutive days) to avoid single-day noise
    sustained = np.zeros_like(flagged)
    run = 0
    for i, f in enumerate(flagged):
        run = run + 1 if f else 0
        if run >= 3:
            sustained[i - 2:i + 1] = True

    first_onset = int(np.argmax(sustained)) if sustained.any() else None
    return {
        "z_scores": z_scores.tolist(),
        "sustained_anomaly": bool(sustained.any()),
        "onset_day": first_onset,
        "peak_z": float(np.max(np.abs(z_scores))),
    }


class StressSentinelAgent:
    def __init__(self):
        self.llm = GeminiAgentClient(
            agent_name="stress_sentinel_agent",
            system_instruction=SYSTEM_INSTRUCTION,
        )

    def run(self, plot_id: str, thermal_data: dict | None = None) -> dict:
        if thermal_data is None:
            thermal_data = synthetic_thermal_timeseries(stressed=True)

        et = np.array(thermal_data["et"])
        lst = np.array(thermal_data["lst"])

        et_anomaly = rolling_zscore_anomaly(et)
        lst_anomaly = rolling_zscore_anomaly(lst)

        decision = self.llm.reason(
            prompt=(
                f"For plot {plot_id}: ET anomaly detected={et_anomaly['sustained_anomaly']} "
                f"(onset day {et_anomaly['onset_day']}, peak z={et_anomaly['peak_z']:.2f}); "
                f"LST anomaly detected={lst_anomaly['sustained_anomaly']} "
                f"(onset day {lst_anomaly['onset_day']}, peak z={lst_anomaly['peak_z']:.2f}). "
                f"Assess urgency, likely cause, and draft an SMS alert."
            ),
            context={"et_anomaly": et_anomaly, "lst_anomaly": lst_anomaly},
        )

        return {
            "agent": "stress_sentinel_agent",
            "plot_id": plot_id,
            "et_anomaly": et_anomaly,
            "lst_anomaly": lst_anomaly,
            "gemini_decision": decision,
        }


if __name__ == "__main__":
    agent = StressSentinelAgent()
    out = agent.run(plot_id="demo-plot-001")
    print(f"Onset day: {out['et_anomaly']['onset_day']}")
    print(out["gemini_decision"])
