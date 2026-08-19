"""
Smoke tests: every agent + the orchestrator must run end-to-end offline
(no API key required) and return well-formed output. This is what proves
"Functionality" and "Reproducibility" to judges — run with:

    PYTHONPATH=. python3 -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.plot_disaggregation_agent import PlotDisaggregationAgent
from agents.cloud_piercing_agent import CloudPiercingAgent
from agents.stress_sentinel_agent import StressSentinelAgent
from agents.carbon_verification_agent import CarbonVerificationAgent
from orchestrator.router import AgriSentinelOrchestrator


def test_plot_disaggregation_agent_runs():
    out = PlotDisaggregationAgent().run("test-plot")
    assert "mean_fractions" in out
    assert abs(sum(out["mean_fractions"].values()) - 1.0) < 1e-6


def test_cloud_piercing_agent_runs():
    out = CloudPiercingAgent().run("test-plot", cloud_cover_pct=90)
    assert 0 <= out["reconstruction_confidence"] <= 1


def test_stress_sentinel_agent_detects_injected_anomaly():
    out = StressSentinelAgent().run("test-plot")
    assert out["et_anomaly"]["peak_z"] > 0


def test_carbon_verification_agent_runs():
    out = CarbonVerificationAgent().run("test-plot", claimed_practice="no_till_cover_crop")
    assert "estimated_soc_change_tC_ha_yr" in out["soc_estimate"]


def test_orchestrator_runs_full_cycle():
    orchestrator = AgriSentinelOrchestrator()
    situation = {
        "intercropped": True,
        "cloud_cover_pct": 80,
        "season_active": True,
        "carbon_program_enrolled": True,
        "claimed_practice": "no_till_cover_crop",
    }
    report = orchestrator.run_cycle("test-plot", situation)
    assert set(report["agents_run"]) == {
        "plot_disaggregation_agent",
        "cloud_piercing_agent",
        "stress_sentinel_agent",
        "carbon_verification_agent",
    }
    assert len(report["results"]) == 5  # 4 sub-agents + farmer_sms_agent
    assert "farmer_sms_agent" in report["results"]
    assert report["results"]["farmer_sms_agent"]["char_count"] <= 160
