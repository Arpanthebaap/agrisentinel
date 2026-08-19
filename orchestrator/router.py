"""
router.py  —  The AgriSentinel Orchestrator
------------------------------------------------
This is the top-level agent. It is the "Fortified Enterprise Fleet"
centerpiece: a Gemini-driven controller that receives a farm plot's
current conditions (season stage, cloud cover, last-known crop mix,
carbon-credit enrollment status) and DECIDES which specialized sub-agent(s)
to invoke, in what order, and how to reconcile their outputs into one
farmer/insurer/carbon-buyer facing report.

Maps to the enterprise capabilities the hackathon rules call out:
  - Agent Registry      -> AGENT_REGISTRY below (catalog + versioning stub)
  - Agent Runtime        -> orchestrator/runtime.py (async job execution)
  - Memory Bank          -> orchestrator/memory_bank.py (cross-season state)
  - Agent Identity/Gateway/Model Armor -> orchestrator/security.py (stubs,
    documented as the production hardening layer — see docs/ROADMAP.md)
  - Agent Observability  -> every agent call is logged via `log_event()`
"""

from __future__ import annotations
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from agents.plot_disaggregation_agent import PlotDisaggregationAgent
from agents.cloud_piercing_agent import CloudPiercingAgent
from agents.stress_sentinel_agent import StressSentinelAgent
from agents.carbon_verification_agent import CarbonVerificationAgent
from agents.farmer_sms_agent import FarmerSMSAgent
from orchestrator.gemini_client import GeminiAgentClient
from orchestrator.memory_bank import MemoryBank

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agrisentinel.orchestrator")

AGENT_REGISTRY = {
    "plot_disaggregation_agent": {
        "version": "0.1.0",
        "owner_team": "remote-sensing",
        "trigger": "on_new_optical_pass",
        "class": PlotDisaggregationAgent,
    },
    "cloud_piercing_agent": {
        "version": "0.1.0",
        "owner_team": "remote-sensing",
        "trigger": "on_cloud_cover_above_threshold",
        "class": CloudPiercingAgent,
    },
    "stress_sentinel_agent": {
        "version": "0.1.0",
        "owner_team": "agronomy",
        "trigger": "on_growing_season_active",
        "class": StressSentinelAgent,
    },
    "carbon_verification_agent": {
        "version": "0.1.0",
        "owner_team": "carbon-markets",
        "trigger": "on_carbon_credit_claim",
        "class": CarbonVerificationAgent,
    },
}

ORCHESTRATOR_SYSTEM_INSTRUCTION = """You are the AgriSentinel Orchestrator,
a fleet controller for four specialized agricultural remote-sensing agents.
Given a plot's current situation (season stage, cloud cover %, whether it
is intercropped, whether it's enrolled in a carbon program), decide the
ORDERED LIST of sub-agents to invoke this cycle and why. Only invoke agents
that are actually relevant this cycle — do not run agents unnecessarily.
Respond with a JSON object: {"invoke": [agent_names...], "reasoning": "..."}
"""


def log_event(event: dict[str, Any]) -> None:
    """Stand-in for OpenTelemetry-compliant audit logging (Agent
    Observability requirement). In production this exports to Cloud
    Logging / Cloud Trace; here it's structured stdout logging so every
    decision in the demo video is visibly auditable."""
    logger.info(json.dumps(event, default=str))


class AgriSentinelOrchestrator:
    def __init__(self):
        self.planner = GeminiAgentClient(
            agent_name="orchestrator",
            system_instruction=ORCHESTRATOR_SYSTEM_INSTRUCTION,
        )
        self.memory = MemoryBank()
        self._agents = {name: meta["class"]() for name, meta in AGENT_REGISTRY.items()}
        # Farmer SMS agent (Gemma) is not part of the parallel plan — it
        # runs LAST, consuming the other agents' combined output rather
        # than raw satellite data, so it must wait for them to finish.
        self._farmer_sms_agent = FarmerSMSAgent()

    def plan(self, plot_id: str, situation: dict[str, Any]) -> list[str]:
        """Ask Gemini which agents are relevant this cycle, given the plot's
        live situation. Falls back to explicit rule-based routing if Gemini
        is unavailable, so the fleet never stalls."""
        plan = self.planner.reason(
            prompt=(
                f"Plan this cycle's agent invocations for plot {plot_id}. "
                f"You MUST only choose from these exact agent names: "
                f"{list(AGENT_REGISTRY.keys())}. Do not invent or rename agents."
            ),
            context=situation,
        )
        invoke = plan.get("invoke")
        if invoke:
            # Defensive filter: Gemini occasionally returns a descriptive
            # label instead of the exact registry key (e.g. "Synthetic
            # Aperture Radar (SAR)" instead of "cloud_piercing_agent").
            # Silently drop anything unrecognized rather than crashing the
            # cycle on a KeyError — log it so it's visible in Observability.
            valid = [a for a in invoke if a in AGENT_REGISTRY]
            dropped = [a for a in invoke if a not in AGENT_REGISTRY]
            if dropped:
                log_event({"event": "plan_invalid_agents_dropped", "plot_id": plot_id, "dropped": dropped})
            if valid:
                return valid
            # if Gemini returned nothing usable, fall through to rule-based plan

        # deterministic rule-based fallback (mirrors AGENT_REGISTRY triggers)
        invoke = []
        if situation.get("intercropped"):
            invoke.append("plot_disaggregation_agent")
        if situation.get("cloud_cover_pct", 0) > 40:
            invoke.append("cloud_piercing_agent")
        if situation.get("season_active"):
            invoke.append("stress_sentinel_agent")
        if situation.get("carbon_program_enrolled"):
            invoke.append("carbon_verification_agent")
        return invoke

    def _invoke_agent(self, agent_name: str, plot_id: str, situation: dict[str, Any]) -> Any:
        agent = self._agents[agent_name]
        log_event({"event": "agent_invoked", "plot_id": plot_id, "agent": agent_name})
        if agent_name == "cloud_piercing_agent":
            out = agent.run(plot_id, cloud_cover_pct=situation.get("cloud_cover_pct", 0))
        elif agent_name == "carbon_verification_agent":
            out = agent.run(plot_id, claimed_practice=situation.get("claimed_practice", "unknown"))
        else:
            out = agent.run(plot_id)
        log_event({"event": "agent_completed", "plot_id": plot_id, "agent": agent_name})
        return out

    def run_cycle(self, plot_id: str, situation: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        log_event({"event": "cycle_start", "plot_id": plot_id, "situation": situation})

        plan = self.plan(plot_id, situation)
        log_event({"event": "plan_decided", "plot_id": plot_id, "agents_to_run": plan})

        # Sub-agents are independent (each targets a different sensor
        # modality) so we run them concurrently rather than sequentially.
        # This is what took the cycle from ~47s to a few seconds on Cloud
        # Run: 4 sequential Gemini calls -> 4 parallel Gemini calls.
        results = {}
        with ThreadPoolExecutor(max_workers=max(len(plan), 1)) as pool:
            futures = {
                pool.submit(self._invoke_agent, agent_name, plot_id, situation): agent_name
                for agent_name in plan
            }
            for future in as_completed(futures):
                agent_name = futures[future]
                results[agent_name] = future.result()

        self.memory.append(plot_id, {"situation": situation, "results_summary": list(results.keys())})

        # Last-mile translation step (Gemma): compress the fleet's combined
        # technical findings into one SMS the farmer can actually read.
        if results:
            log_event({"event": "agent_invoked", "plot_id": plot_id, "agent": "farmer_sms_agent"})
            sms_result = self._farmer_sms_agent.run(plot_id, results)
            results["farmer_sms_agent"] = sms_result
            log_event({"event": "agent_completed", "plot_id": plot_id, "agent": "farmer_sms_agent"})

        report = {
            "plot_id": plot_id,
            "cycle_duration_sec": round(time.time() - start, 3),
            "agents_run": plan,
            "results": results,
            "plot_history_length": len(self.memory.get(plot_id)),
        }
        log_event({"event": "cycle_complete", "plot_id": plot_id, "duration": report["cycle_duration_sec"]})
        return report


if __name__ == "__main__":
    orchestrator = AgriSentinelOrchestrator()
    demo_situation = {
        "intercropped": True,
        "cloud_cover_pct": 78,
        "season_active": True,
        "carbon_program_enrolled": True,
        "claimed_practice": "no_till_cover_crop",
    }
    result = orchestrator.run_cycle("demo-plot-001", demo_situation)
    print(json.dumps(result, indent=2, default=str))
