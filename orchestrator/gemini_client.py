"""
gemini_client.py
-----------------
Thin wrapper around Gemini (via the unified google-genai SDK) used by every
agent in the AgriSentinel fleet. Centralizing this makes it trivial to:
  * swap models (e.g. gemini-3.5-flash -> a future release) in one place
  * add tool/function-calling schemas per agent
  * log every reasoning call for the Agent Observability requirement

ACCESS PATH: Vertex AI (preferred) vs. Gemini API key
--------------------------------------------------------
On Cloud Run / in production we call Gemini through VERTEX AI, not a bare
AI Studio API key. Reasons:
  1. Vertex AI quota is billed against the GCP project (the same billing
     account already backing Cloud Run/Firestore/Pub/Sub), not the AI
     Studio "Free Tier" bucket, which is capped at ~20 requests/day per
     model and is meant for prototyping in aistudio.google.com, not a
     deployed service making several calls per orchestration cycle.
  2. It authenticates via the Cloud Run service account's Application
     Default Credentials (ADC) automatically — no API key secret to
     manage at all in production.
  3. It's one of the two access paths the hackathon rules explicitly
     require ("Gemini 3.5 or newer accessed through Gemini API OR
     Vertex AI").

Locally (no ADC available), we fall back to a plain API key for quick
iteration — see .env.example. If neither is configured, we fall back
further to a deterministic offline stub so the pipeline is always
runnable for CI / judges who just clone and run without any credentials.

REQUIRED FOR VERTEX AI MODE:
  - Set env vars GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
    (e.g. us-central1) on the Cloud Run service.
  - Grant the Cloud Run service account the "Vertex AI User"
    (roles/aiplatform.user) IAM role.
  - aiplatform.googleapis.com must be enabled (already done in this repo's
    setup — see README "Spin-up Instructions").
"""

import os
import json
import logging
from typing import Any, Optional

logger = logging.getLogger("agrisentinel.gemini")

try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except ImportError:  # allows the repo to be inspected/run offline for demo
    _HAS_GENAI = False


DEFAULT_MODEL = os.getenv("AGRISENTINEL_MODEL", "gemini-3.5-flash")
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
# "global" (not a regional endpoint like us-central1) is required for
# several newer Gemini releases on Vertex AI, including gemini-3.5-flash,
# which is not yet published to every regional endpoint. Global also has
# the side benefit of higher availability / fewer 429s. See:
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")


class GeminiAgentClient:
    """
    Wraps a single Gemini "agent persona" — a system instruction + tool
    schema + model — so each sub-agent in orchestrator/router.py can reason
    over its domain (SAR imagery, spectral signatures, thermal bands, soil
    carbon models) and decide what action to take next.
    """

    def __init__(
        self,
        agent_name: str,
        system_instruction: str,
        tools: Optional[list] = None,
        model: str = DEFAULT_MODEL,
    ):
        self.agent_name = agent_name
        self.system_instruction = system_instruction
        self.tools = tools or []
        self.model = model
        self._client = None
        self._backend = "offline"

        if not _HAS_GENAI:
            logger.warning("[%s] google-genai not installed. Running OFFLINE.", agent_name)
        elif GCP_PROJECT:
            # Preferred production path: Vertex AI, billed to the GCP project,
            # authenticated via Cloud Run's Application Default Credentials.
            self._client = genai.Client(
                vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION
            )
            self._backend = "vertexai"
            logger.info(
                "[%s] Using Vertex AI backend (project=%s, location=%s)",
                agent_name, GCP_PROJECT, GCP_LOCATION,
            )
        elif os.getenv("GEMINI_API_KEY"):
            # Local-dev fallback: plain AI Studio API key. Subject to the
            # Free Tier daily quota — fine for a handful of local test runs,
            # not for the deployed service.
            self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            self._backend = "api_key"
            logger.info("[%s] Using Gemini API key backend (local-dev only)", agent_name)
        else:
            logger.warning(
                "[%s] No GOOGLE_CLOUD_PROJECT or GEMINI_API_KEY found. "
                "Running in OFFLINE / DEMO reasoning mode.",
                agent_name,
            )

    def reason(self, prompt: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Send the agent's situation to Gemini and get back a structured
        decision. Falls back to a deterministic rule-based stub when no
        API key is configured, so the pipeline is always runnable end to
        end for local testing / CI, without needing live credentials.
        """
        full_prompt = (
            f"{self.system_instruction}\n\n"
            f"CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
            f"TASK:\n{prompt}\n\n"
            f"Respond ONLY with strict JSON."
        )

        if self._client is not None:
            response = self._client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            try:
                return json.loads(response.text)
            except (json.JSONDecodeError, AttributeError):
                logger.error("[%s] Failed to parse Gemini response as JSON", self.agent_name)
                return {"error": "unparseable_response", "raw": getattr(response, "text", None)}

        # ---- OFFLINE FALLBACK (deterministic, for demo/CI reproducibility) ----
        return self._offline_stub(prompt, context)

    def _offline_stub(self, prompt: str, context: dict[str, Any]) -> dict[str, Any]:
        """Deterministic placeholder reasoning so `python demo.py` always works
        even without a Gemini API key — useful for judges who just clone & run."""
        return {
            "agent": self.agent_name,
            "mode": "offline_stub",
            "decision": "proceed",
            "note": (
                "Offline demo mode: no GEMINI_API_KEY set. "
                "This is a deterministic stand-in for the real Gemini reasoning call."
            ),
            "context_keys_seen": list(context.keys()),
        }
