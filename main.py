"""
main.py — Cloud Run entrypoint
--------------------------------
Wraps the AgriSentinel orchestrator in a minimal FastAPI service so it can
be deployed to Cloud Run and triggered either by a direct HTTP call (for
judges testing the "hosted Project URL" requirement) or by a Pub/Sub push
subscription (for the "new satellite pass landed" async trigger).
"""

import base64
import json
import logging

from fastapi import FastAPI, Request
from pydantic import BaseModel

from orchestrator.router import AgriSentinelOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agrisentinel.api")

app = FastAPI(title="AgriSentinel", version="0.1.0")
orchestrator = AgriSentinelOrchestrator()


class CycleRequest(BaseModel):
    plot_id: str
    intercropped: bool = True
    cloud_cover_pct: float = 0
    season_active: bool = True
    carbon_program_enrolled: bool = False
    claimed_practice: str = "unknown"


@app.get("/")
def health():
    """Health check — also what judges hit first to confirm the service is live."""
    return {"status": "ok", "service": "AgriSentinel orchestrator"}


@app.post("/run-cycle")
def run_cycle(req: CycleRequest):
    """Direct HTTP trigger — run one orchestration cycle for a plot and
    return the full report synchronously. This is the endpoint to use for
    live judging/demo purposes."""
    situation = {
        "intercropped": req.intercropped,
        "cloud_cover_pct": req.cloud_cover_pct,
        "season_active": req.season_active,
        "carbon_program_enrolled": req.carbon_program_enrolled,
        "claimed_practice": req.claimed_practice,
    }
    return orchestrator.run_cycle(req.plot_id, situation)


@app.post("/pubsub-trigger")
async def pubsub_trigger(request: Request):
    """Push-subscription endpoint. Pub/Sub POSTs a base64-encoded message
    here when a new satellite pass / farmer request event fires. See
    infra/README.md for the `gcloud pubsub subscriptions create --push-endpoint`
    command that wires this up.

    IMPORTANT: this handler always returns 200, even on a malformed
    payload. Pub/Sub push subscriptions treat any non-2xx response as
    "redeliver this message" and will retry indefinitely — so a single
    bad test message (e.g. published manually via the console with a
    non-JSON body) would otherwise loop forever and flood the logs. We
    log the parse failure clearly instead so it's still visible for
    debugging, but we ACK the message so the retry storm stops.
    """
    envelope = await request.json()
    message = envelope.get("message", {})
    data = message.get("data", "")

    payload: dict = {}
    if data:
        try:
            decoded = base64.b64decode(data).decode("utf-8")
            payload = json.loads(decoded) if decoded.strip() else {}
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "Pub/Sub message data was not valid JSON, skipping cycle. "
                "Raw decoded content: %r. Error: %s",
                data[:200], e,
            )
            return {"status": "skipped", "reason": "invalid_json_payload"}

    plot_id = payload.get("plot_id", "unknown-plot")
    situation = payload.get("situation", {})
    logger.info("Pub/Sub triggered cycle for plot %s", plot_id)

    report = orchestrator.run_cycle(plot_id, situation)
    return {"status": "processed", "plot_id": plot_id, "agents_run": report["agents_run"]}


if __name__ == "__main__":
    import uvicorn
    import os

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
