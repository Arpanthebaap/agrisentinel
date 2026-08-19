"""
cloud_piercing_agent.py  —  "The Cloud Piercer"
--------------------------------------------------
Problem: Under-Cloud Crop Monitoring (SAR-to-Optical Translation).

STATUS: REAL, TRAINABLE ARCHITECTURE (a compact U-Net / Pix2Pix generator)
implemented in PyTorch. For the hackathon demo, weights are randomly
initialized (untrained) since training a production-grade CycleGAN on
paired Sentinel-1/Sentinel-2 scenes takes GPU-hours we don't have in the
submission window — this is disclosed openly, see README "Honest Scope".
The AGENTIC value being demonstrated is the decision layer: Gemini decides
WHEN to invoke this model (cloud mask > threshold), and what confidence /
caveats to attach to the reconstructed imagery before it reaches a
disease-detection agent or a farmer's phone.

Production path (documented in docs/ROADMAP.md): fine-tune this same
architecture on the SEN12MS-CR or similar paired SAR/optical dataset.
"""

from __future__ import annotations
import numpy as np

from data.synthetic_data import synthetic_sar_scene, synthetic_optical_scene
from orchestrator.gemini_client import GeminiAgentClient

SYSTEM_INSTRUCTION = """You are the Cloud-Piercing Agent in the AgriSentinel
fleet. You are invoked only when optical satellite imagery is unavailable
due to cloud cover and a SAR-reconstructed optical estimate exists instead.
Given reconstruction confidence metrics, decide whether the estimate is
reliable enough to hand to the Stress Sentinel Agent, or whether the system
should wait for the next clear-sky optical pass. Be conservative — false
confidence in synthetic imagery could mislead a farmer or insurer."""

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if _HAS_TORCH:
    class UNetGenerator(nn.Module):
        """Compact Pix2Pix-style U-Net: 2-channel SAR (VV, VH) -> 3-channel
        pseudo-optical (RGB proxy for NDVI-relevant bands)."""

        def __init__(self, in_ch=2, out_ch=3, base=16):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2))
            self.enc2 = nn.Sequential(nn.Conv2d(base, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.LeakyReLU(0.2))
            self.enc3 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.BatchNorm2d(base * 4), nn.LeakyReLU(0.2))
            self.dec1 = nn.Sequential(nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.ReLU())
            self.dec2 = nn.Sequential(nn.ConvTranspose2d(base * 4, base, 4, 2, 1), nn.BatchNorm2d(base), nn.ReLU())
            self.dec3 = nn.Sequential(nn.ConvTranspose2d(base * 2, out_ch, 4, 2, 1), nn.Tanh())

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            d1 = self.dec1(e3)
            d2 = self.dec2(torch.cat([d1, e2], dim=1))
            d3 = self.dec3(torch.cat([d2, e1], dim=1))
            return d3


class CloudPiercingAgent:
    def __init__(self):
        self.llm = GeminiAgentClient(
            agent_name="cloud_piercing_agent",
            system_instruction=SYSTEM_INSTRUCTION,
        )
        self.model = UNetGenerator() if _HAS_TORCH else None

    def reconstruct(self, sar_scene: np.ndarray) -> tuple[np.ndarray, float]:
        """Run the SAR patch through the generator. Returns (image, confidence).
        Confidence here is a simple proxy (output variance vs. expected
        natural-image variance) — production version would use a trained
        discriminator score or ensemble disagreement."""
        if self.model is None:
            # numpy fallback if torch isn't installed in this environment
            pseudo = np.clip(sar_scene[..., :1].repeat(3, axis=-1) * 0.02 + 0.3, 0, 1)
            confidence = 0.35  # low confidence — explicitly not a trained model
            return pseudo, confidence

        x = torch.tensor(sar_scene, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
        img = ((out.squeeze(0).permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        confidence = float(1.0 - np.std(img))  # placeholder heuristic, documented as such
        return img, min(confidence, 0.5)  # capped: model is untrained in this demo run

    def run(self, plot_id: str, cloud_cover_pct: float, sar_scene: np.ndarray | None = None) -> dict:
        if sar_scene is None:
            sar_scene = synthetic_sar_scene()

        reconstructed, confidence = self.reconstruct(sar_scene)

        decision = self.llm.reason(
            prompt=(
                f"Cloud cover is {cloud_cover_pct}% over plot {plot_id}. SAR-to-optical "
                f"reconstruction confidence is {confidence:.2f} (0-1 scale, model is a "
                f"demo/untrained checkpoint). Decide: proceed to Stress Sentinel Agent "
                f"with a caveat, or wait for clear-sky optical pass?"
            ),
            context={"cloud_cover_pct": cloud_cover_pct, "confidence": confidence},
        )

        return {
            "agent": "cloud_piercing_agent",
            "plot_id": plot_id,
            "cloud_cover_pct": cloud_cover_pct,
            "reconstruction_confidence": confidence,
            "reconstructed_shape": list(reconstructed.shape),
            "gemini_decision": decision,
        }


if __name__ == "__main__":
    agent = CloudPiercingAgent()
    print(agent.run(plot_id="demo-plot-001", cloud_cover_pct=87.0))
