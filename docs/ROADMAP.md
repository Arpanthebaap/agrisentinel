# Roadmap: demo → production

This documents exactly what changes to take each agent from its current
hackathon-submission state to a production-grade system. Nothing here is
built yet — it's included so judges (and future contributors) can see the
team understands the gap and has a concrete plan, not just a slide.

## Plot Disaggregation Agent
- **Now**: NNLS linear unmixing against 4 fixed endmember spectra; bicubic+
  sharpen stand-in for super-resolution.
- **Next**: Fine-tune an ESRGAN/Real-ESRGAN checkpoint on paired Sentinel-2
  (10m) → PlanetScope (3m) tiles over known intercropped regions (e.g.
  Malawi, Kenya smallholder plots — public datasets: Radiant Earth MLHub).
  Replace fixed endmembers with a learned endmember library per agro-
  ecological zone. Est. compute: single A100, ~2-3 days fine-tuning.

## Cloud Piercing Agent
- **Now**: Untrained Pix2Pix-style U-Net (architecture only).
- **Next**: Train on SEN12MS-CR (paired Sentinel-1/Sentinel-2, cloud-free
  reference) with an added perceptual + adversarial loss (full CycleGAN).
  Add a discriminator-based confidence score instead of the current
  variance heuristic. Est. compute: single A100, ~3-5 days.

## Stress Sentinel Agent
- **Now**: Rolling z-score + sustained-anomaly detector on ET/LST.
- **Next**: Replace with a temporal transformer trained on multi-year
  MODIS ET + Landsat LST time series labeled with ground-truth yield loss
  events, to reduce false positives from natural seasonal variation and
  detect anomaly *type* (drought vs. nitrogen vs. disease), not just
  presence.

## Carbon Verification Agent
- **Now**: Closed-form Q10-style SOC turnover estimate.
- **Next**: Full physics-informed neural network — a small MLP trained with
  a composite loss (data-fit + soft penalty enforcing the SOC turnover ODE
  as a physical constraint), fused with the SoilGrids geochemical baseline
  layer for prior soil carbon. This is the biggest lift of the four and the
  one most worth prioritizing post-hackathon given carbon-market demand.

## Cross-cutting (enterprise hardening)
- **Agent Identity**: per-agent service accounts with least-privilege IAM,
  not yet wired.
- **Agent Gateway**: currently the orchestrator calls sub-agents directly
  in-process; production routes every call through a gateway enforcing
  rate limits and auth.
- **Model Armor**: no prompt-injection/PII-leak guardrails yet on the
  Gemini reasoning calls — required before any real farmer PII flows
  through this system.
