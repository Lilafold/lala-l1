---
license: cc-by-nc-4.0
library_name: pytorch
tags:
  - cybersecurity
  - endpoint-detection
  - ocsf
  - sequence-model
  - research
pipeline_tag: other
---

# Lala L1

Lala-L1 is a small causal sequence model for Task A: score each endpoint event as part of a scripted attack step. It was trained by the lilafold lab on **owned, real-time Linux lab telemetry** (OCSF-normalized events). It is a research checkpoint, not a product, and it does **not** autonomously block or respond.

- Architecture: 4-layer causal transformer, `d=128`, context = last 256 events per host, 3-seed ensemble (~1–3M parameters).
- License: **CC BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0 International).
- Inference: `python src/infer.py --model checkpoints/lala-l1 --input <events.jsonl|campaign dir|-ocsf.tgz> --out scores.csv --fpr 0.001 --calibration host_baseline` (`train_cv` and `train_cutoffs` are also supported).
- Full model card: `MODEL_CARD.md`. Incident grouping (no retraining): `INCIDENTS.md`.

## Headline results (honest)

| Setting | Result |
|---|---|
| Frozen TEST (12 campaigns, 6 families, eval_freeze_v1) | **0.72 PR-AUC**; host-baseline **0.71 recall at 0.44% event FPR** |
| 5 unseen campaigns in the frozen-split protocol | **0.80 PR-AUC** |
| Small open-weight LLMs (e.g. Qwen-1.5B), same Task A setup | Lala L1 **matches or beats** them **within statistical uncertainty**, at a fraction of the compute |
| 30 fresh real-time validation runs (known families, `benign_mix v1`, stateful incident grouping) | **30/30** campaigns detected at **1.08 false incidents / host-day** |

Frozen TEST and the 30-run check are different evaluations. The 1.08 figure is not a general operating point.

## Limitations

- About **14%** false alarms on lookalike admin commands (active v1.1 work).
- **Five** attack families have no real-time frozen TEST data: collection staging, credential access, defense evasion, persistence, privilege escalation.
- On frozen TEST, stateful grouping still leaves **23.44 false incidents / host-day** (13/15 campaigns detected).
- Frozen TEST is old-collector telemetry; harder `benign_mix v2` mixed-background evaluations are still collecting.
- Calibration does not transfer cleanly across collector or benign generations; prefer host-baseline or same-schema real-time calibration.
- No autonomous blocking.

