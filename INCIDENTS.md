# Lala L1 incident grouping (v1)

No retraining. Per-event Lala L1 scores (eval_freeze_v1, real_only ensemble) are grouped into incidents by host + process tree + time.

- Immediate path: any event with score > high threshold
- Corroborated path: score > moderate threshold and ≥2 distinct behavior categories (not just repeated adjacent windows of the same behavior)
- Episode idle timeout: 15 min (used for missed malicious episodes)
- Attach window: 30 min (same host and process tree)
- Lineage: OCSF process GUID / parent fields when present; otherwise `pid:start_time`, plus parent links reconstructed from raw eBPF `F|` fork records when a campaign `-raw.tgz` is available (23 of 33 campaigns in this run)
- Thresholds tuned on TRAIN_ALLOWED real-time only (18 campaigns, unit-grouped 3-fold CV). Frozen TEST was not used for selection.

Selected thresholds: high = 0.7000, moderate = 0.1800 (CV detection 1.0, CV false incidents/host-day 5.0175504154519714).

Raw-threshold baseline uses the existing train-CV FPR 0.001 cutoff (0.2022).

Bootstrap 95% CIs are campaign-resampled (1000 draws). Time-to-first is seconds from attack start to the first true-positive incident (median / P95).

| split | false incidents / benign host-day | campaign detection | TTD s median / P95 (CI) | missed malicious episodes | raw FP alerts / host-hour / host-day | raw campaign detection |
|---|---|---|---|---|---|---|
| train_real (tune, not a claim) | 4.871 [1.570, 8.890] | 1.000 [1.000, 1.000] | 0.0 [0.0, 0.0] / 0.0 [0.0, 0.0] | 0/51 | 1.73 / 41.5 | 1.000 |
| frozen TEST | 23.440 [18.156, 29.991] | 0.867 [0.667, 1.000] | 0.0 [0.0, 0.0] / 0.0 [0.0, 0.0] | 62/248 | 6.82 / 163.6 | 1.000 |
| TEST unseen families (file_discovery, host_identity) | 20.154 [12.596, 30.076] | 1.000 [1.000, 1.000] | 0.0 [0.0, 0.0] / 0.0 [0.0, 0.0] | 3/58 | 4.39 / 105.2 | 1.000 |

Unseen campaigns are the frozen-TEST campaigns whose family has no TRAIN_ALLOWED real-time example: file_discovery.v1 (`cflin-p25`) and host_identity.v1 (`cflin-p22-rerun1`, `cflin-p23`).

Behavior categories are derived from process name / command / event class only (inventory, network, credentials, persistence, archive, packages, files, shell, plus class-level buckets). Label and technique fields are never used as features or as grouping keys.

On frozen TEST, incident grouping cuts raw false alerts from 163.6 / host-day (6.82 / host-hour; 1089 FP events) to 23.4 false incidents / host-day (156 incidents; about a 7× reduction). Campaign detection falls from 15/15 (raw) to 13/15: the missed pair is `cflin-p12` / `cflin-p17` (service_discovery.v1). Those campaigns do alert at the raw 0.202 threshold, but no window exceeded the high threshold 0.70 and the corroborated path did not fire (too few distinct behavior categories). Unseen TEST families (file_discovery, host_identity) were all detected.

Time-to-first-incident of 0 s means the first true-positive incident starts at attack start (attacks in this split are short).
