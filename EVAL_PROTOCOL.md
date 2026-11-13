# Round-2 strict evaluation protocol (frozen 2026-11-12, v1)

Everything below is implemented in `src/r2lib.py`, `src/make_splits.py` and `results/splits_v1.json`.
Any reported Task A number must follow it or say which rule it breaks.

## 1. Inputs that are forbidden as features
`label`, `step_id`, `mimics_step_id`, `technique_id`, `sub_technique_id`, `severity`, `synthetic`,
`coverage_heartbeat`, and anything derived from them (incl. the JSON key-set/schema of an event, which
differs for labeled events). They are stripped before featurization (`pipeline/prepare_dataset_noleak.py`,
`src/infer.py`). Known spurious cues are also banned from production features but kept as *probes*:
relative position in the campaign, raw PID / PID jumps, campaign length, uid 1001/65534 (control users),
and raw wall-clock timing (distorted by 60x dilation).

## 2. Frozen splits (`results/splits_v1.json`)
| set | content | use |
|---|---|---|
| `campaign_random` train/val/test | 204 current-schema 60x-dilated campaigns, split by `base_id` (reruns stay together), stratified by family: 133/29/42 | train / threshold / in-distribution test |
| `rt_new` | 16 current-schema **real-time** campaigns (p2000024–37, p73, p106) | real-time held-out, never trained on |
| `rt_legacy_all` | 21 old-collector/old-benign-generator campaigns (p11–p21, p31–p33, kvm-pilot, p22-rerun1, p23, p25, p27 + 2 others) | legacy held-out (schema shift) |
| `rt_legacy_atlas` | validated p11–p21, p31–p33 (p22 has no events) | reported separately |
| `canaries_pending` | canary campaigns p74, p75, p193 | join the real-time held-out set when exported |
| `leave_family_out[f]` | all dilated campaigns of family f | test; train = other families (train+test pool), val = val minus f |
| `host_held_out[h]` | all dilated campaigns of host h | test; train = other hosts |

New campaigns are only ever *added* (new held-out or new train pool); existing assignments never move.

## 3. Metrics (all unique-event, no overlapping-window double counting)
* Task A unit = event (`label == malicious` vs everything else, incl. `benign_control` hard negatives).
* PR-AUC (average precision; random = positive rate ≈ 0.0005), ROC-AUC.
* Operating points: thresholds fitted **only on val benign events** at FPR 0.1% and 1% (tie-safe:
  alert iff score > (1-f) quantile). Reported on test: recall, realized FPR, FP count, control-FPR
  (share of `benign_control` events alerted), step recall (≥1 event of an attack step alerted),
  campaign recall. "oracle_recall" = threshold fitted on test (reference only, never quoted as result).
* Breakdowns: per family, per technique, per host, dilated vs real-time.

## 4. Shortcut tests (must pass before any training run counts)
`tests/test_no_shortcut.py`: features invariant to labels and label-derived keys; featurizer refuses
label keys; no single feature column and no depth-2 tree reaches AUC ≥ 0.999 on real campaigns; a
positive control proves the check catches the cloud-158-v1 leak. `tests/test_infer_parity.py`: inference
features == training features. `pipeline/train_cloud_production_r2.py` refuses to train if any input
column is a perfect label oracle on val. A probe experiment (exp1 `all+tok+probes`) quantifies how much
the banned cues would inflate scores.
