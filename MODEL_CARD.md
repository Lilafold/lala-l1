# Model card: Lala L1 (model id lala-l1, formerly cyberfm-seq-v1; Task A, Linux lab telemetry)

Date: 2026-11-13 (CET). Author: lilafold lab. Status: **research v1 checkpoint**. Weights license: **CC BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0 International). This is a research release, not production-ready.

## What it is
* Event-level detector for Task A (is this event part of a scripted attack step?). Output: a score per event plus an
  alert at a calibrated false-positive rate.
* Architecture: small causal transformer (4 layers, d=128, about 1–3M parameters). It runs over the last 256 events of a host
  stream. Each event is represented by hashed name, image and argv tokens (4096 buckets), event class, activity and uid bucket,
  argv and network/file flags, and label-free **novelty** features (first time this name or command line was seen on the host,
  plus running counts). Wall-clock timing is **deliberately excluded**: it encodes the 60x dilation artifact.
* Score = mean of 3 seeds. Training takes about 1–2 min per seed on the local RTX 3060. **No RunPod/cloud was used and $0 was spent.**
  A model this size doesn't need a cloud GPU.
* Code: `src/infer.py` (CLI), `src/seqmodel.py`, `src/incidents.py`, `src/features.py`.
* Checkpoints: `checkpoints/lala-l1/` (**Lala L1 / v1 primary**, 3-seed ensemble). The dilated-training checkpoint is an ablation only and is **not** part of this release.
* Inference: `python src/infer.py --model checkpoints/lala-l1 --input <events.jsonl|campaign dir|-ocsf.tgz> --out scores.csv --fpr 0.001 [--calibration host_baseline|train_cv]`.
  `--model lala-l1` is also accepted.
  Labels and label-derived keys in the input are stripped before featurization. A 7k-event campaign takes about 4 s on CPU.

## Leak fixes vs cyberfm-cloud-158-v1
cloud-158-v1 read the label (`is_mal`, `malicious_count`, `has_mal`) as input and calibrated on test. Round 2 strips
`label/step_id/technique_id/sub_technique_id/severity/mimics_step_id` before featurization and bans known shortcut cues
(relative position, PID, campaign length, control uids, wall-clock timing). Shortcut tests are in `tests/` (7 tests, all pass;
a positive control shows the check catches the old leak). Thresholds are never fit on TEST.

## Evaluation: Atlas eval_freeze_v1 (SPLIT.json sha256 8b3653d4…fd816), used exactly
* TEST = 15 real-time, gate-pass, pre-p34 campaigns (old collector: f666 pre-argv or f1a4). TRAIN_ALLOWED = 18 real-time
  (14 of them are pair-FAIL real legs p2000024–37, allowed for training; 3 are old-schema p15/p27/p31; plus p106) + 193 dilated (ablation only).
* Thresholds (primary): 3-fold unit-grouped out-of-fold calibration on TRAIN_ALLOWED real-time benign events.
* Unit = event. Unique-event scoring (each event scored once, causally). Positive rate on TEST is about 0.5%.
* "reference recall @ true test FPR" uses a threshold fit on TEST benign. It is reported **only** to separate ranking quality
  from calibration and is not an operating point.

### Headline: v1 primary (real-time training only), 6 scorable families pooled (12 TEST campaigns, 388 positive events)
| metric | value |
|---|---|
| PR-AUC (random = 0.005) | **0.72** (per-seed 0.60 / 0.66 / 0.71) |
| ROC-AUC | 0.98 |
| recall @ 0.1% FPR (train-CV threshold) | 0.81, but **realized TEST FPR 1.14%** (threshold does not transfer) |
| recall @ 1% FPR (train-CV threshold) | 0.99, but **realized TEST FPR 28.8%** (unusable) |
| recall @ 0.1% / 1% FPR with label-free host-baseline calibration (first 30% of each stream) | 0.71 at realized FPR 0.44% / 0.80 at realized FPR 0.84% |
| reference recall at true 0.1% / 1% TEST FPR | 0.65 / 0.80 |
| campaign recall (≥1 attack event alerted) | 12/12 |

### Per scorable family (v1 primary). Positive counts are tiny: treat as anecdotes, not estimates
| family | TEST camp. | pos | PR-AUC | ref. recall @0.1% | ref. recall @1% |
|---|---|---|---|---|---|
| user_owner_discovery | 2 | 140 | 0.94 | 0.79 | 1.00 |
| host_identity (effectively 1 run) | 2 | 80 | 0.94 | 0.89 | 0.98 |
| lolbin_admin_mix | 2 | 44 | 0.54 | 0.48 | 0.61 |
| shell_script_exec | 2 | 34 | 0.39 | 0.35 | 0.56 |
| netconn_discovery | 2 | 32 | 0.27 | 0.34 | 0.56 |
| endpoint_discovery (effectively 1 run) | 2 | 58 | 0.24 | 0.21 | 0.48 |
Not reportable (too little data): service_discovery (12 pos), file_discovery (1 campaign). **No TEST data at all**: collection_staging,
credential_access, defense_evasion, persistence, privilege_escalation. v1 makes **no claim** for these 5 families.

### With vs without dilated training data (measured on real-time TEST)
| model | training | PR-AUC scorable6 | ref. recall @0.1% / @1% | host-baseline recall @0.1% (realized FPR) | train-CV threshold realized FPR @0.1% |
|---|---|---|---|---|---|
| seq | real only (v1) | 0.72 | 0.65 / 0.80 | 0.71 (0.44%) | 1.1% |
| seq | real + 193 dilated (ABLATION) | **0.89** | 0.81 / 0.96 | 0.88 (0.23%) | 13.7% |
| GBDT | real only | 0.26 | 0.04 / 0.71 | 0.56 (1.16%) | 0.50% |
| GBDT | real + dilated (ABLATION) | 0.33 | 0.23 / 0.71 | 0.34 (0.57%) | 0.36% |

So dilated data **helps the sequence model rank** real-time attacks on TEST. This is consistent with train-side out-of-fold PR
(0.86 vs 0.78), so the conclusion does not rest on TEST alone. But it **wrecks threshold transfer**. In the family-LOFO protocol
(GBDT) it is mixed: macro PR goes 0.29 → 0.31, but macro recall@0.1% drops 0.48 → 0.42. In seq LOFO it helps ranking (macro PR 0.64 → 0.76)
but triples the median realized FPR at the 0.1% target (0.7% → 2.3%). Per lab
policy, v1 stays real-only. The real+dilated checkpoint is a flagged candidate, to be promoted only after it is confirmed on new
real-time data that nobody has looked at.

### Secondary: family leave-one-out (lab protocol; train-CV thresholds; unseen family evaluated on all its eligible real-time runs)
Macro over the 7 families with ≥30 positives (endpoint, file, host_identity, lolbin, netconn, shell, user_owner). defense_evasion has no eligible run;
service_discovery (12 positives) is listed but not averaged.
| model / training | macro PR-AUC | macro recall @0.1% (train-CV thr) | median realized FPR at that thr | macro recall @1% |
|---|---|---|---|---|
| seq, real only (v1) | 0.64 | 0.76 | 0.71% | 0.90 |
| seq, real + dilated (ABLATION) | 0.76 | 0.93 | 2.3% | 0.96 |
| GBDT, real only | 0.29 | 0.48 | 0.28% | 0.66 |
| GBDT, real + dilated (ABLATION) | 0.31 | 0.42 | 0.32% | 0.59 |
Per-family seq LOFO PR (real only / +dil): user_owner 0.90/0.98, file 0.95/0.99, host_identity 0.93/0.88, endpoint 0.46/0.65, lolbin 0.48/0.85,
netconn 0.40/0.61, shell 0.32/0.34, service 0.00/0.38. The unseen-family recall at a 0.1% train-CV threshold comes with realized FPRs
far above 0.1% for some families (endpoint 27% real-only), so recall@0.1% is not a usable operating point here.

### Task B (ATT&CK technique)
TEST events from the old collector carry no per-event technique labels, so Task B is scored at campaign level: the predicted
technique set (GBDT multiclass over the true attack events) against the manifest set. Real-only: set recall 0.43, precision 0.31;
+dilated: 0.42 / 0.27; dilated-only: 0.50 / 0.23. **Not usable yet**: most errors are confusion between discovery techniques
(T1082/T1016/T1033/T1087) whose commands overlap.

## Known limits (read before quoting anything)
1. **Small real-time data.** TEST is 15 campaigns and 438 positive events. Training real-time is 18 campaigns and 222 positives. Per-family
   numbers rest on 1–2 runs and 32–140 events. A ±0.1–0.2 swing between seeds is normal (PR 0.60–0.71).
2. **Calibration does not transfer across collector/benign generations.** Thresholds fit on (mostly new-schema) training data give
   10–30x the intended FPR on old-schema TEST. Use host-baseline calibration (label-free, assumes the first hours are clean), or
   calibrate per collector version with real-time benign-only runs.
3. **Frozen TEST is all old collector (pre-p34; 9 of 15 have no argv).** At the original freeze, no current-collector real-time
   campaign passed the gate (`realtime_post_p34 n = 0`). A separate 2026-11-13 check on 30 fresh real-time lab runs (known families,
   `benign_mix v1`) is reported below; it does **not** replace frozen TEST, and harder `benign_mix v2` evaluations are still collecting.
4. **Lab benign is unrealistically simple.** About 25% of all events are one heartbeat loop (`read gate; date; ls /usr/share`), and all campaigns
   use benign_mix v1. The unused benign_mix_v2 exists. Novelty features are strong partly because lab background barely changes.
   Expect much higher FPR on real hosts.
5. **Hard negatives are under-resisted.** Lookalike `benign_control` events are alerted at 39% (primary, train-CV threshold). In the
   in-distribution dilated test that figure was 7–10%, and removing controls from training raised it to 33%.
6. 5 of 13 families have no real-time TEST data. No claim is possible for them.
7. Event-level alerts cluster around attacks (post-attack benign events such as `debian-sa1` also score high). Stateful incident
   grouping (host + process tree + time; no retraining) is now implemented and cuts raw false alerts substantially, but it is not a
   deploy-ready operating point: frozen TEST still shows **23.44 false incidents per host-day** (13/15 campaigns detected). See below.
8. Task B/C are not production-grade. Task C (severity) was not evaluated: later campaigns use a lab-internal label schema, not severity_rubric.v2.

## License and what is excluded from this release
* **Weights and first-party inference/eval code** are released under **CC BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0 International). Commercial use is not permitted without a separate license.
* Training and the headline evaluations used **owned Linux lab telemetry** (OCSF-normalized endpoint events from isolated CyberFM lab VMs) plus this repository's code and checkpoints.
* These third-party datasets have **unresolved licenses and are explicitly excluded** from this release, from any accompanying data drop, and from any claim that Lala L1 v1 was trained or evaluated on them: **UNSW-NB15**, **OTRF**, **OpTC-corrected**, **CERT**, **BOTS**. Do not upload them to Hugging Face or any other public mirror with this model.

## Fresh real-time check (2026-11-13 CET) — known families, not a TEST substitute
Thirty overnight real-time campaigns from the lab (mostly attack types the model has already seen; `benign_mix v1` background). Harder `benign_mix v2` test runs were still collecting at the time of this card.

| metric | value |
|---|---|
| campaign detection | **30/30** |
| mean PR-AUC | 0.979 (lowest 0.753, a `service_discovery` run) |
| host-baseline recall (actual event FPR) | 0.995 at 0.035% FPR |
| train-CV recall (actual event FPR) | 0.973 at 0.013% FPR |
| stateful incident grouping | **1.08 false incidents / host-day** (95% CI 0.65–1.51); ~11× below 12.31 raw alerts / host-day; 30/30 still detected; median time-to-detect 0.017 s |

## Incident grouping on frozen TEST (no retraining)
Grouping uses host + process tree + time. Thresholds were tuned only on real-time training runs (high = 0.70, moderate = 0.18). Frozen TEST was not used for selection. Details: `INCIDENTS.md`.

| split | false incidents / host-day (95% CI) | campaign detection | notes |
|---|---|---|---|
| frozen TEST | **23.44** [18.16, 29.99] | 13/15 | raw event alerts were 163.6 / host-day (~7× reduction). Missed: two `service_discovery` runs that stayed under the 0.70 high-confidence cutoff without corroboration |
| TEST unseen families (`file_discovery`, `host_identity`) | 20.15 [12.60, 30.08] | 100% | raw alerts 105.2 / host-day |

## Unseen-campaign ranking and small-LLM comparison
On 5 unseen campaigns in the frozen-split protocol, Lala L1 reaches **0.80 PR-AUC**. In this Task A setup it **matches or beats** small open-weight LLMs such as Qwen-1.5B, **within statistical uncertainty**, at a fraction of the compute. That is not a general claim that Lala L1 outperforms LLMs outside this lab protocol.

## What v1 is not
* It does **not** autonomously block, isolate, or otherwise act on a host. It scores events and can group them into incidents.
* Lookalike admin commands still cause about **14%** false alarms and are active work for v1.1.
* Five attack families remain untested on real-time frozen TEST: `collection_staging`, `credential_access`, `defense_evasion`, `persistence`, `privilege_escalation`.
* The 1.08 false-incidents/host-day figure is for 30 fresh known-family, `benign_mix v1` runs. The honest frozen-TEST grouped rate is 23.44 / host-day. Do not quote 1.08 as a general operating point.
