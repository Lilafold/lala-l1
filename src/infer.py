#!/usr/bin/env python3
"""Lala L1 (model id lala-l1, formerly cyberfm-seq-v1) inference. Scores every event of an OCSF-ish Linux events.jsonl (or a campaign dir /
-ocsf.tgz bundle) with the leak-free sequence model. Labels in the input are ignored (stripped before
featurization). Usage:
  infer.py --model checkpoints/lala-l1 --input /path/events.jsonl|campaign_dir|bundle.tgz --out scores.csv [--fpr 0.001]
  infer.py --model lala-l1 --input ...   # alias for checkpoints/lala-l1
The former path checkpoints/cyberfm-seq-v1-eval_freeze/real_only is still accepted.
Output CSV: idx (line number in events.jsonl), time, process_name, cmd (truncated), score, alert.

Robustness: events.jsonl is streamed (not loaded as a single bytes blob). Malformed JSON lines are
skipped with a warning. A missing events.jsonl, an empty file, or a one-event file warns and is
skipped instead of crashing. --calibration train_cutoffs applies the training-set thresholds from
the model config (same cutoffs as train_cv) and does not need a clean host baseline.
"""
import argparse, io, json, sys, tarfile, numpy as np, pandas as pd, orjson, scipy.sparse as sp, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Name aliases (paths under the repo / archive root). Existing absolute or relative paths still work.
MODEL_ALIASES = {
    "lala-l1": ROOT / "checkpoints" / "lala-l1",
    "cyberfm-seq-v1": ROOT / "checkpoints" / "cyberfm-seq-v1-eval_freeze" / "real_only",
    "lala-l1-ablation-dilated": ROOT / "checkpoints" / "lala-l1-ablation-dilated",
}
# Former ablation path, used when the lala-l1-ablation-dilated alias dir is not present.
_ABLATION_FALLBACK = ROOT / "checkpoints" / "cyberfm-seq-v1-eval_freeze" / "real+dilated_ABLATION"

def resolve_model_dir(spec):
    p = Path(spec)
    if p.exists():
        return p
    alias = MODEL_ALIASES.get(spec)
    if alias is not None and alias.exists():
        return alias
    if spec == "lala-l1-ablation-dilated" and _ABLATION_FALLBACK.exists():
        return _ABLATION_FALLBACK
    rel = ROOT / spec
    if rel.exists():
        return rel
    return p
from seqmodel import SeqDetector, event_arrays, score_campaign
from sklearn.feature_extraction.text import HashingVectorizer
FORBIDDEN = {"label", "step_id", "mimics_step_id", "technique_id", "sub_technique_id", "severity", "synthetic", "coverage_heartbeat"}
KEEP = ["time", "class_uid", "activity_id", "process_pid", "process_name", "process_image", "user_uid", "process_cmd_line",
        "argv_argc", "dst_ip", "dst_port", "status_code", "file_path"]
MIN_EVENTS = 2  # a single event is not a usable sequence; skip with a warning
OUT_COLS = ["idx", "time", "process_name", "cmd", "score", "alert"]
TRAIN_CUTOFF_MODES = ("train_cv", "train_cutoffs")

def _warn(msg):
    print(f"warning: {msg}", file=sys.stderr)

def _open_events_stream(inp):
    """Open events.jsonl as a streamed binary file-like. Returns (label, fh, closer) or (None, None, None)."""
    p = Path(inp)
    if p.is_dir():
        try:
            p = next(p.rglob("events.jsonl"))
        except StopIteration:
            _warn(f"missing events.jsonl in directory {inp}")
            return None, None, None
        fh = p.open("rb")
        return str(p), fh, fh.close
    if not p.exists():
        _warn(f"input not found: {inp}")
        return None, None, None
    if str(p).endswith(".tgz") or str(p).endswith(".tar.gz"):
        try:
            tf = tarfile.open(p)
        except Exception as e:
            _warn(f"cannot open bundle {p}: {e}")
            return None, None, None
        try:
            m = next(m for m in tf.getmembers() if m.name.endswith("/events.jsonl") and m.name.count("/") == 1)
        except StopIteration:
            tf.close()
            _warn(f"missing events.jsonl in bundle {p}")
            return None, None, None
        rawf = tf.extractfile(m)
        if rawf is None:
            tf.close()
            _warn(f"missing events.jsonl in bundle {p}")
            return None, None, None
        def _close():
            rawf.close()
            tf.close()
        return f"{p}:{m.name}", rawf, _close
    fh = p.open("rb")
    return str(p), fh, fh.close

def read_events(inp, warn=True):
    """Stream an events.jsonl / campaign dir / .tgz bundle into a DataFrame.

    Does not slurp the whole file with Path.read_bytes. Blank lines are ignored. Malformed JSON
    lines are skipped (warning). A missing events.jsonl returns an empty frame and a warning
    instead of raising. idx is the 0-based line number in the original file (same as before).
    Labels and other forbidden keys are stripped before featurization.
    """
    label, fh, closer = _open_events_stream(inp)
    recs = []
    n_bad = 0
    try:
        if fh is None:
            return pd.DataFrame(columns=KEEP + ["idx"])
        for i, raw in enumerate(fh):
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            line = raw.rstrip(b"\r\n")
            if not line.strip():
                continue
            try:
                e = orjson.loads(line)
            except orjson.JSONDecodeError as ex:
                n_bad += 1
                if warn:
                    preview = line[:120].decode("utf-8", "replace")
                    _warn(f"skipping malformed JSON at {label}:{i}: {ex}; {preview!r}")
                continue
            if not isinstance(e, dict):
                n_bad += 1
                if warn:
                    _warn(f"skipping non-object JSON at {label}:{i}")
                continue
            e = {k: v for k, v in e.items() if k not in FORBIDDEN}
            r = {k: e.get(k) for k in KEEP}
            r["idx"] = i
            recs.append(r)
    finally:
        if closer is not None:
            closer()
    if warn and n_bad:
        _warn(f"skipped {n_bad} malformed line(s) in {label}")
    if not recs:
        return pd.DataFrame(columns=KEEP + ["idx"])
    return pd.DataFrame.from_records(recs)

def featurize(ev):
    """Single-stream re-implementation of src/features.py (kept in sync; see tests/test_infer_parity.py)."""
    ev = ev.copy()
    for c in ["class_uid", "activity_id", "process_pid", "user_uid", "argv_argc", "dst_port", "status_code"]:
        ev[c] = pd.to_numeric(ev[c], errors="coerce").astype("float64")
    ev["t"] = pd.to_datetime(ev["time"], utc=True, errors="coerce").astype("datetime64[us, UTC]").astype("int64")
    ev = ev.sort_values(["t", "idx"], kind="stable").reset_index(drop=True)
    F = pd.DataFrame(index=ev.index)
    F["class_uid"] = ev.class_uid.fillna(-1); F["activity_id"] = ev.activity_id.fillna(-1)
    u = ev.user_uid; F["uid_bucket"] = np.select([u.isna(), u == 0, u < 1000], [-1, 0, 1], 2)
    cmd = ev.process_cmd_line.fillna("").astype(str)
    F["has_cmd"] = (cmd != "").astype(int); F["argc"] = ev.argv_argc.fillna(-1); F["cmd_len"] = cmd.str.len()
    F["has_dst"] = ev.dst_ip.notna().astype(int); F["dst_port"] = ev.dst_port.fillna(-1)
    F["status_code"] = ev.status_code.fillna(-999); F["has_file"] = ev.file_path.notna().astype(int)
    name = ev.process_name.fillna("<none>").astype(str); img = ev.process_image.fillna("").astype(str)
    F["new_name"] = (~name.duplicated()).astype(int); F["new_cmd"] = (~cmd.duplicated()).astype(int); F["new_image"] = (~img.duplicated()).astype(int)
    F["log_cnt_name_sofar"] = np.log1p(name.groupby(name).cumcount()); F["log_cnt_cmd_sofar"] = np.log1p(cmd.groupby(cmd).cumcount())
    for w in (32, 256):
        F[f"new_cmd_last{w}"] = F.new_cmd.rolling(w, min_periods=1).sum(); F[f"new_name_last{w}"] = F.new_name.rolling(w, min_periods=1).sum()
    docs = ("n:" + name + " i:" + img + " " + cmd.str.replace(r"[\s/=,;:'\"()\[\]{}|&<>]+", " ", regex=True).str.slice(0, 400))
    hv = HashingVectorizer(n_features=2**12, token_pattern=r"[^ ]+", lowercase=True, alternate_sign=False, norm=None, binary=True, dtype=np.float32)
    return ev, F.astype(np.float32), hv.transform(docs.values).tocsr()

def load_model(mdir, device):
    cfg = json.loads((Path(mdir) / "config.json").read_text())
    m = SeqDetector(d=cfg["d"], layers=cfg["layers"], L=cfg["L"]).to(device)
    m.load_state_dict(torch.load(Path(mdir) / "model.pt", map_location=device)); m.eval()
    return m, cfg

def score_events(ev_df, mdir, device="cpu"):
    """Single model dir or ensemble dir (config.json with 'members'); ensemble = mean of member scores."""
    cfg = json.loads((Path(mdir) / "config.json").read_text())
    ev, F, T = featurize(ev_df)
    ss = []
    for md in ([Path(mdir) / x for x in cfg["members"]] if "members" in cfg else [Path(mdir)]):
        m, mc = load_model(md, device)
        arr, _, _ = event_arrays(F, T, np.arange(len(F)), np.array(mc["mean"], np.float32), np.array(mc["std"], np.float32))
        ss.append(score_campaign(m, arr, torch.device(device), mc["L"], mc["stride"]))
    return ev, np.mean(ss, 0), cfg

def _threshold(cfg, scores, calibration, fpr, baseline_frac):
    """Return (thr, baseline_n) where baseline_n is 0 when no host baseline is used."""
    if calibration in TRAIN_CUTOFF_MODES:
        if "thresholds" not in cfg or fpr not in cfg["thresholds"]:
            raise SystemExit(f"error: model config has no training-set cutoff for FPR {fpr}")
        return cfg["thresholds"][fpr], 0
    n0 = int(baseline_frac * len(scores))
    if n0 < 1:
        raise ValueError(f"host_baseline window is empty (n={len(scores)}, baseline_frac={baseline_frac}); "
                         f"need more events or use --calibration train_cutoffs")
    return float(np.quantile(scores[:n0], 1 - float(fpr), method="higher")), n0

def _empty_scores():
    return pd.DataFrame(columns=OUT_COLS)

def _write_scores(out_df, dest):
    out_df.to_csv(sys.stdout if dest == "-" else dest, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="-"); ap.add_argument("--fpr", default="0.001"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--calibration", choices=["train_cv", "host_baseline", "train_cutoffs"], default="host_baseline",
                    help="train_cv: thresholds from config (fit on TRAIN_ALLOWED real-time OOF); "
                         "train_cutoffs: same training-set cutoffs, no host baseline required; "
                         "host_baseline: (1-fpr) quantile of this stream's first --baseline_frac events (assumed clean)")
    ap.add_argument("--baseline_frac", type=float, default=0.3)
    a = ap.parse_args()
    ev = read_events(a.input)
    n = 0 if ev is None or ev.empty else len(ev)
    if n == 0:
        _warn(f"no events to score in {a.input} (empty or missing events.jsonl); skipping")
        _write_scores(_empty_scores(), a.out)
        print(f"events=0 alerts=0 thr=nan ({a.calibration}, target FPR {a.fpr})", file=sys.stderr)
        sys.exit(0)
    if n < MIN_EVENTS:
        _warn(f"only {n} event(s) in {a.input}; need at least {MIN_EVENTS} to score; skipping")
        _write_scores(_empty_scores(), a.out)
        print(f"events={n} alerts=0 thr=nan ({a.calibration}, target FPR {a.fpr})", file=sys.stderr)
        sys.exit(0)
    ev, s, cfg = score_events(ev, resolve_model_dir(a.model), a.device)
    try:
        thr, n0 = _threshold(cfg, s, a.calibration, a.fpr, a.baseline_frac)
    except ValueError as e:
        _warn(str(e) + "; skipping")
        _write_scores(_empty_scores(), a.out)
        print(f"events={len(s)} alerts=0 thr=nan ({a.calibration}, target FPR {a.fpr})", file=sys.stderr)
        sys.exit(0)
    out = pd.DataFrame({"idx": ev.idx, "time": ev.time, "process_name": ev.process_name, "cmd": ev.process_cmd_line.fillna("").astype(str).str.slice(0, 160),
                        "score": s, "alert": s > thr})
    if a.calibration == "host_baseline": out.loc[:n0 - 1, "alert"] = False  # baseline period is not scored
    _write_scores(out, a.out)
    print(f"events={len(out)} alerts={int(out.alert.sum())} thr={thr:.4g} ({a.calibration}, target FPR {a.fpr})", file=sys.stderr)
