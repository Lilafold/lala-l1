#!/usr/bin/env python3
"""Ingest helper: read cflin-*-ocsf.tgz bundles and build data/events.parquet +
data/campaigns.csv. Writes only under this repository root.

Source directories come from $CYBERFM_EXPORT_DIRS (os.pathsep-separated).
If unset, defaults to ./data/export.
"""
import hashlib, io, json, os, re, sys, tarfile
from pathlib import Path
import orjson, pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
def _src_dirs():
    env = os.environ.get("CYBERFM_EXPORT_DIRS")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p.strip()]
    return [ROOT / "data" / "export"]
SRC_DIRS = _src_dirs()
KEEP = ["time","label","class_uid","activity_id","process_pid","process_name","process_image","user_uid",
        "process_cmd_line","argv_argc","dst_ip","dst_port","status_code","file_path","event_class",
        "technique_id","sub_technique_id","severity","step_id","mimics_step_id"]

def bundles():
    seen = {}
    for d in SRC_DIRS:
        for p in sorted(d.glob("cflin-*-ocsf.tgz")):
            cid = p.name[:-len("-ocsf.tgz")]
            if cid not in seen: seen[cid] = p
    return seen

def main():
    rows_c, frames = [], []
    for cid, p in sorted(bundles().items()):
        try:
            with tarfile.open(p, "r:gz") as tf:
                man = ev = None
                for m in tf.getmembers():
                    if m.name.endswith("/manifest.json") and m.name.count("/") == 1: man = json.load(tf.extractfile(m))
                    elif m.name.endswith("/events.jsonl") and m.name.count("/") == 1: ev = tf.extractfile(m).read()
        except Exception as e:
            print("SKIP", cid, "unreadable", e); continue
        if man is None or ev is None:
            print("SKIP", cid, "no manifest/events"); continue
        recs = []
        for i, line in enumerate(ev.splitlines()):
            if not line.strip(): continue
            try: e = orjson.loads(line)
            except Exception: continue
            r = {k: e.get(k) for k in KEEP}
            r["keyset"] = hashlib.md5(",".join(sorted(k for k in e.keys() if k not in ("dst_ip","dst_port","status_code","file_path","file_name"))).encode()).hexdigest()[:8]
            r["idx"] = i
            recs.append(r)
        if not recs: print("SKIP", cid, "empty"); continue
        df = pd.DataFrame.from_records(recs)
        df.insert(0, "campaign_id", cid)
        for c in ["process_cmd_line","process_name","process_image","dst_ip","file_path","event_class","technique_id","sub_technique_id","severity","step_id","mimics_step_id","label"]:
            df[c] = df[c].astype("string")
        for c in ["class_uid","activity_id","process_pid","user_uid","argv_argc","dst_port","status_code"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        df["t_ns"] = pd.to_datetime(df["time"], utc=True, errors="coerce").astype("int64")
        df = df.drop(columns=["time"])
        frames.append(df)
        techs = man.get("techniques") or []
        rows_c.append(dict(
            campaign_id=cid, base_id=re.sub(r"-rerun\d+$", "", cid), family=man.get("campaign_family"),
            host=man.get("host_of_origin") or ("unknown:" + (man.get("hosts") or [{}])[0].get("hostname","?")),
            hostname=(man.get("hosts") or [{}])[0].get("hostname"), dilation=float(man.get("time_dilation") or 1.0),
            image_sha=(man.get("image_sha256") or "")[:12], collector_sha=(man.get("collector_config_sha256") or "")[:12],
            kernel=man.get("kernel_version"), benign_before_s=man.get("benign_before_s"), benign_after_s=man.get("benign_after_s"),
            attack_start=man.get("attack_start_utc"), attack_end=man.get("attack_end_utc"), completed_at=man.get("completed_at"),
            techniques="|".join(sorted({t.get("attack_id","?") for t in techs})), n_tech_steps=len(techs),
            window_integrity=man.get("window_integrity"), n_events=len(df),
            n_mal=int((df.label=="malicious").sum()), n_ctrl=int((df.label=="benign_control").sum()),
            src=str(p.parent.name)))
        print(cid, len(df), rows_c[-1]["n_mal"], flush=True)
    cat = pd.DataFrame(rows_c)
    cat.to_csv(ROOT/"data/campaigns.csv", index=False)
    ev = pd.concat(frames, ignore_index=True)
    ev.to_parquet(ROOT/"data/events.parquet", index=False)
    print("campaigns", len(cat), "events", len(ev), "mal", int((ev.label=="malicious").sum()))

if __name__ == "__main__":
    main()
