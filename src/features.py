#!/usr/bin/env python3
"""Leak-free event features for round-2 baselines. Label-derived keys (label, step_id, technique_id,
severity, mimics_step_id, schema keyset) are NEVER used as features. Output: data/feat.parquet
(dense) + data/tok.npz (sparse hashed tokens), row-aligned with sorted events."""
import re, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import HashingVectorizer
ROOT = Path(__file__).resolve().parents[1]
ev = pd.read_parquet(ROOT/"data/events.parquet")
ev = ev.sort_values(["campaign_id", "t_ns", "idx"], kind="stable").reset_index(drop=True)  # t_ns is actually microseconds (pandas3)
t = ev.t_ns.values.astype(np.int64)
g = ev.groupby("campaign_id", sort=False)
F = pd.DataFrame(index=ev.index)
# ---- BASE (schema-level, what Antigravity's model saw minus the leak)
F["class_uid"] = ev.class_uid.fillna(-1).astype(np.int16)
F["activity_id"] = ev.activity_id.fillna(-1).astype(np.int16)
u = ev.user_uid
F["uid_bucket"] = np.select([u.isna(), u == 0, u < 1000], [-1, 0, 1], 2).astype(np.int8)
# ---- ARGS / NET / FILE (content)
cmd = ev.process_cmd_line.fillna("")
F["has_cmd"] = (cmd != "").astype(np.int8)
F["argc"] = ev.argv_argc.fillna(-1).astype(np.float32)
F["cmd_len"] = cmd.str.len().astype(np.float32)
F["has_dst"] = ev.dst_ip.notna().astype(np.int8)
F["dst_port"] = ev.dst_port.fillna(-1).astype(np.float32)
F["status_code"] = ev.status_code.fillna(-999).astype(np.float32)
F["has_file"] = ev.file_path.notna().astype(np.int8)
# ---- TIME (raw wall-clock; distorted by 60x dilation)
dt_prev = np.diff(t, prepend=t[0]).astype(np.float64); first = (g.cumcount() == 0).values; dt_prev[first] = 1e9
F["log_dt_prev"] = np.log10(np.clip(dt_prev, 1, None)).astype(np.float32)
dt_next = np.r_[np.diff(t), 1e9].astype(np.float64); last = (g.cumcount(ascending=False) == 0).values; dt_next[last] = 1e9
F["log_dt_next"] = np.log10(np.clip(dt_next, 1, None)).astype(np.float32)
n1 = np.zeros(len(ev), np.float32); n10 = np.zeros(len(ev), np.float32)
for _, ix in g.indices.items():
    tt = t[ix]
    n1[ix] = np.arange(len(ix)) - np.searchsorted(tt, tt - 1_000_000, "left")
    n10[ix] = np.arange(len(ix)) - np.searchsorted(tt, tt - 10_000_000, "left")
F["n_last_1s"] = n1; F["n_last_10s"] = n10
name = ev.process_name.fillna("<none>")
F["log_dt_prev_same_name"] = np.log10(np.clip(ev.assign(n=name).groupby(["campaign_id", "n"], sort=False).t_ns.diff().fillna(1e12).values.astype(float), 1, None)).astype(np.float32)
# ---- NOVELTY (online, per campaign, label-free): has this name / exe / cmdline been seen before on this host-run?
key_c = ev.campaign_id.astype(str)
def first_seen(s):
    return (~pd.Series(list(zip(key_c, s))).duplicated()).values.astype(np.int8)
F["new_name"] = first_seen(name)
F["new_cmd"] = first_seen(cmd)
F["new_image"] = first_seen(ev.process_image.fillna(""))
F["log_cnt_name_sofar"] = np.log1p(ev.assign(n=name).groupby(["campaign_id", "n"], sort=False).cumcount().values).astype(np.float32)
F["log_cnt_cmd_sofar"] = np.log1p(ev.assign(n=cmd).groupby(["campaign_id", "n"], sort=False).cumcount().values).astype(np.float32)
for w in (32, 256):
    F[f"new_cmd_last{w}"] = pd.Series(F.new_cmd.values).groupby(ev.campaign_id.values).transform(lambda s: s.rolling(w, min_periods=1).sum()).values.astype(np.float32)
    F[f"new_name_last{w}"] = pd.Series(F.new_name.values).groupby(ev.campaign_id.values).transform(lambda s: s.rolling(w, min_periods=1).sum()).values.astype(np.float32)
# ---- PROBES (known spurious / positional cues; NOT used by default, only for shortcut analysis)
pos = g.cumcount().values; n = g.idx.transform("size").values
F["probe_rel_pos"] = (pos / n).astype(np.float32)
F["probe_pid"] = ev.process_pid.fillna(-1).astype(np.float32)
F["probe_pid_jump"] = (ev.process_pid.fillna(0).values - pd.Series(ev.process_pid.fillna(0).values).groupby(ev.campaign_id.values).transform(lambda s: s.rolling(32, min_periods=1).max().shift(1)).fillna(0).values).astype(np.float32)
F["probe_uid1001"] = (u == 1001).astype(np.int8)
F["probe_n_events_campaign"] = n.astype(np.float32)
meta = ev[["campaign_id", "label", "step_id", "technique_id", "mimics_step_id", "idx"]].copy()
meta.to_parquet(ROOT/"data/meta.parquet", index=False)
F.to_parquet(ROOT/"data/feat.parquet", index=False)
# ---- TOKENS: process name, image path pieces, cmdline tokens (hashed bag)
img = ev.process_image.fillna("")
def tok(i):
    return None
docs = ("n:" + name + " i:" + img + " " + cmd.str.replace(r"[\s/=,;:'\"()\[\]{}|&<>]+", " ", regex=True).str.slice(0, 400))
hv = HashingVectorizer(n_features=2**12, token_pattern=r"[^ ]+", lowercase=True, alternate_sign=False, norm=None, binary=True, dtype=np.float32)
X = hv.transform(docs.values)
sp.save_npz(ROOT/"data/tok.npz", X.tocsr())
print("done", F.shape, X.shape, X.nnz)
