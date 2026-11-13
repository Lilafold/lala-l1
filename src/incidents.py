"""Incident grouping for Lala L1 scores (no retraining).

Turns per-event scores into incidents grouped by host + process tree + time.
Two emission paths:
  * immediate: any event with score > high_threshold
  * corroborated: moderate score plus >= 2 distinct behavior categories
    (not just repeated adjacent windows of the same behavior)

Thresholds must be tuned on train/dev only, never on frozen TEST.
"""
from __future__ import annotations

import hashlib
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

EPISODE_IDLE_S = 15 * 60
ATTACH_WINDOW_S = 30 * 60
PID_REUSE_GAP_S = 3600.0  # only treat as pid reuse after a long silent gap with no fork evidence
DEFAULT_BOOTSTRAP = 1000

# OCSF-style lineage fields (flat or dotted). Used when present.
GUID_FIELDS = (
    "process_uid", "process.uid", "process_guid", "process.guid",
    "process_unique_id", "process.unique_id", "actor.process.uid",
)
PARENT_PID_FIELDS = (
    "process_parent_pid", "parent_pid", "ppid", "process.parent_pid",
    "parent_process.pid", "process.parent.pid", "actor.process.pid",
    "parent_process_pid",
)
PARENT_GUID_FIELDS = (
    "parent_process_uid", "process.parent_uid", "parent_process.uid",
    "parent_guid", "process.parent.guid", "parent_process.guid",
    "actor.process.parent_uid",
)
PARENT_START_FIELDS = (
    "parent_process_start", "process.parent.start_time", "parent_process.start_time",
)
PROCESS_START_FIELDS = (
    "process_start_time", "process.start_time", "process_created_time",
)

# Observable behavior buckets from process/event fields only (never labels).
_NAME_CATS = {
    "inventory": {
        "ps", "id", "uname", "hostname", "hostnamectl", "lscpu", "lsblk", "ls",
        "find", "getent", "whoami", "who", "w", "users", "uptime", "free",
        "df", "du", "lshw", "lsusb", "lspci", "dmidecode", "timedatectl",
    },
    "network": {
        "ip", "ss", "ping", "dig", "nslookup", "host", "curl", "wget", "nc",
        "ncat", "nmap", "tracepath", "traceroute", "netstat", "arp", "iw",
        "resolvectl", "systemd-resolve",
    },
    "credentials": {
        "passwd", "sudo", "login", "last", "lastlog", "chage", "gpasswd",
        "useradd", "usermod", "userdel", "groupadd", "visudo", "su", "ssh",
        "ssh-keygen", "faillog", "loginctl",
    },
    "persistence": {
        "crontab", "cron", "systemctl", "systemd", "timedatectl",
    },
    "archive": {
        "tar", "gzip", "gunzip", "rsync", "cp", "scp", "zip", "unzip", "dd",
        "sha256sum", "sha1sum", "md5sum", "cpio", "install",
    },
    "packages": {
        "apt", "apt-get", "apt-cache", "dpkg", "dpkg-deb", "dpkg-query",
        "rpm", "snap", "pip", "pip3", "yum", "dnf",
    },
    "files": {
        "cat", "less", "more", "head", "tail", "vi", "vim", "nano", "ed",
        "chmod", "chown", "stat", "touch", "mkdir", "rm", "mv", "ln",
        "readlink", "getfacl", "setfacl", "journalctl",
    },
    "shell": {"bash", "sh", "dash", "zsh", "python", "python3", "perl", "awk", "sed"},
}
_CLASS_CATS = {1007: "process", 4001: "network_event", 3002: "file_event", 1001: "file_event"}


# ---------------------------------------------------------------------------
# Lineage / process trees
# ---------------------------------------------------------------------------

class _UF:
    def __init__(self):
        self.p = {}

    def add(self, x):
        self.p.setdefault(x, x)

    def find(self, x):
        self.add(x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
        return self.find(a)


def _get(rec, names):
    if rec is None:
        return None
    get = rec.get if hasattr(rec, "get") else lambda k, d=None: rec[k] if k in rec else d
    for n in names:
        if hasattr(rec, "get") and n in rec and rec[n] is not None and not (isinstance(rec[n], float) and math.isnan(rec[n])):
            v = rec[n]
            if v == "" or v is None:
                continue
            return v
        # dotted
        if "." in n and hasattr(rec, "get"):
            cur = rec
            ok = True
            for part in n.split("."):
                if not hasattr(cur, "get") or part not in cur or cur[part] is None:
                    ok = False
                    break
                cur = cur[part]
            if ok and cur not in ("", None):
                return cur
    return None


def _pid(v):
    if v is None:
        return None
    try:
        x = int(float(v))
        return x if x >= 0 else None
    except (TypeError, ValueError):
        return None


def build_identities(events: list[dict], forks: Optional[list[dict]] = None,
                     pid_reuse_gap_s: float = PID_REUSE_GAP_S) -> list[dict]:
    """Attach identity / parent_identity / tree_id to each event dict (in-place).

    Identity is process GUID if present, else ``pid:start_time`` (pid reuse
    after a gap or a new launch starts a new generation). Parent links come
    from OCSF parent fields when present, else from optional fork records
    ``{parent_pid, child_pid, t}`` in the same time domain as ``event['t']``.
    """
    evs = sorted(enumerate(events), key=lambda z: (z[1].get("t") is None, z[1].get("t") or 0, z[0]))
    # last sighting of each pid -> (identity, last_t, start_t). A pid keeps its
    # generation across quiet periods; a new generation starts only on a later
    # fork of that pid, a changed start_time field, or a very long silent gap.
    live = {}
    uf = _UF()
    ident_of = [None] * len(events)
    parent_of = [None] * len(events)
    last_fork_t = {}  # child_pid -> latest fork t

    # index forks by child_pid, time
    fork_by_child = defaultdict(list)
    if forks:
        for f in forks:
            cp, t = _pid(f.get("child_pid")), f.get("t")
            if cp is None or t is None:
                continue
            fork_by_child[cp].append((_pid(f.get("parent_pid")), float(t), f.get("parent_identity")))
        for k in fork_by_child:
            fork_by_child[k].sort(key=lambda x: x[1])
            last_fork_t[k] = fork_by_child[k][-1][1]

    def generation_break(pid, t, start_field):
        if pid not in live:
            return True
        ident, last_t, start_t = live[pid]
        if start_field is not None and str(start_field) != str(start_t):
            return True
        # a fork of this pid after the current generation started
        ft = last_fork_t.get(pid)
        # handled per-event below using forks at this t
        if t - last_t > pid_reuse_gap_s:
            return True
        return False

    for i, e in evs:
        t = float(e.get("t") or 0.0)
        pid = _pid(e.get("process_pid") if "process_pid" in e else e.get("pid"))
        guid = _get(e, GUID_FIELDS)
        start_field = _get(e, PROCESS_START_FIELDS)
        if guid is not None:
            ident = f"guid:{guid}"
        else:
            if pid is None:
                ident = f"anon:{e.get('campaign_id','?')}:{i}"
            else:
                # If a fork created this pid after the current generation started, new identity.
                new_gen = pid not in live
                if pid in live:
                    _, last_t, start_t = live[pid]
                    if start_field is not None and str(start_field) != str(start_t):
                        new_gen = True
                    elif t - last_t > pid_reuse_gap_s:
                        new_gen = True
                    else:
                        cands = [f for f in fork_by_child.get(pid, []) if live[pid][2] < f[1] <= t + 1e-3]
                        if cands:
                            new_gen = True
                if not new_gen:
                    ident, _, start_t = live[pid]
                else:
                    # prefer fork timestamp as start if available
                    start_t = start_field if start_field is not None else t
                    cands = [f for f in fork_by_child.get(pid, []) if f[1] <= t + 1e-3]
                    if cands and start_field is None:
                        start_t = cands[-1][1]
                    ident = f"pid:{pid}:{start_t}"
        if pid is not None:
            start_t = live[pid][2] if pid in live and ident == live[pid][0] else (start_field if start_field is not None else t)
            # keep start of this identity
            if pid in live and ident == live[pid][0]:
                start_t = live[pid][2]
            else:
                start_t = start_field if start_field is not None else t
                cands = [f for f in fork_by_child.get(pid, []) if f[1] <= t + 1e-3] if pid is not None else []
                if cands and start_field is None and ident.endswith(str(cands[-1][1])):
                    start_t = cands[-1][1]
            live[pid] = (ident, t, start_t)
        uf.add(ident)

        parent_ident = None
        pguid = _get(e, PARENT_GUID_FIELDS)
        ppid = _pid(_get(e, PARENT_PID_FIELDS))
        pstart = _get(e, PARENT_START_FIELDS)
        if pguid is not None:
            parent_ident = f"guid:{pguid}"
        elif ppid is not None:
            if pstart is not None:
                parent_ident = f"pid:{ppid}:{pstart}"
            elif ppid in live:
                parent_ident = live[ppid][0]
        if parent_ident is None and pid is not None and fork_by_child.get(pid):
            cands = [f for f in fork_by_child[pid] if f[1] <= t + 1e-3]
            if cands:
                fpp, ft, fpi = cands[-1]
                if fpi:
                    parent_ident = fpi
                elif fpp is not None:
                    if fpp in live:
                        parent_ident = live[fpp][0]
                    else:
                        parent_ident = f"pid:{fpp}:{ft}"
        ident_of[i] = ident
        parent_of[i] = parent_ident
        if parent_ident:
            uf.union(ident, parent_ident)

    # second pass: if later events reveal parents for an identity, trees already unioned
    out = []
    for i, e in enumerate(events):
        ident = ident_of[i]
        e = dict(e)
        e["identity"] = ident
        e["parent_identity"] = parent_of[i]
        e["tree_id"] = uf.find(ident) if ident is not None else f"anon:{i}"
        out.append(e)
    return out


def extract_forks_from_ebpf_raw(raw_text: str, exec_events: list[dict]) -> list[dict]:
    """Map eBPF monotonic nsecs to event wall times via exec (pid, cmd) matches."""
    forks_ns = []  # (ns, ppid, cpid)
    execs_ns = []  # (ns, pid, cmd)
    for line in raw_text.splitlines():
        if not line:
            continue
        if line.startswith("F|"):
            parts = line.split("|")
            if len(parts) >= 4:
                try:
                    forks_ns.append((int(parts[1]), int(parts[2]), int(parts[3])))
                except ValueError:
                    pass
        elif line.startswith("A|"):
            parts = line.split("|", 5)
            if len(parts) >= 5:
                try:
                    cmd = parts[5].strip() if len(parts) > 5 else ""
                    execs_ns.append((int(parts[1]), int(parts[2]), cmd))
                except ValueError:
                    pass
        elif line.startswith("E|"):
            parts = line.split("|")
            if len(parts) >= 5:
                try:
                    cmd = parts[4]
                    execs_ns.append((int(parts[1]), int(parts[2]), cmd))
                except ValueError:
                    pass
    # pair execs with process events by (pid, normalized cmd) in time order
    ev_exec = []
    for e in exec_events:
        pid = _pid(e.get("process_pid") if "process_pid" in e else e.get("pid"))
        if pid is None or e.get("t") is None:
            continue
        cmd = str(e.get("process_cmd_line") or e.get("process_image") or e.get("process_name") or "")
        ev_exec.append((float(e["t"]), pid, cmd, e))
    ev_exec.sort(key=lambda x: x[0])
    execs_ns.sort(key=lambda x: x[0])
    used = set()
    pairs = []  # (ns, t_wall)
    ev_by_pid = defaultdict(list)
    for item in ev_exec:
        ev_by_pid[item[1]].append(item)
    ns_by_pid = defaultdict(list)
    for item in execs_ns:
        ns_by_pid[item[1]].append(item)
    for pid, evs in ev_by_pid.items():
        nslist = ns_by_pid.get(pid, [])
        j = 0
        for ev in evs:
            # greedy: next unused ns exec with same pid; prefer cmd prefix match
            best = None
            for k in range(j, len(nslist)):
                if (pid, k) in used:
                    continue
                ns, _, cmd = nslist[k]
                if ev[2] and cmd and not (cmd.startswith(ev[2][:40]) or ev[2].startswith(cmd[:40])):
                    # allow later match
                    continue
                best = k
                break
            if best is None:
                # fallback next unused
                for k in range(j, len(nslist)):
                    if (pid, k) not in used:
                        best = k
                        break
            if best is None:
                continue
            used.add((pid, best))
            pairs.append((nslist[best][0], ev[0]))
            j = best + 1
    if len(pairs) < 2:
        return []
    pairs.sort()
    xs = np.array([p[0] for p in pairs], dtype=np.float64)
    ys = np.array([p[1] for p in pairs], dtype=np.float64)

    def ns_to_t(ns):
        return float(np.interp(ns, xs, ys))

    out = []
    for ns, ppid, cpid in forks_ns:
        out.append({"parent_pid": ppid, "child_pid": cpid, "t": ns_to_t(ns)})
    return out


def load_forks_from_raw_tgz(path: Path, exec_events: list[dict]) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        with tarfile.open(path, "r:gz") as tf:
            member = None
            for name in ("pilot/ebpf.raw", "ebpf.raw"):
                try:
                    member = tf.getmember(name)
                    break
                except KeyError:
                    continue
            if member is None:
                for m in tf.getmembers():
                    if m.name.endswith("ebpf.raw"):
                        member = m
                        break
            if member is None:
                return []
            raw = tf.extractfile(member).read().decode("utf-8", "replace")
        return extract_forks_from_ebpf_raw(raw, exec_events)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Behavior categories
# ---------------------------------------------------------------------------

def behavior_category(event: dict) -> str:
    def _s(v):
        if v is None:
            return ""
        s = str(v).strip()
        return "" if s.lower() in ("nan", "none", "<na>") else s
    name = _s(event.get("process_name"))
    image = _s(event.get("process_image"))
    cmd = _s(event.get("process_cmd_line"))
    raw = name or Path(image).name or ""
    base = (raw.split()[0].lower() if raw.split() else "")
    base = Path(base).name if base else ""
    for cat, names in _NAME_CATS.items():
        if base in names:
            return cat
    # argv0 from cmd
    if cmd:
        argv0 = Path(cmd.strip().split()[0]).name.lower()
        for cat, names in _NAME_CATS.items():
            if argv0 in names:
                return cat
        low = cmd.lower()
        if "systemctl" in low or "/etc/cron" in low or "crontab" in low:
            return "persistence"
        if any(tok in low for tok in ("/etc/shadow", "sudo -l", "getent passwd")):
            return "credentials"
        if any(tok in low for tok in ("ss -", "ip route", "ip a", "curl ")):
            return "network"
    try:
        cu = int(float(event.get("class_uid"))) if event.get("class_uid") is not None else None
    except (TypeError, ValueError):
        cu = None
    if cu in _CLASS_CATS:
        return _CLASS_CATS[cu]
    if event.get("event_class"):
        return str(event["event_class"])
    return "other"


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_alerts(events: list[dict], *, high_thr: float, moderate_thr: float,
                 attach_s: float = ATTACH_WINDOW_S,
                 min_categories: int = 2) -> list[dict]:
    """Group candidate alerts (score > moderate) into incidents.

    An incident is emitted if it has a high-confidence event (immediate path)
    or at least ``min_categories`` distinct behavior categories (corroborated).
    Events are attached to an open incident on the same host + process tree
    when the gap since that incident's last event is <= attach_s (default 30 min).
    """
    cands = [e for e in events if e.get("score") is not None and float(e["score"]) > moderate_thr]
    cands.sort(key=lambda e: (str(e.get("host")), e.get("t") or 0, e.get("idx") or 0))
    open_inc = {}  # (host, tree_id) -> incident dict
    closed = []
    next_id = 1

    def close(key):
        inc = open_inc.pop(key, None)
        if inc:
            closed.append(_finalize(inc, high_thr, min_categories))

    for e in cands:
        host = str(e.get("host") or e.get("campaign_id") or "")
        tree = str(e.get("tree_id") or e.get("identity") or "")
        key = (host, tree)
        t = float(e.get("t") or 0)
        inc = open_inc.get(key)
        if inc is not None and t - inc["t_last"] <= attach_s:
            inc["events"].append(e)
            inc["t_last"] = t
            inc["max_score"] = max(inc["max_score"], float(e["score"]))
        else:
            if inc is not None:
                close(key)
            open_inc[key] = {
                "incident_id": f"inc-{next_id}",
                "host": host,
                "tree_id": tree,
                "campaign_id": e.get("campaign_id"),
                "t_first": t,
                "t_last": t,
                "max_score": float(e["score"]),
                "events": [e],
            }
            next_id += 1
    for key in list(open_inc):
        close(key)
    return [inc for inc in closed if inc["emitted"]]


def _finalize(inc, high_thr, min_categories):
    evs = inc["events"]
    cats = []
    seen = set()
    for e in evs:
        c = e.get("category") or behavior_category(e)
        if c not in seen:
            seen.add(c)
            cats.append(c)
    high = [e for e in evs if float(e["score"]) > high_thr]
    immediate = bool(high)
    corroborated = (len(seen) >= min_categories)
    path = "both" if immediate and corroborated else ("immediate" if immediate else ("corroborated" if corroborated else "none"))
    inc = dict(inc)
    inc["categories"] = cats
    inc["n_events"] = len(evs)
    inc["n_categories"] = len(cats)
    inc["path"] = path
    inc["emitted"] = immediate or corroborated
    inc["n_malicious"] = int(sum(1 for e in evs if e.get("y") == 1 or e.get("label") == "malicious"))
    inc["true_positive"] = inc["n_malicious"] > 0
    # drop bulky event bodies from serialized incidents; keep compact refs
    inc["event_idx"] = [e.get("idx") for e in evs]
    inc["event_times"] = [e.get("t") for e in evs]
    inc["event_scores"] = [float(e["score"]) for e in evs]
    inc["event_labels"] = [e.get("label") for e in evs]
    inc["event_categories"] = [e.get("category") or behavior_category(e) for e in evs]
    inc.pop("events", None)
    return inc


# ---------------------------------------------------------------------------
# Episodes (ground-truth malicious) and metrics
# ---------------------------------------------------------------------------

def malicious_episodes(events: list[dict], idle_s: float = EPISODE_IDLE_S) -> list[dict]:
    mals = [e for e in events if e.get("y") == 1 or e.get("label") == "malicious"]
    mals.sort(key=lambda e: (str(e.get("host")), str(e.get("tree_id")), e.get("t") or 0))
    eps = []
    cur = None
    for e in mals:
        host, tree = str(e.get("host") or ""), str(e.get("tree_id") or "")
        t = float(e.get("t") or 0)
        if cur and cur["host"] == host and cur["tree_id"] == tree and t - cur["t_last"] <= idle_s:
            cur["t_last"] = t
            cur["n"] += 1
            cur["idx"].append(e.get("idx"))
        else:
            if cur:
                eps.append(cur)
            cur = {"host": host, "tree_id": tree, "campaign_id": e.get("campaign_id"),
                   "t_first": t, "t_last": t, "n": 1, "idx": [e.get("idx")]}
    if cur:
        eps.append(cur)
    return eps


def _overlap(a0, a1, b0, b1):
    return not (a1 < b0 or b1 < a0)


def campaign_metrics(events: list[dict], incidents: list[dict], *,
                     attack_start: Optional[float], attack_end: Optional[float],
                     benign_seconds: Optional[float] = None,
                     raw_thr: float = None) -> dict:
    evs = events
    incs = incidents
    y = [1 if (e.get("y") == 1 or e.get("label") == "malicious") else 0 for e in evs]
    n_pos = int(sum(y))
    times = [float(e.get("t") or 0) for e in evs]
    t0, t1 = (min(times), max(times)) if times else (0.0, 0.0)
    if benign_seconds is None:
        if n_pos == 0:
            benign_seconds = max(0.0, t1 - t0)
        else:
            # wall span minus attack window if known, else span of non-malicious events
            if attack_start is not None and attack_end is not None and attack_end > attack_start:
                total = max(0.0, t1 - t0)
                atk = max(0.0, min(t1, attack_end) - max(t0, attack_start))
                benign_seconds = max(0.0, total - atk)
            else:
                bt = [t for t, yy in zip(times, y) if yy == 0]
                benign_seconds = (max(bt) - min(bt)) if bt else 0.0
    host_days = benign_seconds / 86400.0
    host_hours = benign_seconds / 3600.0

    fps = [i for i in incs if not i.get("true_positive")]
    tps = [i for i in incs if i.get("true_positive")]
    detected = bool(tps)
    ttd = None
    if detected and attack_start is not None:
        first = min(i["t_first"] for i in tps)
        ttd = max(0.0, first - float(attack_start))

    eps = malicious_episodes(evs)
    missed = 0
    for ep in eps:
        hit = False
        for inc in tps:
            if inc.get("host") == ep["host"] and inc.get("tree_id") == ep["tree_id"] and _overlap(
                inc["t_first"], inc["t_last"], ep["t_first"], ep["t_last"]
            ):
                hit = True
                break
            # also count if any shared event idx
            if set(inc.get("event_idx") or []) & set(ep.get("idx") or []):
                hit = True
                break
        if not hit:
            missed += 1

    raw = {}
    if raw_thr is not None:
        raw_alerts = [e for e in evs if e.get("score") is not None and float(e["score"]) > raw_thr and not (e.get("y") == 1 or e.get("label") == "malicious")]
        raw_tp = [e for e in evs if e.get("score") is not None and float(e["score"]) > raw_thr and (e.get("y") == 1 or e.get("label") == "malicious")]
        raw_det = bool(raw_tp)
        raw_ttd = None
        if raw_det and attack_start is not None:
            first = min(float(e["t"]) for e in raw_tp)
            raw_ttd = max(0.0, first - float(attack_start))
        raw = {
            "n_alerts": len(raw_alerts) + len(raw_tp),
            "n_fp_alerts": len(raw_alerts),
            "alerts_per_host_hour": (len(raw_alerts) / host_hours) if host_hours > 0 else None,
            "alerts_per_host_day": (len(raw_alerts) / host_days) if host_days > 0 else None,
            "detected": raw_det,
            "ttd_s": raw_ttd,
        }

    return {
        "campaign_id": evs[0].get("campaign_id") if evs else None,
        "n_events": len(evs),
        "n_pos": n_pos,
        "benign_seconds": benign_seconds,
        "host_days": host_days,
        "n_incidents": len(incs),
        "n_fp_incidents": len(fps),
        "n_tp_incidents": len(tps),
        "false_incidents_per_host_day": (len(fps) / host_days) if host_days > 0 else (float(len(fps)) if len(fps) else 0.0),
        "detected": detected,
        "ttd_s": ttd,
        "n_malicious_episodes": len(eps),
        "n_missed_episodes": missed,
        "raw": raw,
    }


def _pct(xs, q):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def aggregate(per_campaign: list[dict], *, n_bootstrap: int = DEFAULT_BOOTSTRAP, seed: int = 0) -> dict:
    attack = [c for c in per_campaign if c.get("n_pos", 0) > 0]
    benign = [c for c in per_campaign if c.get("n_pos", 0) == 0]
    # include attack campaigns' benign periods in host-day denominator
    def _fp_rate(cs):
        fp = sum(c["n_fp_incidents"] for c in cs)
        days = sum(c["host_days"] for c in cs)
        return (fp / days) if days > 0 else 0.0

    def _det(cs):
        atk = [c for c in cs if c.get("n_pos", 0) > 0]
        return (sum(1 for c in atk if c["detected"]) / len(atk)) if atk else None

    def _ttds(cs):
        return [c["ttd_s"] for c in cs if c.get("n_pos", 0) > 0 and c.get("detected") and c.get("ttd_s") is not None]

    def _missed(cs):
        atk = [c for c in cs if c.get("n_pos", 0) > 0]
        miss = sum(c["n_missed_episodes"] for c in atk)
        tot = sum(c["n_malicious_episodes"] for c in atk)
        return {"n_missed": miss, "n_episodes": tot, "miss_rate": (miss / tot) if tot else 0.0}

    def _raw_rate(cs, key):
        vals = []
        w = []
        for c in cs:
            r = c.get("raw") or {}
            if r.get(key) is None:
                continue
            vals.append(r[key] * (c["host_days"] if "day" in key else c["benign_seconds"] / 3600.0))
            w.append(c["host_days"] if "day" in key else c["benign_seconds"] / 3600.0)
        s = sum(w)
        # recompute from counts if possible
        return None

    fp_alerts = sum((c.get("raw") or {}).get("n_fp_alerts") or 0 for c in per_campaign)
    days = sum(c["host_days"] for c in per_campaign)
    hours = sum(c["benign_seconds"] / 3600.0 for c in per_campaign)
    raw_det = ([c for c in attack if (c.get("raw") or {}).get("detected")] )
    raw_ttd = [(c.get("raw") or {}).get("ttd_s") for c in attack if (c.get("raw") or {}).get("detected")]

    point = {
        "n_campaigns": len(per_campaign),
        "n_attack_campaigns": len(attack),
        "n_benign_only_campaigns": len(benign),
        "host_days": days,
        "false_incidents_per_host_day": _fp_rate(per_campaign),
        "n_fp_incidents": sum(c["n_fp_incidents"] for c in per_campaign),
        "n_tp_incidents": sum(c["n_tp_incidents"] for c in per_campaign),
        "campaign_detection_rate": _det(per_campaign),
        "time_to_first_incident_s": {
            "median": _pct(_ttds(per_campaign), 50),
            "p95": _pct(_ttds(per_campaign), 95),
            "n": len(_ttds(per_campaign)),
        },
        "missed_malicious_episodes": _missed(per_campaign),
        "raw_threshold": {
            "fp_alerts_per_host_hour": (fp_alerts / hours) if hours > 0 else None,
            "fp_alerts_per_host_day": (fp_alerts / days) if days > 0 else None,
            "n_fp_alerts": fp_alerts,
            "campaign_detection_rate": (len(raw_det) / len(attack)) if attack else None,
            "time_to_first_alert_s": {
                "median": _pct(raw_ttd, 50),
                "p95": _pct(raw_ttd, 95),
                "n": len([x for x in raw_ttd if x is not None]),
            },
        },
    }

    rng = np.random.RandomState(seed)
    n = len(per_campaign)
    boots = {"false_incidents_per_host_day": [], "campaign_detection_rate": [],
             "ttd_median": [], "ttd_p95": [], "miss_rate": [],
             "raw_fp_per_host_hour": [], "raw_fp_per_host_day": [], "raw_campaign_detection_rate": []}
    if n and n_bootstrap:
        for _ in range(n_bootstrap):
            ix = rng.randint(0, n, size=n)
            sample = [per_campaign[i] for i in ix]
            boots["false_incidents_per_host_day"].append(_fp_rate(sample))
            boots["campaign_detection_rate"].append(_det(sample))
            tt = _ttds(sample)
            boots["ttd_median"].append(_pct(tt, 50))
            boots["ttd_p95"].append(_pct(tt, 95))
            boots["miss_rate"].append(_missed(sample)["miss_rate"])
            fp_a = sum((c.get("raw") or {}).get("n_fp_alerts") or 0 for c in sample)
            d = sum(c["host_days"] for c in sample)
            h = sum(c["benign_seconds"] / 3600.0 for c in sample)
            boots["raw_fp_per_host_hour"].append((fp_a / h) if h > 0 else None)
            boots["raw_fp_per_host_day"].append((fp_a / d) if d > 0 else None)
            atk = [c for c in sample if c.get("n_pos", 0) > 0]
            boots["raw_campaign_detection_rate"].append(
                (sum(1 for c in atk if (c.get("raw") or {}).get("detected")) / len(atk)) if atk else None
            )

    def ci(arr):
        xs = [x for x in arr if x is not None]
        if not xs:
            return {"lo": None, "hi": None, "n": 0}
        return {"lo": float(np.percentile(xs, 2.5)), "hi": float(np.percentile(xs, 97.5)), "n": len(xs)}

    point["bootstrap"] = {
        "n": n_bootstrap,
        "unit": "campaign",
        "false_incidents_per_host_day": ci(boots["false_incidents_per_host_day"]),
        "campaign_detection_rate": ci(boots["campaign_detection_rate"]),
        "time_to_first_incident_s_median": ci(boots["ttd_median"]),
        "time_to_first_incident_s_p95": ci(boots["ttd_p95"]),
        "missed_episode_rate": ci(boots["miss_rate"]),
        "raw_fp_alerts_per_host_hour": ci(boots["raw_fp_per_host_hour"]),
        "raw_fp_alerts_per_host_day": ci(boots["raw_fp_per_host_day"]),
        "raw_campaign_detection_rate": ci(boots["raw_campaign_detection_rate"]),
    }
    return point


def score_incidents(events_by_campaign: dict, *, high_thr: float, moderate_thr: float,
                    attach_s: float = ATTACH_WINDOW_S, min_categories: int = 2,
                    raw_thr: Optional[float] = None, n_bootstrap: int = DEFAULT_BOOTSTRAP,
                    meta_by_campaign: Optional[dict] = None) -> dict:
    """events_by_campaign: cid -> list of event dicts with t, score, host, lineage, label/y."""
    meta_by_campaign = meta_by_campaign or {}
    per = []
    incidents = {}
    for cid, evs in events_by_campaign.items():
        if not evs or evs[0].get("tree_id") is None:
            evs = build_identities(evs, forks=meta_by_campaign.get(cid, {}).get("forks"))
        for e in evs:
            if not e.get("category"):
                e["category"] = behavior_category(e)
            if "host" not in e or e["host"] is None:
                e["host"] = meta_by_campaign.get(cid, {}).get("host") or cid
        incs = group_alerts(evs, high_thr=high_thr, moderate_thr=moderate_thr,
                            attach_s=attach_s, min_categories=min_categories)
        md = meta_by_campaign.get(cid, {})
        per.append(campaign_metrics(
            evs, incs,
            attack_start=md.get("attack_start"),
            attack_end=md.get("attack_end"),
            benign_seconds=md.get("benign_seconds"),
            raw_thr=raw_thr if raw_thr is not None else moderate_thr,
        ))
        incidents[cid] = incs
    agg = aggregate(per, n_bootstrap=n_bootstrap)
    agg["per_campaign"] = per
    agg["thresholds"] = {"high": high_thr, "moderate": moderate_thr, "attach_s": attach_s,
                         "min_categories": min_categories, "raw_thr": raw_thr}
    return {"aggregate": agg, "incidents": incidents, "per_campaign": per}


# ---------------------------------------------------------------------------
# Threshold tuning (train/dev only)
# ---------------------------------------------------------------------------

def _unit_folds(cids, units: dict, k=3):
    buckets = defaultdict(list)
    for c in cids:
        buckets[units.get(c, c)].append(c)
    order = sorted(buckets)
    folds = [[] for _ in range(min(k, max(1, len(order))))]
    for i, u in enumerate(order):
        folds[i % len(folds)].extend(buckets[u])
    return [f for f in folds if f]


def tune_thresholds(events_by_campaign: dict, *, train_cids: list[str],
                    units: Optional[dict] = None,
                    high_grid: Optional[list] = None,
                    moderate_grid: Optional[list] = None,
                    min_detection: float = 0.9,
                    raw_thr: Optional[float] = None,
                    attach_s: float = ATTACH_WINDOW_S,
                    min_categories: int = 2,
                    forbidden_cids: Optional[set] = None,
                    n_bootstrap: int = 0,
                    meta_by_campaign: Optional[dict] = None) -> dict:
    """Grid-search thresholds with unit-grouped CV on train_cids only.

    ``forbidden_cids`` (e.g. frozen TEST) is asserted unused.
    """
    forbidden_cids = set(forbidden_cids or [])
    overlap = set(train_cids) & forbidden_cids
    if overlap:
        raise ValueError(f"tune_thresholds saw forbidden TEST campaigns: {sorted(overlap)[:8]}")
    units = units or {c: c for c in train_cids}
    evs = {c: events_by_campaign[c] for c in train_cids if c in events_by_campaign}
    scores = np.concatenate([np.asarray([e["score"] for e in evs[c]], dtype=np.float64) for c in evs] or [np.array([0.0])])
    if high_grid is None:
        high_grid = sorted(set(float(x) for x in
                               list(np.quantile(scores, [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]))
                               + [0.15, 0.20, 0.25, 0.35, 0.5, 0.7]))
    if moderate_grid is None:
        moderate_grid = sorted(set(float(x) for x in
                                   list(np.quantile(scores, [0.80, 0.90, 0.95, 0.98]))
                                   + [0.01, 0.02, 0.05, 0.08, 0.12, 0.18, 0.20]))
    folds = _unit_folds(list(evs), units, k=3)
    results = []
    for high in high_grid:
        for mod in moderate_grid:
            if not (0 <= mod < high):
                continue
            fold_fp, fold_det, fold_ttd = [], [], []
            for held in folds:
                tr_hold = {c: evs[c] for c in held if c in evs}
                if not tr_hold:
                    continue
                out = score_incidents(tr_hold, high_thr=high, moderate_thr=mod, attach_s=attach_s,
                                      min_categories=min_categories, raw_thr=raw_thr or high,
                                      n_bootstrap=0, meta_by_campaign=meta_by_campaign)
                a = out["aggregate"]
                fold_fp.append(a["false_incidents_per_host_day"])
                if a["campaign_detection_rate"] is not None:
                    fold_det.append(a["campaign_detection_rate"])
                if a["time_to_first_incident_s"]["median"] is not None:
                    fold_ttd.append(a["time_to_first_incident_s"]["median"])
            if not fold_fp:
                continue
            det = float(np.mean(fold_det)) if fold_det else 0.0
            rec = {
                "high": float(high), "moderate": float(mod),
                "cv_false_incidents_per_host_day": float(np.mean(fold_fp)),
                "cv_campaign_detection_rate": det,
                "cv_ttd_median_s": float(np.mean(fold_ttd)) if fold_ttd else None,
                "meets_min_detection": det >= min_detection,
            }
            results.append(rec)
    # select: among those meeting min detection, minimize FP/host-day; tie-break lower TTD then higher det
    ok = [r for r in results if r["meets_min_detection"]]
    pool = ok if ok else results
    def key(r):
        ttd = r["cv_ttd_median_s"] if r["cv_ttd_median_s"] is not None else 1e18
        return (r["cv_false_incidents_per_host_day"], ttd, -r["cv_campaign_detection_rate"])
    pool.sort(key=key)
    best = pool[0] if pool else {"high": 0.5, "moderate": 0.2, "note": "fallback, empty grid"}
    # refit summary on all train cids (still not TEST)
    full = score_incidents(evs, high_thr=best["high"], moderate_thr=best["moderate"],
                           attach_s=attach_s, min_categories=min_categories,
                           raw_thr=raw_thr or best["high"], n_bootstrap=n_bootstrap,
                           meta_by_campaign=meta_by_campaign)
    return {
        "selected": best,
        "grid": results,
        "n_grid": len(results),
        "min_detection": min_detection,
        "train_aggregate": full["aggregate"],
        "forbidden_cids_checked": sorted(forbidden_cids),
    }


def lookup_raw_tgz(cid: str, export_roots: Iterable[Path]) -> Optional[Path]:
    for root in export_roots:
        for name in (f"{cid}-raw.tgz", f"{cid}_raw.tgz"):
            p = Path(root) / name
            if p.exists():
                return p
    return None
