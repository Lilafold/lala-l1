"""Round-2 strict evaluation + fast GBDT baseline (CPU)."""
import json, numpy as np, pandas as pd, scipy.sparse as sp, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
 "BASE": ["class_uid", "activity_id", "uid_bucket"],
 "CONTENT": ["has_cmd", "argc", "cmd_len", "has_dst", "dst_port", "status_code", "has_file"],
 "TIME": ["log_dt_prev", "n_last_1s", "n_last_10s", "log_dt_prev_same_name"],
 "NOVELTY": ["new_name", "new_cmd", "new_image", "log_cnt_name_sofar", "log_cnt_cmd_sofar",
             "new_cmd_last32", "new_name_last32", "new_cmd_last256", "new_name_last256"],
}
PROBES = ["probe_rel_pos", "probe_pid", "probe_pid_jump", "probe_uid1001", "probe_n_events_campaign", "log_dt_next"]
_cache = {}
def load():
    if not _cache:
        F = pd.read_parquet(ROOT/"data/feat.parquet"); M = pd.read_parquet(ROOT/"data/meta.parquet")
        C = pd.read_csv(ROOT/"data/campaigns_enriched.csv").set_index("campaign_id")
        T = sp.load_npz(ROOT/"data/tok.npz").tocsr()
        M["y"] = (M.label == "malicious").astype(np.int8)
        M["family"] = M.campaign_id.map(C.family); M["host"] = M.campaign_id.map(C.host)
        M["dilated"] = M.campaign_id.map(C.dilation) != 1
        M["step"] = M.campaign_id.astype(str) + "|" + M.step_id.fillna(M.technique_id).fillna("?").astype(str)
        rows = {cid: np.asarray(ix) for cid, ix in M.groupby("campaign_id", sort=False).indices.items()}
        _cache.update(F=F, M=M, C=C, T=T, rows=rows, S=json.loads((ROOT/"results/splits_v1.json").read_text()))
    return _cache

def rows_for(cids):
    d = load(); return np.concatenate([d["rows"][c] for c in cids if c in d["rows"]])

def design(ix, cols, tok):
    d = load(); X = d["F"][cols].values[ix].astype(np.float32)
    if tok: return sp.hstack([sp.csr_matrix(X), d["T"][ix]], format="csr")
    return X

def fit(train_cids, cols, tok=True, benign_frac=0.05, seed=0, params=None, label_weights=None):
    d = load(); ix = rows_for(train_cids); M = d["M"]
    y = M.y.values[ix]; lab = M.label.values[ix]
    rng = np.random.RandomState(seed)
    keep = (y == 1) | (lab != "benign") | (rng.rand(len(ix)) < benign_frac)
    w = np.where((y == 0) & (lab == "benign"), 1.0 / benign_frac, 1.0)
    ix, y, w = ix[keep], y[keep], w[keep]
    if label_weights is not None: w = w * label_weights(ix)
    p = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=20, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=6, verbose=-1, seed=seed,
             scale_pos_weight=50.0)
    if params: p.update(params)
    ds = lgb.Dataset(design(ix, cols, tok), y, weight=w, free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=p.pop("rounds", 300) if "rounds" in p else 300)

def score(model, cids, cols, tok=True):
    ix = rows_for(cids)
    return ix, model.predict(design(ix, cols, tok), num_threads=6)

def thresholds(val_scores, val_y, fprs=(0.001, 0.01)):
    # tie-safe: alert iff score > thr, thr = (1-f) quantile ('higher'), so val FPR <= f even with ties
    neg = val_scores[val_y == 0]
    return {f: float(np.quantile(neg, 1 - f, method="higher")) for f in fprs}

def metrics(ix, s, thr, by=None):
    d = load(); M = d["M"]; y = M.y.values[ix]
    out = {"n": int(len(ix)), "n_pos": int(y.sum()), "n_campaigns": int(M.campaign_id.values[ix].nunique() if hasattr(M.campaign_id.values[ix], 'nunique') else len(set(M.campaign_id.values[ix])))}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, s)); out["pr_auc"] = float(average_precision_score(y, s))
        out["pr_auc_random"] = float(y.mean())
    steps = M.step.values[ix]; ctrl = M.label.values[ix] == "benign_control"
    for f, t in thr.items():
        fl = s > t; k = f"fpr{f:g}"
        out[k] = {"thr": t, "recall": float(fl[y == 1].mean()) if y.sum() else None, "test_fpr": float(fl[y == 0].mean()),
                  "fp": int((fl & (y == 0)).sum()), "tp": int((fl & (y == 1)).sum()),
                  "control_fpr": float(fl[ctrl].mean()) if ctrl.any() else None}
        st = pd.Series(fl[y == 1]).groupby(steps[y == 1]).max()
        out[k]["step_recall"] = float(st.mean()) if len(st) else None
        cm = pd.Series(fl[y == 1]).groupby(M.campaign_id.values[ix][y == 1]).max()
        out[k]["campaign_recall"] = float(cm.mean()) if len(cm) else None
    # oracle-on-test operating point (reported only for reference; never used for claims)
    if 0 < y.sum() < len(y):
        for f in (0.001, 0.01):
            t = np.quantile(s[y == 0], 1 - f, method="higher"); out[f"oracle_recall_fpr{f:g}"] = float((s[y == 1] > t).mean())
    if by:
        out["by_" + by] = {}
        keys = pd.Series(M[by].values[ix]).astype(object).fillna("none").astype(str).values
        for kval in pd.unique(keys):
            m = keys == kval
            if y[m].sum() == 0: continue
            sub = {"n_pos": int(y[m].sum())}
            for f, t in thr.items():
                sub[f"recall_fpr{f:g}"] = float((s[m][y[m] == 1] > t).mean())
                sub[f"fp_fpr{f:g}"] = int(((s[m] > t) & (y[m] == 0)).sum())
            if 0 < y[m].sum() < m.sum(): sub["pr_auc"] = float(average_precision_score(y[m], s[m]))
            out["by_" + by][str(kval)] = sub
    return out

def run(train_cids, val_cids, test_sets, cols, tok=True, by=None, **kw):
    m = fit(train_cids, cols, tok, **kw)
    vix, vs = score(m, val_cids, cols, tok); thr = thresholds(vs, load()["M"].y.values[vix])
    res = {"val": metrics(vix, vs, thr)}
    for name, cids in test_sets.items():
        ix, s = score(m, cids, cols, tok); res[name] = metrics(ix, s, thr, by=by)
    return m, res
