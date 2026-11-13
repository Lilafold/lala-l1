"""CyberFM-seq: small causal transformer over per-event embeddings (label-free inputs only)."""
import numpy as np, torch, torch.nn as nn
DENSE = ["class_uid", "activity_id", "uid_bucket", "has_cmd", "argc", "cmd_len", "has_dst", "dst_port", "status_code", "has_file",
         "new_name", "new_cmd", "new_image", "log_cnt_name_sofar", "log_cnt_cmd_sofar", "new_cmd_last32", "new_name_last32",
         "new_cmd_last256", "new_name_last256"]
NTOK, MAXTOK = 4096, 16
CLASS_VOCAB = {-1: 0, 1001: 1, 1007: 2, 3002: 3, 4001: 4}

class SeqDetector(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, ff=256, L=256, dropout=0.1, n_dense=len(DENSE)):
        super().__init__()
        self.tok = nn.EmbeddingBag(NTOK + 1, d, mode="mean", padding_idx=NTOK)
        self.dense = nn.Sequential(nn.Linear(n_dense, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Embedding(8, d); self.uid = nn.Embedding(4, d); self.act = nn.Embedding(32, d)
        self.pos = nn.Embedding(L, d); self.norm = nn.LayerNorm(d)
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.L = L
    def forward(self, tok, dense, cls, act, uid):
        B, T, K = tok.shape
        e = self.tok(tok.reshape(B * T, K)).reshape(B, T, -1)
        e = e + self.dense(dense) + self.cls(cls) + self.act(act) + self.uid(uid)
        e = self.norm(e + self.pos(torch.arange(T, device=tok.device))[None])
        mask = torch.triu(torch.full((T, T), float("-inf"), device=tok.device), 1)
        return self.head(self.tf(e, mask=mask, is_causal=True)).squeeze(-1)

def event_arrays(F, T_csr, rows, mean=None, std=None):
    """Build model inputs for given row indices (in order). F: feature DataFrame, T_csr: hashed tokens."""
    X = F[DENSE].values[rows].astype(np.float32)
    if mean is None: mean, std = X.mean(0), X.std(0) + 1e-6
    X = (X - mean) / std
    sub = T_csr[rows]; tok = np.full((len(rows), MAXTOK), NTOK, np.int16)
    for i in range(len(rows)):
        a = sub.indices[sub.indptr[i]:sub.indptr[i + 1]][:MAXTOK]; tok[i, :len(a)] = a
    cls = np.vectorize(lambda v: CLASS_VOCAB.get(int(v), 5))(F.class_uid.values[rows]).astype(np.int64)
    act = np.clip(F.activity_id.values[rows].astype(np.int64) + 1, 0, 31)
    uid = (F.uid_bucket.values[rows].astype(np.int64) + 1)
    return dict(tok=tok, dense=X, cls=cls, act=act, uid=uid), mean, std

@torch.no_grad()
def score_campaign(model, arr, device, L=256, stride=128, bs=64):
    """Causal streaming score for every event exactly once (unique-event scoring)."""
    n = len(arr["cls"]); out = np.zeros(n, np.float32)
    starts = list(range(0, max(1, n - L + 1), stride))
    if starts[-1] + L < n: starts.append(n - L)
    starts = [max(0, s) for s in starts]
    model.eval()
    for b in range(0, len(starts), bs):
        ss = starts[b:b + bs]; Lw = min(L, n)
        batch = {k: torch.as_tensor(np.stack([v[s:s + Lw] for s in ss])).to(device) for k, v in arr.items()}
        batch["tok"] = batch["tok"].long()
        p = torch.sigmoid(model(**batch).float()).cpu().numpy()
        for j, s in enumerate(ss):
            if s == 0: out[s:s + Lw] = p[j]
            else:
                keep_from = max(0, (starts[b + j - 1] + Lw if b + j > 0 else 0) - s)
                out[s + keep_from:s + Lw] = p[j][keep_from:]
    return out
