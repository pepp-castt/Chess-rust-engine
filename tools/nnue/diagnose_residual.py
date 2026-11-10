#!/usr/bin/env python3
"""Why does a residual net with a big held-out-loss gain still lose games? Numbers on the held-out split.
    python tools/nnue/diagnose_residual.py --data data/x.bin --net resid.bin
"""
import argparse, importlib.util, os, struct
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("train_b", os.path.join(HERE, "train.py")); T = importlib.util.module_from_spec(sp); sp.loader.exec_module(T)
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--net", required=True); ap.add_argument("--val-mod", type=int, default=50); ap.add_argument("--cap", type=float, default=400)
a = ap.parse_args()
r = np.fromfile(a.data, dtype=T.REC); hce = np.fromfile(a.data + ".hce", dtype="<i2").astype(np.float64)
h = T.pos_hash(r); _, first = np.unique(h, return_index=True); first.sort(); r, hce, h = r[first], hce[first], h[first]
v = (h % np.uint64(a.val_mod)) == 0; r, hce = r[v], hce[v]
raw = open(a.net, "rb").read(); ver, H, B, F = struct.unpack("<IIII", raw[4:20]); o = 20
def take(n, dt):
    global o; x = np.frombuffer(raw, dtype=dt, count=n, offset=o); o += x.nbytes; return x.astype(np.float64)
ftw = take(F * H, "<i2").reshape(F, H); ftb = take(H, "<i2"); ow = take(B * 2 * H, "<i2").reshape(B, 2 * H); ob = take(B, "<i4")
A, Bk, K, S, W = T.decode(r)
def acc(idx):
    pad = np.vstack([ftw, np.zeros((1, H))]); return pad[idx.astype(np.int64)].sum(1) + ftb
au, at = acc(A), acc(Bk)
sc = lambda x: np.clip(x, 0, 255) ** 2
tot = (sc(au) * ow[K, :H]).sum(1) + (sc(at) * ow[K, H:]).sum(1)
delta = (tot / 255 + ob[K]) * 400 / (255 * 64)
lab = S.astype(np.float64)
ev = hce + np.clip(delta, -a.cap, a.cap)
c = lambda x, y: np.corrcoef(x, y)[0, 1]
print(f"n={len(lab)}  version={ver}")
print(f"corr(label, HCE)={c(lab,hce):.3f}   corr(label, HCE+delta[cap {a.cap:.0f}])={c(lab,ev):.3f}")
print(f"std  label={lab.std():.0f}  HCE={hce.std():.0f}  HCE+delta={ev.std():.0f}    mean|label|={np.abs(lab).mean():.0f} mean|HCE|={np.abs(hce).mean():.0f} mean|HCE+delta|={np.abs(ev).mean():.0f}")
print(f"mean|delta|={np.abs(delta).mean():.0f}  corr(delta, -HCE)={c(delta,-hce):.3f}   (1.0 = the net just shrinks HCE toward 0)")
dec = np.abs(lab) > 200
print(f"decisive labels (|SF|>200cp): {dec.mean()*100:.1f}% of positions; sign agreement HCE={np.mean(np.sign(hce[dec])==np.sign(lab[dec]))*100:.1f}%  HCE+delta={np.mean(np.sign(ev[dec])==np.sign(lab[dec]))*100:.1f}%")
q = ~dec
print(f"near-equal labels (|SF|<=200): mean|HCE|={np.abs(hce[q]).mean():.0f}  mean|HCE+delta|={np.abs(ev[q]).mean():.0f}")
