#!/usr/bin/env python3
"""Train an HCE-anchored *residual* net:  eval = HCE_static + 400 * net(x).

The output layer is zero-initialised, so at step 0 the engine is exactly HCE. The loss is the
same sigmoid-space MSE as `train.py`, but the logit is `hce/400 + out`, so the net only has to
learn *what HCE gets wrong* relative to the label (SF19 / own deep search), not re-derive
material and piece-square values from scratch.

Reuses `train.py` for the record layout (`REC`), position hash (dedupe + the deterministic
`hash % val_mod` held-out split, so validation is never trained on), feature decode and the
quantiser constants. Writes a CRN1 file with header **version = 2** (= residual).

Reports, on held-out validation positions, the loss of pure HCE (delta = 0) and of HCE+net,
and the gain. `--min-gain` rolls back (writes nothing) when the net does not beat HCE.

    python tools/nnue/hce_labels.py --engine target/release/chess-rust-engine data/x.bin
    python tools/nnue/train_residual.py --data 'data/x.bin' --out nets/resid.bin --minutes 10
"""
import argparse, importlib.util, math, os, struct, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("train_b", os.path.join(HERE, "train.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)  # B's trainer: decode, pos_hash, REC, Net, constants (main() is guarded)
H, FEATS, BUCKETS, QA, QB, SCALE = T.H, T.FEATS, T.BUCKETS, T.QA, T.QB, T.SCALE


class ResNet(T.Net):
    def __init__(self):
        super().__init__()
        with torch.no_grad():
            self.out_w.zero_()
            self.out_b.zero_()


def export_v2(net, path):
    q = lambda t, s, lim: np.clip(np.round(t.detach().numpy() * s), -lim, lim).astype("<i2")
    with open(path, "wb") as f:
        f.write(b"CRN1" + struct.pack("<IIII", 2, H, BUCKETS, FEATS))
        f.write(q(net.ft.weight[:FEATS], QA, 32767).tobytes() + q(net.ft_b, QA, 32767).tobytes())
        f.write(q(net.out_w, QB, 32767).tobytes())
        f.write(np.round(net.out_b.detach().numpy() * QA * QB).astype("<i4").tobytes())
    print(f"exported {path} ({os.path.getsize(path)} bytes, version 2 = residual)", flush=True)


def load(spec_glob):
    import glob
    files = sorted(set(sum([sorted(glob.glob(g.strip(), recursive=True)) for g in spec_glob.split(",")], [])))
    recs, hce = [], []
    for f in files:
        r = np.fromfile(f, dtype=T.REC)
        h = np.fromfile(f + ".hce", dtype="<i2")
        assert len(h) == len(r), f"{f}.hce missing/misaligned - run hce_labels.py"
        recs.append(r); hce.append(h)
    return files, np.concatenate(recs), np.concatenate(hce)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="resid.bin")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=0.75, help="weight of search/SF score vs game result")
    ap.add_argument("--val-mod", type=int, default=50)
    ap.add_argument("--min-gain", type=float, default=0.0, help="required relative val-loss gain over HCE to export")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(1)
    files, recs, hce = load(a.data)
    h = T.pos_hash(recs)
    _, first = np.unique(h, return_index=True); first.sort()
    recs, hce, h = recs[first], hce[first], h[first]
    is_val = (h % np.uint64(a.val_mod)) == 0
    vi, ti = np.nonzero(is_val)[0], np.nonzero(~is_val)[0]
    rng = np.random.default_rng(1); rng.shuffle(ti)
    print(f"files={len(files)} unique={len(recs)} train={len(ti)} val={len(vi)}", flush=True)

    def dec(idx):
        ch = [T.decode(recs[idx[i:i + 400000]]) for i in range(0, len(idx), 400000)]
        A, B, K, S, W = [np.concatenate([c[j] for c in ch]) for j in range(5)]
        return (torch.from_numpy(A), torch.from_numpy(B), torch.from_numpy(K), torch.from_numpy(S), torch.from_numpy(W),
                torch.from_numpy(hce[idx].astype(np.float32)))
    TA, TB, TK, TS, TW, TH = dec(ti)
    VA, VB, VK, VS, VW, VH = dec(vi)
    tgt = lambda s, w: a.lam * torch.sigmoid(s / SCALE) + (1 - a.lam) * w

    def val_loss(net):
        net.eval()
        with torch.no_grad():
            tot = 0.0
            for i in range(0, len(VA), 65536):
                sl = slice(i, i + 65536)
                o = net(VA[sl].long(), VB[sl].long(), VK[sl])
                p = torch.sigmoid(VH[sl] / SCALE + o)
                tot += ((p - tgt(VS[sl], VW[sl])) ** 2).sum().item()
        net.train()
        return tot / len(VA)

    net = ResNet()
    base = val_loss(net)  # delta == 0  ->  pure HCE
    print(f"== HCE-only held-out loss {base:.5f}   (net must beat this)", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    n, spe = len(TA), max(1, len(TA) // a.batch)
    t0, budget, step, best, done = time.time(), a.minutes * 60, 0, (base, None), False
    last_val = t0
    while not done:
        perm = torch.randperm(n)
        for bi in range(spe):
            frac = (time.time() - t0) / budget
            if frac >= 1.0:
                done = True; break
            lr = a.lr * (0.5 * (1 + math.cos(math.pi * frac)) * 0.98 + 0.02)
            for g in opt.param_groups: g["lr"] = lr
            idx = perm[bi * a.batch:(bi + 1) * a.batch]
            o = net(TA[idx].long(), TB[idx].long(), TK[idx])
            loss = ((torch.sigmoid(TH[idx] / SCALE + o) - tgt(TS[idx], TW[idx])) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); net.clip(); step += 1
            if time.time() - last_val > 60:
                last_val = time.time(); v = val_loss(net)
                print(f"step {step} train {loss.item():.5f} val {v:.5f} (HCE {base:.5f}, gain {100*(base-v)/base:+.2f}%) {(time.time()-t0)/60:.1f}m", flush=True)
                if v < best[0]:
                    best = (v, {k: t.clone() for k, t in net.state_dict().items()})
    v = val_loss(net)
    if v < best[0]:
        best = (v, {k: t.clone() for k, t in net.state_dict().items()})
    gain = (base - best[0]) / base
    print(f"== FINAL best val {best[0]:.5f} vs HCE {base:.5f}: gain {100*gain:+.2f}%", flush=True)
    if best[1] is None or gain < a.min_gain:
        print("== ROLLBACK: net does not beat HCE on held-out data; nothing exported", flush=True)
        sys.exit(3)
    net.load_state_dict(best[1]); net.eval(); export_v2(net, a.out)
    # magnitude of the correction the net applies (cp), so the gate can pick CRE_DELTA_CAP sensibly
    with torch.no_grad():
        d = torch.cat([net(VA[i:i+65536].long(), VB[i:i+65536].long(), VK[i:i+65536]) for i in range(0, len(VA), 65536)]) * SCALE
    print(f"== delta on val (cp): mean|d|={d.abs().mean():.1f} p50={d.abs().median():.1f} p95={d.abs().quantile(0.95):.1f} max={d.abs().max():.0f}", flush=True)


if __name__ == "__main__":
    main()
