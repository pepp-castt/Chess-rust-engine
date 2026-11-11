#!/usr/bin/env python3
"""CPU PyTorch trainer for the (768 -> H)x2 -> 1 SCReLU net with 8 material output buckets.
Exports `CRN1` net files that src/nnue.rs reads. Loss: MSE in sigmoid space against
    target = lam * sigmoid(sf_cp / 400) + (1 - lam) * game_result.
"""
import argparse, glob, math, os, struct, sys, time
import numpy as np
import torch

H, FEATS, BUCKETS = 128, 768, 8
QA, QB, SCALE = 255, 64, 400
REC = np.dtype([("occ", "<u8"), ("pcs", "u1", (16,)), ("stm", "u1"), ("score", "<i2"), ("wdl", "u1")])
assert REC.itemsize == 28


def pos_hash(recs: np.ndarray) -> np.ndarray:
    """64-bit hash of (occ, pieces, stm) - the position identity used for dedupe and for the
    deterministic train/validation split (so validation never sees training positions, even
    across stages and data sources)."""
    M = np.uint64
    pcs = np.ascontiguousarray(recs["pcs"]).view("<u8").reshape(len(recs), 2)
    h = recs["occ"].astype(M) * M(0x9E3779B97F4A7C15)
    h ^= pcs[:, 0] * M(0xC2B2AE3D27D4EB4F)
    h ^= pcs[:, 1] * M(0x165667B19E3779F9)
    h ^= recs["stm"].astype(M) * M(0x27D4EB2F165667C5)
    h ^= h >> M(31)
    h *= M(0xBF58476D1CE4E5B9)
    h ^= h >> M(29)
    return h


def load_net(net, path: str):
    """Load a quantised CRN1 file back into float parameters (stage-2 fine-tune / rollback)."""
    raw = open(path, "rb").read()
    assert raw[:4] == b"CRN1", "not a CRN1 net"
    ver, h, b, f = struct.unpack("<IIII", raw[4:20])
    assert (h, b, f) == (H, BUCKETS, FEATS), f"net shape {f}x{h} b{b} does not match trainer {FEATS}x{H} b{BUCKETS}"
    o = 20
    def take(n, dt):
        nonlocal o
        a = np.frombuffer(raw, dtype=dt, count=n, offset=o)
        o += a.nbytes
        return a.astype(np.float32)
    ft_w = take(FEATS * H, "<i2").reshape(FEATS, H) / QA
    ft_b = take(H, "<i2") / QA
    out_w = take(BUCKETS * 2 * H, "<i2").reshape(BUCKETS, 2 * H) / QB
    out_b = take(BUCKETS, "<i4") / (QA * QB)
    assert o == len(raw), "trailing bytes in net file"
    with torch.no_grad():
        net.ft.weight[:FEATS].copy_(torch.from_numpy(ft_w))
        net.ft.weight[FEATS].zero_()
        net.ft_b.copy_(torch.from_numpy(ft_b))
        net.out_w.copy_(torch.from_numpy(out_w))
        net.out_b.copy_(torch.from_numpy(out_b))


def load_records(spec: str):
    files = []
    for g in spec.split(","):
        files += sorted(glob.glob(g.strip(), recursive=True))
    files = sorted(set(files))
    arrs = [np.fromfile(f, dtype=REC) for f in files]
    return files, (np.concatenate(arrs) if arrs else np.zeros(0, REC))


def decode(recs: np.ndarray):
    """-> f_us, f_them (N,32) int16 padded with 768, bucket (N,), score f32, wdl f32"""
    n = len(recs)
    occ = np.ascontiguousarray(recs["occ"]).view(np.uint8).reshape(n, 8)
    mask = np.unpackbits(occ, axis=1, bitorder="little").astype(bool)  # (n,64)
    cnt = mask.sum(1)
    pcs = recs["pcs"]
    nib = np.empty((n, 32), np.uint8)
    nib[:, 0::2] = pcs & 15
    nib[:, 1::2] = pcs >> 4
    valid = np.arange(32)[None, :] < cnt[:, None]
    rows, cols = np.nonzero(mask)
    p = nib[valid].astype(np.int64)
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    slot = np.arange(len(rows)) - off[rows]
    stm = recs["stm"].astype(np.int64)[rows]
    c, t = p // 6, p % 6
    f_us = ((c != stm) * 6 + t) * 64 + (cols ^ (56 * stm))
    them = 1 - stm
    f_th = ((c != them) * 6 + t) * 64 + (cols ^ (56 * them))
    A = np.full((n, 32), FEATS, np.int16)
    B = np.full((n, 32), FEATS, np.int16)
    A[rows, slot] = f_us
    B[rows, slot] = f_th
    bucket = np.minimum(np.maximum(cnt - 2, 0) // 4, BUCKETS - 1).astype(np.int64)
    return A, B, bucket, recs["score"].astype(np.float32), recs["wdl"].astype(np.float32) / 2.0


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = torch.nn.EmbeddingBag(FEATS + 1, H, mode="sum", padding_idx=FEATS)
        torch.nn.init.normal_(self.ft.weight, 0.0, 0.08)
        with torch.no_grad():
            self.ft.weight[FEATS].zero_()
        self.ft_b = torch.nn.Parameter(torch.zeros(H))
        self.out_w = torch.nn.Parameter(torch.randn(BUCKETS, 2 * H) * 0.05)
        self.out_b = torch.nn.Parameter(torch.zeros(BUCKETS))

    def forward(self, f_us, f_th, bucket):
        a = torch.clamp(self.ft(f_us) + self.ft_b, 0, 1) ** 2
        b = torch.clamp(self.ft(f_th) + self.ft_b, 0, 1) ** 2
        x = torch.cat([a, b], 1)
        return (x * self.out_w[bucket]).sum(1) + self.out_b[bucket]

    def clip(self):
        with torch.no_grad():
            self.ft.weight.clamp_(-1.98, 1.98)
            self.ft.weight[FEATS].zero_()
            self.ft_b.clamp_(-1.98, 1.98)
            self.out_w.clamp_(-1.98, 1.98)


def export(net: Net, path: str):
    q = lambda t, s, lim: np.clip(np.round(t.detach().numpy() * s), -lim, lim).astype("<i2")
    ft_w = q(net.ft.weight[:FEATS], QA, 32767)
    ft_b = q(net.ft_b, QA, 32767)
    out_w = q(net.out_w, QB, 32767)
    out_b = np.round(net.out_b.detach().numpy() * QA * QB).astype("<i4")
    with open(path, "wb") as f:
        f.write(b"CRN1" + struct.pack("<IIII", 1, H, BUCKETS, FEATS))
        f.write(ft_w.tobytes() + ft_b.tobytes() + out_w.tobytes() + out_b.tobytes())
    print(f"exported {path} ({os.path.getsize(path)} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="glob of shard files")
    ap.add_argument("--out", default="net.bin")
    ap.add_argument("--minutes", type=float, default=60, help="wall-clock training budget")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=0.75, help="weight of SF score vs game result")
    ap.add_argument("--val-mod", type=int, default=50, help="validation = positions with hash %% N == 0 (~2%%), never trained on")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--min-lr-frac", type=float, default=0.02)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(1)
    files, recs = load_records(a.data)
    raw_n = len(recs)
    h = pos_hash(recs)
    if not a.no_dedupe:
        _, first = np.unique(h, return_index=True)
        first.sort()
        recs, h = recs[first], h[first]
    is_val = (h % np.uint64(a.val_mod)) == 0
    val, tr = recs[is_val], recs[~is_val]
    rng = np.random.default_rng(1)
    rng.shuffle(tr)
    if len(val) < 1000:
        raise SystemExit(f"only {len(val)} validation positions; need more data (or lower --val-mod)")
    res = np.bincount(recs["wdl"], minlength=3)
    print(f"files={len(files)} raw={raw_n} unique={len(recs)} (dedupe removed {raw_n-len(recs)}) "
          f"train={len(tr)} val={len(val)} loss/draw/win={res[0]}/{res[1]}/{res[2]}", flush=True)
    t0 = time.time()
    chunks = [decode(tr[i : i + 500000]) for i in range(0, len(tr), 500000)]
    TA, TB, TK, TS, TW = [np.concatenate([c[j] for c in chunks]) for j in range(5)]
    VA, VB, VK, VS, VW = decode(val)
    del chunks, recs
    print(f"decoded in {time.time()-t0:.0f}s", flush=True)
    TA, TB, TK = map(torch.from_numpy, (TA, TB, TK))
    TS, TW = torch.from_numpy(TS), torch.from_numpy(TW)

    def target(s, w):
        return a.lam * torch.sigmoid(s / SCALE) + (1 - a.lam) * w

    net = Net()
    if a.resume:
        load_net(net, a.resume)
        print(f"resumed from {a.resume}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    n = len(TA)
    steps_per_epoch = max(1, n // a.batch)
    t_start = time.time()
    budget = a.minutes * 60
    step = 0
    epoch = 0
    done = False
    last_export = time.time()

    def validate():
        net.eval()
        with torch.no_grad():
            tot = 0.0
            for i in range(0, len(VA), 65536):
                sl = slice(i, i + 65536)
                o = net(torch.from_numpy(VA[sl]).long(), torch.from_numpy(VB[sl]).long(), torch.from_numpy(VK[sl]))
                t = target(torch.from_numpy(VS[sl]), torch.from_numpy(VW[sl]))
                tot += ((torch.sigmoid(o) - t) ** 2).sum().item()
        net.train()
        return tot / len(VA)

    best = {"loss": validate(), "state": {k: v.clone() for k, v in net.state_dict().items()}, "epoch": 0}
    print(f"== start val_loss {best['loss']:.5f} (a resumed net that cannot improve on this is rolled back)", flush=True)

    def checkpoint(tag):
        v = validate()
        print(f"== {tag} val_loss {v:.5f} best {best['loss']:.5f} elapsed {(time.time()-t_start)/60:.1f}m", flush=True)
        if v < best["loss"]:
            best.update(loss=v, state={k: t.clone() for k, t in net.state_dict().items()}, epoch=epoch)
            net.eval(); export(net, a.out); net.train()

    while not done:
        perm = torch.randperm(n)
        run = 0.0
        for bi in range(steps_per_epoch):
            frac = (time.time() - t_start) / budget
            if frac >= 1.0:
                done = True
                break
            lr = a.lr * (0.5 * (1 + math.cos(math.pi * frac)) * (1 - a.min_lr_frac) + a.min_lr_frac)
            for g in opt.param_groups:
                g["lr"] = lr
            idx = perm[bi * a.batch : (bi + 1) * a.batch]
            out = net(TA[idx].long(), TB[idx].long(), TK[idx])
            loss = ((torch.sigmoid(out) - target(TS[idx], TW[idx])) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            net.clip()
            run += loss.item()
            step += 1
            if step % 100 == 0:
                el = time.time() - t_start
                print(f"ep {epoch} step {step} loss {run/(bi+1):.5f} lr {lr:.2e} {step*a.batch/el:,.0f} pos/s elapsed {el/60:.1f}m", flush=True)
        epoch += 1
        checkpoint(f"epoch {epoch}")
    checkpoint("final")
    net.load_state_dict(best["state"])  # rollback to the best validation checkpoint
    print(f"final val_loss {best['loss']:.5f} (best, epoch {best['epoch']})")
    export(net, a.out)
    # parity file: float eval (cp) for a few FENs, compared against `eval` in the engine
    import chess
    fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"]
    rows = []
    for fen in fens:
        b = chess.Board(fen)
        r = np.zeros(1, REC)[0:1]
        occ = 0; nib = []
        for sq in range(64):
            p = b.piece_at(sq)
            if p is not None:
                occ |= 1 << sq
                nib.append((0 if p.color else 6) + p.piece_type - 1)
        nib += [0] * (32 - len(nib))
        r["occ"] = occ
        r["pcs"] = np.array([[nib[2*i] | (nib[2*i+1] << 4) for i in range(16)]], np.uint8)
        r["stm"] = 0 if b.turn else 1
        A, B, K, _, _ = decode(r)
        with torch.no_grad():
            o = net(torch.from_numpy(A).long(), torch.from_numpy(B).long(), torch.from_numpy(K)).item()
        rows.append(f"{fen}|{o*SCALE:.1f}")
    open(a.out + ".parity.txt", "w").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()

