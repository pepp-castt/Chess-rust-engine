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
    ap.add_argument("--val", type=float, default=0.02)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(1)
    files = sorted(glob.glob(a.data, recursive=True))
    recs = np.concatenate([np.fromfile(f, dtype=REC) for f in files])
    # drop unlabeled-noise: huge scores already filtered at datagen; dedupe is skipped on purpose
    rng = np.random.default_rng(1)
    rng.shuffle(recs)
    nval = max(1000, int(len(recs) * a.val))
    val, tr = recs[:nval], recs[nval:]
    print(f"files={len(files)} positions={len(recs)} train={len(tr)} val={len(val)}", flush=True)
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
        raw = open(a.resume, "rb").read()
        print("resume not implemented for quantised files; training from scratch")
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

    while not done:
        perm = torch.randperm(n)
        run = 0.0
        for bi in range(steps_per_epoch):
            frac = (time.time() - t_start) / budget
            if frac >= 1.0:
                done = True
                break
            lr = a.lr * (0.5 * (1 + math.cos(math.pi * frac)) * 0.98 + 0.02)
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
        print(f"== epoch {epoch} val_loss {validate():.5f} elapsed {(time.time()-t_start)/60:.1f}m", flush=True)
        if time.time() - last_export > 300:
            export(net, a.out)
            last_export = time.time()
    print(f"final val_loss {validate():.5f}")
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
