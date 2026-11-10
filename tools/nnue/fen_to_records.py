#!/usr/bin/env python3
"""datagen text (`fen | cp | wdl`, cp and wdl from the side to move) -> 28-byte records (B's format).
Order is preserved, so consecutive records of one game stay consecutive (needed for --split-block).
    python tools/nnue/fen_to_records.py out.bin shard1.txt shard2.txt ...
"""
import sys
import numpy as np
REC = np.dtype([("occ", "<u8"), ("pcs", "u1", (16,)), ("stm", "u1"), ("score", "<i2"), ("wdl", "u1")])
CODE = {c: i for i, c in enumerate("PNBRQKpnbrqk")}

def main():
    out, files = sys.argv[1], sys.argv[2:]
    n = sum(1 for f in files for _ in open(f))
    rec = np.zeros(n, REC)
    k = 0
    for f in files:
        for line in open(f):
            fen, cp, wdl = [x.strip() for x in line.split("|")]
            parts = fen.split()
            sq_codes = []
            occ = 0
            for ri, row in enumerate(parts[0].split("/")):
                rank = 7 - ri
                file_ = 0
                for ch in row:
                    if ch.isdigit():
                        file_ += int(ch)
                    else:
                        sq = rank * 8 + file_
                        occ |= 1 << sq
                        sq_codes.append((sq, CODE[ch]))
                        file_ += 1
            sq_codes.sort()
            nib = [c for _, c in sq_codes] + [0] * (32 - len(sq_codes))
            pcs = bytes(nib[2 * i] | (nib[2 * i + 1] << 4) for i in range(16))
            rec[k] = (occ, np.frombuffer(pcs, np.uint8), 0 if parts[1] == "w" else 1, int(cp), int(round(float(wdl) * 2)))
            k += 1
    rec[:k].tofile(out)
    print(f"{k} records -> {out}; loss/draw/win = {np.bincount(rec['wdl'][:k], minlength=3)}")

if __name__ == "__main__":
    main()
