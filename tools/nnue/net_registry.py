#!/usr/bin/env python3
"""Net checkpoint registry with provenance, evidence-gated promotion and rollback (Agent D).

Stdlib only.  State lives in nets/registry.json; net binaries are archived by content hash in
nets/archive/<sha12>.nnue so an old champion can always be restored byte-for-byte.

  add       validate the CRN1 header, archive the file, record parent / data manifest / metrics
  promote   make a net the champion.  REFUSES without --evidence (e.g. an SPRT/ladder run URL)
            and without --h1-accepted unless --override-reason is given.
  rollback  restore the previous champion (or a named id) into nets/default.nnue
  list      show history

CRN1 layout (matches src/nnue.rs / train.py): "CRN1" | u32 ver,hidden,buckets,feats |
  i16 ft_w[feats*H] | i16 ft_b[H] | i16 out_w[buckets*2H] | i32 out_b[buckets]
"""
import argparse, hashlib, json, os, shutil, struct, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REG = os.path.join(ROOT, "nets", "registry.json")
ARCH = os.path.join(ROOT, "nets", "archive")
LIVE = os.path.join(ROOT, "nets", "default.nnue")
now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load():
    if os.path.exists(REG):
        return json.load(open(REG))
    return {"schema": 1, "champion": None, "history": [], "nets": {}}


def save(reg):
    os.makedirs(os.path.dirname(REG), exist_ok=True)
    json.dump(reg, open(REG, "w"), indent=1, sort_keys=True)


def check_crn1(path):
    raw = open(path, "rb").read()
    if raw[:4] != b"CRN1":
        sys.exit(f"{path}: bad magic {raw[:4]!r}")
    ver, h, b, f = struct.unpack("<IIII", raw[4:20])
    want = 20 + 2 * f * h + 2 * h + 2 * b * 2 * h + 4 * b
    if len(raw) != want:
        sys.exit(f"{path}: size {len(raw)} != expected {want} for ver={ver} H={h} buckets={b} feats={f}")
    return hashlib.sha256(raw).hexdigest(), {"version": ver, "hidden": h, "buckets": b, "features": f, "bytes": len(raw)}


def cmd_add(a):
    reg = load()
    digest, shape = check_crn1(a.net)
    nid = a.id or digest[:12]
    if nid in reg["nets"]:
        sys.exit(f"id {nid} already registered")
    if a.parent and a.parent not in reg["nets"]:
        sys.exit(f"unknown parent {a.parent}")
    os.makedirs(ARCH, exist_ok=True)
    shutil.copyfile(a.net, os.path.join(ARCH, digest[:12] + ".nnue"))
    manifest = None
    if a.data_manifest:
        m = json.load(open(a.data_manifest))
        manifest = {k: m.get(k) for k in ("records", "unique", "provenance", "split")}
        manifest["files"] = [{"file": x["file"], "sha256": x["sha256"], "records": x["records"]} for x in m.get("files", [])]
    reg["nets"][nid] = {
        "sha256": digest, "archive": f"nets/archive/{digest[:12]}.nnue", "shape": shape,
        "parent": a.parent, "created_utc": now(), "trainer_args": a.trainer_args, "data": manifest,
        "metrics": json.loads(a.metrics) if a.metrics else {}, "notes": a.notes, "status": "candidate",
    }
    save(reg)
    print(f"registered {nid} sha256={digest[:16]} {shape}")


def cmd_promote(a):
    reg = load()
    n = reg["nets"].get(a.id) or sys.exit(f"unknown id {a.id}")
    if not a.evidence:
        sys.exit("REFUSED: --evidence <run url / SPRT verdict> is required. No evidence, no promotion.")
    if not a.h1_accepted and not a.override_reason:
        sys.exit("REFUSED: pass --h1-accepted (SPRT / head-to-head gate passed) or --override-reason '<why>'.")
    prev = reg["champion"]
    n.update(status="champion", evidence=a.evidence, override_reason=a.override_reason)
    if prev and prev != a.id:
        reg["nets"][prev]["status"] = "retired"
    reg["champion"] = a.id
    reg["history"].append({"t": now(), "event": "promote", "id": a.id, "prev": prev, "evidence": a.evidence})
    shutil.copyfile(os.path.join(ROOT, n["archive"]), LIVE)
    save(reg)
    print(f"champion = {a.id} (prev {prev}); wrote nets/default.nnue")


def cmd_rollback(a):
    reg = load()
    target = a.id
    if not target:
        promos = [h for h in reg["history"] if h["event"] in ("promote", "rollback")]
        if not promos or not promos[-1].get("prev"):
            sys.exit("nothing to roll back to")
        target = promos[-1]["prev"]
    n = reg["nets"].get(target) or sys.exit(f"unknown id {target}")
    cur = reg["champion"]
    digest, _ = check_crn1(os.path.join(ROOT, n["archive"]))
    if digest != n["sha256"]:
        sys.exit("archive corrupted: hash mismatch")
    shutil.copyfile(os.path.join(ROOT, n["archive"]), LIVE)
    if cur and cur != target:
        reg["nets"][cur]["status"] = "rolled_back"
    n["status"] = "champion"
    reg["champion"] = target
    reg["history"].append({"t": now(), "event": "rollback", "id": target, "prev": cur, "reason": a.reason})
    save(reg)
    print(f"rolled back: champion = {target} (was {cur})")


def cmd_list(_a):
    reg = load()
    print("champion:", reg["champion"])
    for k, v in sorted(reg["nets"].items(), key=lambda kv: kv[1]["created_utc"]):
        s = v["shape"]
        print(f"{k:14} {v['status']:11} H={s['hidden']} B={s['buckets']} parent={v['parent']} metrics={v['metrics']}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("add")
    p.add_argument("net"); p.add_argument("--id"); p.add_argument("--parent"); p.add_argument("--data-manifest")
    p.add_argument("--metrics", help='JSON, e.g. \'{"val_mse":0.031}\'')
    p.add_argument("--trainer-args", default=""); p.add_argument("--notes", default=""); p.set_defaults(f=cmd_add)
    p = sp.add_parser("promote")
    p.add_argument("id"); p.add_argument("--evidence"); p.add_argument("--h1-accepted", action="store_true")
    p.add_argument("--override-reason"); p.set_defaults(f=cmd_promote)
    p = sp.add_parser("rollback")
    p.add_argument("id", nargs="?"); p.add_argument("--reason", default=""); p.set_defaults(f=cmd_rollback)
    p = sp.add_parser("list"); p.set_defaults(f=cmd_list)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
