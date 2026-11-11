//! Texel-style distillation of the hand-crafted evaluation (Agent I).
//!
//! Fits every number in `eval::Params` so that `sigmoid(hce / K)` matches
//! `sigmoid(stockfish_score / K)` on SF19 MAX labelled positions (the 28-byte
//! records produced by tools/nnue/datagen*.py / relabel_perturbed.py).
//!
//! * Stockfish is only a label source here. Nothing of its network or code is used.
//! * Validation positions (`hash % 20 == 0`, same rule as tools/nnue/train.py and
//!   data_audit.py) are NEVER optimised on. The returned parameters are the
//!   best-VALIDATION ones, not the last ones (rollback).
//! * Pure std, no dependencies. Deterministic for a given seed and thread count.
//!
//! usage: texel --data a.bin,b.bin [--out tuned.params] [--emit-in hce_params.in]
//!              [--steps 4,2,1] [--passes 12] [--k 400] [--reg 1e-8] [--threads 4]
//!              [--init file] [--limit N] [--eval-only] [--dump-default]

use chess_rust_engine::board::Position;
use chess_rust_engine::eval::*;
use chess_rust_engine::types::{Color, Piece, PieceType, Square};
use std::collections::HashSet;
use std::io::Write;

struct Sample {
    pos: Position,
    target: f64, // sigmoid(sf / K) from the side to move's point of view
    score: i32,
    src: u8,
    val: bool,
}

fn sigmoid(x: f64, k: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-x / k))
}

fn load(path: &str, src: u8, k: f64, seen: &mut HashSet<(u64, u8)>, out: &mut Vec<Sample>) -> (usize, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert!(bytes.len() % 28 == 0, "{path}: size {} is not a multiple of 28", bytes.len());
    let (mut kept, mut dup) = (0, 0);
    for r in bytes.chunks_exact(28) {
        let occ = u64::from_le_bytes(r[0..8].try_into().unwrap());
        let nib = &r[8..24];
        let stm = r[24];
        let score = i16::from_le_bytes([r[25], r[26]]) as i32;
        let mut pos = Position::empty();
        let (mut wk, mut bk) = (0, 0);
        let mut o = occ;
        let mut n = 0usize;
        while o != 0 {
            let sq = o.trailing_zeros() as u8;
            o &= o - 1;
            let v = (nib[n / 2] >> (4 * (n % 2))) & 0xF;
            n += 1;
            if v >= 12 {
                wk = 99;
                break;
            }
            let (c, t) = ((v / 6) as usize, (v % 6) as usize);
            if t == 5 {
                if c == 0 { wk += 1 } else { bk += 1 }
            }
            pos.place(Square(sq), Piece::new(Color::from_index(c), PieceType::from_index(t)));
        }
        if wk != 1 || bk != 1 || stm > 1 || n > 32 {
            continue;
        }
        pos.side = Color::from_index(stm as usize);
        if !seen.insert((pos.hash, stm)) {
            dup += 1;
            continue;
        }
        let val = pos.hash % 20 == 0;
        out.push(Sample { pos, target: sigmoid(score as f64, k), score, src, val });
        kept += 1;
    }
    (kept, dup)
}

fn sse(samples: &[Sample], p: &Params, k: f64, threads: usize) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let chunk = samples.len().div_ceil(threads);
    std::thread::scope(|s| {
        let hs: Vec<_> = samples
            .chunks(chunk)
            .map(|c| {
                s.spawn(move || {
                    let mut t = 0.0;
                    for x in c {
                        let e = evaluate_hce_with(&x.pos, p) as f64;
                        let d = sigmoid(e, k) - x.target;
                        t += d * d;
                    }
                    t
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

fn reg_term(p: &Params, p0: &Params, lam: f64) -> f64 {
    let mut t = 0.0;
    for i in 0..NP {
        let d = (p[i] - p0[i]) as f64;
        t += d * d;
    }
    lam * t
}

fn report(tag: &str, p: &Params, val: &[Sample], k: f64) {
    let bands: [(i32, i32); 5] = [(0, 50), (50, 100), (100, 300), (300, 1000), (1000, 99999)];
    let mut cnt = [0usize; 5];
    let mut err = [0f64; 5];
    let mut sign_ok = 0usize;
    let mut sign_n = 0usize;
    let mut by_src = [(0usize, 0f64); 8];
    let mut total = 0.0;
    for x in val {
        let e = evaluate_hce_with(&x.pos, p);
        let d = sigmoid(e as f64, k) - x.target;
        total += d * d;
        let a = x.score.abs();
        for (i, (lo, hi)) in bands.iter().enumerate() {
            if a >= *lo && a < *hi {
                cnt[i] += 1;
                err[i] += d * d;
            }
        }
        if a >= 100 {
            sign_n += 1;
            if (e > 0) == (x.score > 0) && e != 0 {
                sign_ok += 1;
            }
        }
        let s = (x.src as usize).min(7);
        by_src[s].0 += 1;
        by_src[s].1 += d * d;
    }
    println!("[{tag}] VAL mse {:.6} (n={})  sign-agree(|sf|>=100) {:.2}%", total / val.len().max(1) as f64, val.len(), 100.0 * sign_ok as f64 / sign_n.max(1) as f64);
    let mut line = String::from("   by |sf|:");
    for (i, (lo, hi)) in bands.iter().enumerate() {
        line += &format!("  [{lo},{hi}) n={} mse={:.5}", cnt[i], err[i] / cnt[i].max(1) as f64);
    }
    println!("{line}");
    let mut line = String::from("   by src: ");
    for (i, (n, e)) in by_src.iter().enumerate() {
        if *n > 0 {
            line += &format!("  src{i} n={n} mse={:.5}", e / *n as f64);
        }
    }
    println!("{line}");
}

fn write_params(path: &str, p: &Params) {
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "# hce params v1 NP={NP}").unwrap();
    for i in 0..NP {
        write!(f, "{}{}", p[i], if i + 1 == NP { "\n" } else { " " }).unwrap();
    }
}

fn read_params(path: &str) -> Params {
    let t = std::fs::read_to_string(path).unwrap();
    let v: Vec<i32> = t.lines().filter(|l| !l.starts_with('#')).flat_map(|l| l.split_whitespace()).map(|x| x.parse().unwrap()).collect();
    assert_eq!(v.len(), NP, "params file must hold {NP} ints");
    let mut p = [0; NP];
    p.copy_from_slice(&v);
    p
}

fn emit_in(path: &str, p: &Params, header: &str) {
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "// {header}").unwrap();
    writeln!(f, "[").unwrap();
    for (i, c) in p.chunks(16).enumerate() {
        let s: Vec<String> = c.iter().map(|x| x.to_string()).collect();
        writeln!(f, "    {},{}", s.join(", "), if i == 0 { " // layout: see eval.rs P_* constants" } else { "" }).unwrap();
    }
    writeln!(f, "]").unwrap();
}

/// Parameters that cannot influence the evaluation (or are gauge-fixed).
fn frozen(i: usize) -> bool {
    if i == P_MG_VALUE + 5 || i == P_EG_VALUE + 5 {
        return true; // king value
    }
    for base in [P_MG_PST, P_EG_PST] {
        if i >= base && i < base + 64 {
            let sq = i - base; // pawns on rank 1/8 cannot exist
            return sq < 8 || sq >= 56;
        }
    }
    if i == P_PASSED_MG || i == P_PASSED_EG || i == P_PASSED_MG + 7 || i == P_PASSED_EG + 7 {
        return true;
    }
    false
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1)).cloned();
    let has = |f: &str| args.iter().any(|a| a == f);
    let _ = chess_rust_engine::attacks::tables();

    if has("--dump-default") {
        let out = get("--emit-in").unwrap_or_else(|| "hce_params.in".into());
        emit_in(&out, &PESTO_PARAMS, "PeSTO baseline (src/eval.rs consts). Generated by `texel --dump-default`.");
        println!("wrote {out}");
        return;
    }
    let k: f64 = get("--k").map(|v| v.parse().unwrap()).unwrap_or(400.0);
    let lam: f64 = get("--reg").map(|v| v.parse().unwrap()).unwrap_or(1e-8);
    let passes: usize = get("--passes").map(|v| v.parse().unwrap()).unwrap_or(12);
    let threads: usize = get("--threads").map(|v| v.parse().unwrap()).unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let steps: Vec<i32> = get("--steps").unwrap_or_else(|| "4,2,1".into()).split(',').map(|x| x.parse().unwrap()).collect();
    let limit: usize = get("--limit").map(|v| v.parse().unwrap()).unwrap_or(usize::MAX);
    let out = get("--out").unwrap_or_else(|| "tuned.params".into());

    let mut samples = Vec::new();
    let mut seen = HashSet::new();
    for (i, path) in get("--data").expect("--data a.bin,b.bin").split(',').enumerate() {
        let (kept, dup) = load(path, i as u8, k, &mut seen, &mut samples);
        println!("src{i} {path}: kept {kept}, dropped {dup} duplicates");
    }
    // deterministic order independent of file order is not needed; limit for smoke runs
    if samples.len() > limit {
        samples.truncate(limit);
    }
    let (val, train): (Vec<Sample>, Vec<Sample>) = samples.into_iter().partition(|s| s.val);
    println!("train {} / val {} (val = pos_hash % 20 == 0, never optimised on)", train.len(), val.len());

    let p0 = PESTO_PARAMS;
    let mut p: Params = match get("--init") {
        Some(f) => read_params(&f),
        None => DEFAULT_PARAMS,
    };
    report("start", &p, &val, k);
    if has("--eval-only") {
        return;
    }
    let active: Vec<usize> = (0..NP).filter(|i| !frozen(*i)).collect();
    println!("tuning {} params, K={k}, reg={lam:e}, steps={steps:?}, passes<={passes}, threads={threads}", active.len());

    let t0 = std::time::Instant::now();
    let mut cur = sse(&train, &p, k, threads) / train.len() as f64 + reg_term(&p, &p0, lam);
    let mut best_val = sse(&val, &p, k, threads) / val.len() as f64;
    let mut best = p;
    let mut rng = 0x9E3779B97F4A7C15u64;
    println!("start: train {cur:.6} val {best_val:.6}");
    for &step in &steps {
        let mut bad = 0;
        for pass in 0..passes {
            // deterministic shuffle
            let mut order = active.clone();
            for i in (1..order.len()).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                order.swap(i, (rng % (i as u64 + 1)) as usize);
            }
            let mut improved = 0;
            for &i in &order {
                let orig = p[i];
                for d in [step, -step] {
                    p[i] = orig + d;
                    let l = sse(&train, &p, k, threads) / train.len() as f64 + reg_term(&p, &p0, lam);
                    if l < cur - 1e-12 {
                        cur = l;
                        improved += 1;
                        break;
                    }
                    p[i] = orig;
                }
            }
            let v = sse(&val, &p, k, threads) / val.len() as f64;
            println!("step {step} pass {pass}: improved {improved:4}  train {cur:.6}  val {v:.6}  best-val {best_val:.6}  [{:.0}s]", t0.elapsed().as_secs_f64());
            if v < best_val - 1e-7 {
                best_val = v;
                best = p;
                write_params(&out, &best);
                bad = 0;
            } else {
                bad += 1;
            }
            if improved == 0 || bad >= 2 {
                break; // converged at this step size, or validation stopped improving
            }
        }
        p = best; // rollback to best validation before refining
        cur = sse(&train, &p, k, threads) / train.len() as f64 + reg_term(&p, &p0, lam);
    }
    write_params(&out, &best);
    report("PeSTO", &p0, &val, k);
    report("tuned", &best, &val, k);
    if let Some(f) = get("--emit-in") {
        emit_in(&f, &best, &format!("Texel-tuned vs SF19 MAX labels; best-val mse {best_val:.6}; K={k} reg={lam:e}"));
        println!("wrote {f}");
    }
    let changed = (0..NP).filter(|i| best[*i] != p0[*i]).count();
    println!("done: {changed}/{NP} params differ from PeSTO; best val mse {best_val:.6}; wrote {out}");
}
