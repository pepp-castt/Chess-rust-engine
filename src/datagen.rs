//! Self-play datagen for NNUE. Depth-limited search, never 100–120 ms teacher
//! games. Positions are quiet, |eval| < 1500, not in check.

use crate::board::Position;
use crate::movegen::legal_moves;
use crate::search::{iterative_deepening, ThreadData};
use crate::time::{Limits, TimeManager};
use crate::tt::TranspositionTable;
use crate::types::{Color, Move, MAX_MOVES};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn u32(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
}

fn load_openings() -> Vec<String> {
    // Real book lines, not the match-harness EPD (that file is random legal
    // junk used to diversify SF ladder games — it put kings on f6 at move 3
    // and the first net lost 0–10–10 to HCE).
    let mut v: Vec<String> = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
        "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1",
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
        "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
        "rnbqkbnr/pppppp1p/6p1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
        "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 1 3",
        "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
        "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
        "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    for path in ["tools/nnue_openings.epd"] {
        if let Ok(text) = std::fs::read_to_string(path) {
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let fen = line.split(';').next().unwrap_or(line).trim();
                let parts: Vec<&str> = fen.split_whitespace().collect();
                if parts.len() >= 4 {
                    v.push(format!("{} {} {} {} 0 1", parts[0], parts[1], parts[2], parts[3]));
                }
            }
        }
    }
    v
}

pub fn run(n: usize, depth: i32, out_path: &str) {
    let openings = load_openings();
    let tt = Arc::new(TranspositionTable::new(16));
    // Shards must not replay the same games. `CRE_DATAGEN_SEED=<n>` selects a
    // different stream; unset keeps the historical seed (bit-identical output).
    const DEFAULT_SEED: u64 = 0xC0FF_EE12_3456_789A;
    let seed = std::env::var("CRE_DATAGEN_SEED")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(|v| (v.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ DEFAULT_SEED).max(1))
        .unwrap_or(DEFAULT_SEED);
    let mut rng = Rng(seed);
    let file = File::create(out_path).expect("datagen out");
    let mut w = BufWriter::new(file);
    let start = Instant::now();
    let mut written = 0usize;
    let mut games = 0u32;

    while written < n {
        games += 1;
        let fen = &openings[rng.u32(openings.len() as u32) as usize];
        let mut pos = Position::from_fen(fen).unwrap_or_else(Position::startpos);
        // short scramble from the book — long random walks produce junk kings
        // on f6 at move 7, which even SF19 labels into a net that loses to HCE.
        let scramble = rng.u32(5) as i32;
        for _ in 0..scramble {
            let mut moves = [Move::NONE; MAX_MOVES];
            let k = legal_moves(&pos, &mut moves);
            if k == 0 {
                break;
            }
            let mv = moves[rng.u32(k as u32) as usize];
            pos.make(mv);
            if pos.was_illegal() {
                break;
            }
        }
        let mut samples: Vec<(String, i32, Color)> = Vec::new();
        let mut result_white = 0.5f32;
        for _ply in 0..160 {
            if pos.halfmove >= 100 {
                result_white = 0.5;
                break;
            }
            let mut moves = [Move::NONE; MAX_MOVES];
            let k = legal_moves(&pos, &mut moves);
            if k == 0 {
                result_white = if pos.in_check() {
                    if pos.side == Color::White {
                        0.0
                    } else {
                        1.0
                    }
                } else {
                    0.5
                };
                break;
            }
            let limits = Limits {
                depth: Some(depth),
                infinite: false,
                ..Default::default()
            };
            let tm = Arc::new(TimeManager::new(limits, pos.side == Color::White, 0));
            let mut td = ThreadData::new(tt.clone(), tm);
            td.silent = true;
            let res = iterative_deepening(&pos, &mut td, false);
            if res.best.is_none() {
                break;
            }
            let in_check = pos.in_check();
            let noisy = res.best.is_capture() || res.best.is_promo();
            if !in_check && !noisy && res.score.abs() < 1500 && k > 1 {
                samples.push((pos.to_fen(), res.score, pos.side));
            }
            // 4% random to leave book / avoid loops (was 12%; too much noise)
            let mv = if rng.u32(100) < 4 {
                moves[rng.u32(k as u32) as usize]
            } else {
                res.best
            };
            pos.make(mv);
            if pos.was_illegal() {
                break;
            }
            if res.score.abs() > 12_000 {
                result_white = if pos.side == Color::White { 0.0 } else { 1.0 };
                break;
            }
        }
        for (fen, cp, stm) in samples {
            if written >= n {
                break;
            }
            let wdl = if stm == Color::White {
                result_white
            } else {
                1.0 - result_white
            };
            let _ = writeln!(w, "{} | {} | {:.1}", fen, cp, wdl);
            written += 1;
        }
        tt.clear();
        if games % 25 == 0 {
            eprintln!(
                "datagen games {} positions {} ({:.1}/s)",
                games,
                written,
                written as f64 / start.elapsed().as_secs_f64().max(0.001)
            );
        }
    }
    let _ = w.flush();
    eprintln!(
        "datagen done {} positions in {:.1}s -> {}",
        written,
        start.elapsed().as_secs_f64(),
        out_path
    );
}

/// Dump HCE eval for FENs on stdin (one per line). Used to distill PeSTO.
pub fn dump_hce() {
    use std::io::{self, BufRead};
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let fen = line.trim();
        if fen.is_empty() {
            continue;
        }
        if let Some(pos) = Position::from_fen(fen) {
            println!("{} | {}", fen, crate::eval::evaluate_hce(&pos));
        }
    }
}
