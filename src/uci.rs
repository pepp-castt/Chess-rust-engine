//! Universal Chess Interface. Reckless / Viridithas / Stockfish all speak UCI
//! over stdin/stdout; GUIs (Cute Chess, Fastchess, En Croissant) drive us.

use crate::board::{parse_uci_move, Position, START_FEN};
use crate::eval::evaluate;
use crate::perft::{divide, perft};
use crate::search::{bench, iterative_deepening, ThreadData};
use crate::time::{Limits, TimeManager};
use crate::tt::TranspositionTable;
use crate::types::Color;
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;

pub const ENGINE_NAME: &str = "Chess-rust-engine";
pub const ENGINE_AUTHOR: &str = "Chess-rust-engine authors";
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub struct Engine {
    pos: Position,
    moves_played: Vec<u64>,
    tt: Arc<TranspositionTable>,
    hash_mb: usize,
    overhead_ms: u64,
}

impl Engine {
    pub fn new() -> Engine {
        Engine {
            pos: Position::startpos(),
            moves_played: Vec::new(),
            tt: Arc::new(TranspositionTable::new(16)),
            hash_mb: 16,
            overhead_ms: 20,
        }
    }

    fn reset_hist(&mut self) {
        self.moves_played.clear();
        self.moves_played.push(self.pos.hash);
    }

    fn apply_uci_moves(&mut self, moves: &[&str]) {
        for tok in moves {
            if let Some(mv) = parse_uci_move(&self.pos, tok) {
                self.pos.make(mv);
                self.moves_played.push(self.pos.hash);
            }
        }
    }
}

pub fn run() {
    let mut engine = Engine::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    // Pre-init attack tables so `isready` is instant.
    let _ = crate::attacks::tables();
    let mut search_job: Option<(Arc<TimeManager>, JoinHandle<()>)> = None;

    fn wait_search(job: &mut Option<(Arc<TimeManager>, JoinHandle<()>)>) {
        if let Some((tm, handle)) = job.take() {
            tm.request_stop();
            let _ = handle.join();
        }
    }

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(cmd) = parts.next() else { continue };
        match cmd {
            "uci" => {
                println!("id name {} {}", ENGINE_NAME, ENGINE_VERSION);
                println!("id author {}", ENGINE_AUTHOR);
                println!("option name Hash type spin default 16 min 1 max 65536");
                println!("option name Threads type spin default 1 min 1 max 1");
                println!("option name MoveOverhead type spin default 20 min 0 max 5000");
                println!("option name Clear Hash type button");
                println!("option name UseNNUE type check default true");
                println!("uciok");
            }
            "isready" => {
                let _ = crate::attacks::tables();
                println!("readyok");
            }
            "ucinewgame" => {
                wait_search(&mut search_job);
                engine.tt.clear();
                engine.pos = Position::startpos();
                engine.reset_hist();
            }
            "setoption" => {
                let toks: Vec<&str> = line.split_whitespace().collect();
                // setoption name Hash value 128
                if let Some(name_i) = toks.iter().position(|&t| t == "name") {
                    let value_i = toks.iter().position(|&t| t == "value");
                    let name = if let Some(v) = value_i {
                        toks[name_i + 1..v].join(" ")
                    } else {
                        toks[name_i + 1..].join(" ")
                    };
                    let value = value_i.map(|v| toks[v + 1..].join(" "));
                    match name.as_str() {
                        "Hash" => {
                            if let Some(v) = value.and_then(|s| s.parse().ok()) {
                                engine.hash_mb = v;
                                engine.tt = Arc::new(TranspositionTable::new(v));
                            }
                        }
                        "MoveOverhead" => {
                            if let Some(v) = value.and_then(|s| s.parse().ok()) {
                                engine.overhead_ms = v;
                            }
                        }
                        "Clear Hash" => engine.tt.clear(),
                        "UseNNUE" => {
                            if let Some(v) = value {
                                crate::nnue::set_enabled(v.eq_ignore_ascii_case("true"));
                            }
                        }
                        _ => {}
                    }
                }
            }
            "position" => {
                wait_search(&mut search_job);
                let rest: Vec<&str> = line.split_whitespace().skip(1).collect();
                if rest.is_empty() {
                    continue;
                }
                let (fen, moves_at) = if rest[0] == "startpos" {
                    let m = rest.iter().position(|&t| t == "moves");
                    (START_FEN.to_string(), m)
                } else if rest[0] == "fen" {
                    let m = rest.iter().position(|&t| t == "moves");
                    let end = m.unwrap_or(rest.len());
                    (rest[1..end].join(" "), m)
                } else {
                    continue;
                };
                if let Some(p) = Position::from_fen(&fen) {
                    engine.pos = p;
                    engine.reset_hist();
                    if let Some(mi) = moves_at {
                        engine.apply_uci_moves(&rest[mi + 1..]);
                    }
                }
            }
            "go" => {
                wait_search(&mut search_job);
                let rest: Vec<&str> = line.split_whitespace().skip(1).collect();
                let mut limits = Limits::default();
                let mut i = 0;
                while i < rest.len() {
                    match rest[i] {
                        "wtime" if i + 1 < rest.len() => {
                            limits.wtime = rest[i + 1].parse().ok();
                            i += 2;
                        }
                        "btime" if i + 1 < rest.len() => {
                            limits.btime = rest[i + 1].parse().ok();
                            i += 2;
                        }
                        "winc" if i + 1 < rest.len() => {
                            limits.winc = rest[i + 1].parse().unwrap_or(0);
                            i += 2;
                        }
                        "binc" if i + 1 < rest.len() => {
                            limits.binc = rest[i + 1].parse().unwrap_or(0);
                            i += 2;
                        }
                        "movestogo" if i + 1 < rest.len() => {
                            limits.movestogo = rest[i + 1].parse().ok();
                            i += 2;
                        }
                        "depth" if i + 1 < rest.len() => {
                            limits.depth = rest[i + 1].parse().ok();
                            i += 2;
                        }
                        "nodes" if i + 1 < rest.len() => {
                            limits.nodes = rest[i + 1].parse().ok();
                            i += 2;
                        }
                        "movetime" if i + 1 < rest.len() => {
                            limits.movetime = rest[i + 1].parse().ok();
                            i += 2;
                        }
                        "infinite" => {
                            limits.infinite = true;
                            i += 1;
                        }
                        "perft" if i + 1 < rest.len() => {
                            let d: u32 = rest[i + 1].parse().unwrap_or(1);
                            let start = std::time::Instant::now();
                            let n = perft(&engine.pos, d);
                            let ms = start.elapsed().as_millis().max(1);
                            println!();
                            println!(
                                "info string perft {} = {} ({:.3} Mnps)",
                                d,
                                n,
                                n as f64 / ms as f64 / 1000.0
                            );
                            i = rest.len();
                            limits.depth = Some(0);
                        }
                        _ => i += 1,
                    }
                }
                if limits.depth == Some(0) {
                    let _ = stdout.flush();
                    continue;
                }
                let use_book = limits.wtime.is_some()
                    || limits.btime.is_some()
                    || limits.movetime.is_some()
                    || limits.infinite;
                let tm = Arc::new(TimeManager::new(
                    limits,
                    engine.pos.side == Color::White,
                    engine.overhead_ms,
                ));
                let mut td = ThreadData::new(engine.tt.clone(), tm.clone());
                td.hash_hist = engine.moves_played.clone();
                let pos = engine.pos;
                let tm_for_job = tm.clone();
                let handle = thread::spawn(move || {
                    let res = iterative_deepening(&pos, &mut td, use_book);
                    println!("bestmove {}", res.best);
                    let _ = io::stdout().flush();
                });
                search_job = Some((tm_for_job, handle));
            }
            "stop" => {
                if let Some((tm, _)) = &search_job {
                    tm.request_stop();
                }
            }
            "quit" => {
                wait_search(&mut search_job);
                break;
            }
            "d" => print!("{}", engine.pos.display()),
            "eval" => {
                let e = evaluate(&engine.pos);
                println!("info string eval {} cp (side to move)", e);
                println!(
                    "info string eval_hce {} eval_nnue {}",
                    crate::eval::evaluate_hce(&engine.pos),
                    crate::nnue::evaluate(&engine.pos).map_or("none".to_string(), |v| v.to_string())
                );
            }
            "perft" => {
                let d: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
                let start = std::time::Instant::now();
                let n = perft(&engine.pos, d);
                println!("info string perft {} nodes {} time {} ms", d, n, start.elapsed().as_millis());
            }
            "divide" => {
                let d: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
                divide(&engine.pos, d);
            }
            "bench" => {
                let d: i32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(8);
                let _ = bench(d);
            }
            "compiler" => {
                println!(
                    "info string rustc {} target {} opt-level 3 lto",
                    option_env!("RUSTC_VERSION").unwrap_or("unknown"),
                    std::env::consts::ARCH
                );
            }
            _ => {}
        }
        let _ = stdout.flush();
    }
    wait_search(&mut search_job);
}

/// `stop` while searching requires a non-blocking stdin. Spawn a helper that
/// only watches for stop/quit when we later add Lazy SMP. Kept for the API.
#[allow(dead_code)]
fn spawn_stop_watcher(tm: Arc<TimeManager>) {
    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            match line.trim() {
                "stop" => tm.request_stop(),
                "quit" => {
                    tm.request_stop();
                    break;
                }
                _ => {}
            }
        }
    });
}
