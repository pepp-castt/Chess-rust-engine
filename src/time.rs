//! Soft/hard time bounds, matching the Reckless / Stockfish time model:
//! remaining/20 + increment*0.8, with move-overhead and stability hooks.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: u64,
    pub binc: u64,
    pub movestogo: Option<u32>,
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    pub movetime: Option<u64>,
    pub infinite: bool,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits {
            wtime: None,
            btime: None,
            winc: 0,
            binc: 0,
            movestogo: None,
            depth: None,
            nodes: None,
            movetime: None,
            infinite: false,
        }
    }
}

pub struct TimeManager {
    pub start: Instant,
    pub soft: Duration,
    pub hard: Duration,
    pub max_depth: i32,
    pub max_nodes: u64,
    pub infinite: bool,
    pub stop: AtomicBool,
    pub overhead_ms: u64,
}

impl TimeManager {
    pub fn new(limits: Limits, white: bool, overhead_ms: u64) -> TimeManager {
        let start = Instant::now();
        let mut soft = Duration::from_secs(60 * 60);
        let mut hard = Duration::from_secs(60 * 60);
        let mut infinite = limits.infinite;
        let max_depth = limits.depth.unwrap_or(128);
        let max_nodes = limits.nodes.unwrap_or(u64::MAX);

        if let Some(mt) = limits.movetime {
            let t = mt.saturating_sub(overhead_ms);
            soft = Duration::from_millis(t);
            hard = Duration::from_millis(t);
        } else {
            let (time, inc) = if white {
                (limits.wtime, limits.winc)
            } else {
                (limits.btime, limits.binc)
            };
            if let Some(remain) = time {
                let mtg = limits.movestogo.unwrap_or(20).max(1) as u64;
                // Spend a bit more of the clock than remain/20 — 10+0.1 games
                // were finishing with time left while SF19 still had a vote.
                let alloc = remain / mtg.max(15) + inc * 9 / 10;
                let alloc = alloc.min(remain / 2).saturating_sub(overhead_ms);
                let alloc = alloc.max(5);
                soft = Duration::from_millis(alloc);
                let hard_ms = (alloc.saturating_mul(4))
                    .min(remain.saturating_sub(overhead_ms + 80))
                    .max(10);
                hard = Duration::from_millis(hard_ms);
            } else if limits.depth.is_none() && limits.nodes.is_none() && !limits.infinite {
                infinite = true;
            }
        }

        TimeManager {
            start,
            soft,
            hard,
            max_depth,
            max_nodes,
            infinite,
            stop: AtomicBool::new(false),
            overhead_ms,
        }
    }

    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    #[inline]
    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    pub fn hard_expired(&self) -> bool {
        !self.infinite && self.elapsed() >= self.hard
    }

    /// True when the budget came from a game clock (wtime/btime). With
    /// `movetime`, soft == hard and must never be scaled.
    #[inline]
    pub fn is_clocked(&self) -> bool {
        !self.infinite && self.soft < self.hard
    }

    /// Stop-between-iterations test. `scale` multiplies the soft budget
    /// (clamped to the hard budget); ignored for fixed movetime.
    pub fn soft_expired_scaled(&self, scale: f64) -> bool {
        if self.infinite {
            return false;
        }
        let e = self.elapsed();
        if !self.is_clocked() {
            return e >= self.soft;
        }
        let limit = self.soft.mul_f64(scale.clamp(0.4, 3.0)).min(self.hard / 2);
        e >= limit
    }
}
