//! Lockless transposition table (XOR-key verification, Stockfish-style).
//! Shared by Lazy SMP threads without a mutex.

use crate::types::{Move, INF};
use std::sync::atomic::{AtomicU64, Ordering};

pub const BOUND_NONE: u8 = 0;
pub const BOUND_LOWER: u8 = 1; // fail-high / beta
pub const BOUND_UPPER: u8 = 2; // fail-low / alpha
pub const BOUND_EXACT: u8 = 3;

const PACK_MOVE: u64 = 0xFFFF;
const PACK_DEPTH: u64 = 0xFF;
const PACK_BOUND: u64 = 0x3;
const PACK_AGE: u64 = 0x3F;

/// data layout: move 16 | score 16 | depth 8 | bound 2 | age 6 | unused 16
#[inline]
fn pack(mv: Move, score: i32, depth: i32, bound: u8, age: u8) -> u64 {
    let s = (score as i16) as u16 as u64;
    (mv.0 as u64)
        | (s << 16)
        | ((depth.clamp(0, 255) as u64) << 32)
        | ((bound as u64) << 40)
        | ((age as u64) << 42)
}

#[inline]
fn unpack(data: u64) -> (Move, i32, i32, u8, u8) {
    let mv = Move((data & PACK_MOVE) as u16);
    let score = ((data >> 16) as u16) as i16 as i32;
    let depth = ((data >> 32) & PACK_DEPTH) as i32;
    let bound = ((data >> 40) & PACK_BOUND) as u8;
    let age = ((data >> 42) & PACK_AGE) as u8;
    (mv, score, depth, bound, age)
}

struct Entry {
    key: AtomicU64,
    data: AtomicU64,
}

impl Entry {
    fn empty() -> Entry {
        Entry {
            key: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

pub struct TranspositionTable {
    table: Vec<Entry>,
    mask: u64,
    age: std::sync::atomic::AtomicU8,
}

impl TranspositionTable {
    pub fn new(mb: usize) -> TranspositionTable {
        let bytes = mb.max(1) * 1024 * 1024;
        let mut n = 1usize;
        while n * std::mem::size_of::<Entry>() * 2 < bytes {
            n *= 2;
        }
        n = n.max(1024);
        let mut table = Vec::with_capacity(n);
        for _ in 0..n {
            table.push(Entry::empty());
        }
        TranspositionTable {
            table,
            mask: (n - 1) as u64,
            age: std::sync::atomic::AtomicU8::new(1),
        }
    }

    pub fn resize(&mut self, mb: usize) {
        *self = TranspositionTable::new(mb);
    }

    pub fn clear(&self) {
        for e in &self.table {
            e.key.store(0, Ordering::Relaxed);
            e.data.store(0, Ordering::Relaxed);
        }
    }

    pub fn new_search(&self) {
        self.age.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    fn idx(&self, hash: u64) -> usize {
        (hash & self.mask) as usize
    }

    /// Hint the CPU to pull this hash's slot into cache. Called right after a
    /// move is made so the DRAM latency overlaps with the child node's setup
    /// (a pure performance hint: it never changes search results).
    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let p = self.table.as_ptr().add(self.idx(hash)) as *const i8;
            _mm_prefetch(p, _MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = hash;
    }

    pub fn probe(&self, hash: u64) -> Option<(Move, i32, i32, u8)> {
        let e = &self.table[self.idx(hash)];
        let key = e.key.load(Ordering::Relaxed);
        let data = e.data.load(Ordering::Relaxed);
        if key ^ data == hash {
            let (mv, score, depth, bound, _) = unpack(data);
            Some((mv, score, depth, bound))
        } else {
            None
        }
    }

    pub fn store(&self, hash: u64, mv: Move, mut score: i32, depth: i32, bound: u8) {
        score = score.clamp(-INF + 1, INF - 1);
        let idx = self.idx(hash);
        let e = &self.table[idx];
        let old_data = e.data.load(Ordering::Relaxed);
        let old_key = e.key.load(Ordering::Relaxed);
        let age = self.age.load(Ordering::Relaxed);
        let replace = if old_key ^ old_data == hash {
            let (_, _, old_depth, _, old_age) = unpack(old_data);
            depth + 2 >= old_depth || old_age != age
        } else {
            true
        };
        if replace {
            let data = pack(mv, score, depth, bound, age);
            e.data.store(data, Ordering::Relaxed);
            e.key.store(hash ^ data, Ordering::Relaxed);
        }
    }
}

/// Adjust mate scores so they stay relative to the root when stored in the TT.
pub fn to_tt(score: i32, ply: i32) -> i32 {
    if score >= crate::types::MATE_IN_MAX {
        score + ply
    } else if score <= -crate::types::MATE_IN_MAX {
        score - ply
    } else {
        score
    }
}

pub fn from_tt(score: i32, ply: i32) -> i32 {
    if score >= crate::types::MATE_IN_MAX {
        score - ply
    } else if score <= -crate::types::MATE_IN_MAX {
        score + ply
    } else {
        score
    }
}
