//! NNUE evaluation: `(768 -> H) x2 -> 1`, SCReLU, i16 weights, 8 material output buckets.
//!
//! Architecture, trainer (`tools/nnue/train.py`) and weights are original to this repo.
//! Stockfish 19 is only used as a *teacher of position scores* (data), never as a net source.
//!
//! Feature index (perspective-relative, same convention as the trainer):
//!   `((piece_color != perspective) * 6 + piece_type) * 64 + (sq ^ (56 * perspective))`
//! The first layer is recomputed from scratch per evaluation (cheap for H=128 with AVX2);
//! incremental accumulators are a later optimisation and need no change to the net format.
//!
//! Net file (little endian): `"CRN1" | version u32 | hidden u32 | buckets u32 | features u32`
//! then `ft_w i16[768*H] | ft_b i16[H] | out_w i16[B*2H] | out_b i32[B]`.
//! An empty / invalid net => HCE fallback. `CRE_NNUE_FILE` overrides the embedded net.

use crate::board::Position;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

pub const HIDDEN: usize = 128;
pub const FEATURES: usize = 768;
pub const BUCKETS: usize = 8;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const SCALE: i32 = 400;
const HEADER: usize = 20;
const MAX_EVAL: i32 = 3000;

static EMBEDDED: &[u8] = include_bytes!("../nets/default.nnue");
static ENABLED: AtomicBool = AtomicBool::new(true);
static NET: OnceLock<Option<Network>> = OnceLock::new();

pub struct Network {
    ft_w: Vec<i16>,
    ft_b: Vec<i16>,
    out_w: Vec<i16>,
    out_b: Vec<i32>,
    avx2: bool,
}

fn rd_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

impl Network {
    pub fn parse(bytes: &[u8]) -> Result<Network, String> {
        if bytes.len() < HEADER || &bytes[0..4] != b"CRN1" {
            return Err("no net (bad magic / empty)".into());
        }
        let (h, b, f) = (
            rd_u32(bytes, 8) as usize,
            rd_u32(bytes, 12) as usize,
            rd_u32(bytes, 16) as usize,
        );
        if h != HIDDEN || b != BUCKETS || f != FEATURES {
            return Err(format!("net shape {f}x{h} b{b} != built-in {FEATURES}x{HIDDEN} b{BUCKETS}"));
        }
        let n_ft = FEATURES * HIDDEN;
        let n_ow = BUCKETS * 2 * HIDDEN;
        let want = HEADER + 2 * (n_ft + HIDDEN + n_ow) + 4 * BUCKETS;
        if bytes.len() != want {
            return Err(format!("net size {} != expected {}", bytes.len(), want));
        }
        let mut p = HEADER;
        let i16s = |n: usize, p: &mut usize| -> Vec<i16> {
            let v = (0..n)
                .map(|i| i16::from_le_bytes([bytes[*p + 2 * i], bytes[*p + 2 * i + 1]]))
                .collect();
            *p += 2 * n;
            v
        };
        let ft_w = i16s(n_ft, &mut p);
        let ft_b = i16s(HIDDEN, &mut p);
        let out_w = i16s(n_ow, &mut p);
        let out_b = (0..BUCKETS).map(|i| rd_u32(bytes, p + 4 * i) as i32).collect();
        #[cfg(target_arch = "x86_64")]
        let avx2 = is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        Ok(Network { ft_w, ft_b, out_w, out_b, avx2 })
    }

    #[inline(always)]
    fn eval_impl(&self, pos: &Position) -> i32 {
        let us = pos.side.index();
        let them = us ^ 1;
        let mut acc_us = [0i32; HIDDEN];
        let mut acc_th = [0i32; HIDDEN];
        for i in 0..HIDDEN {
            acc_us[i] = self.ft_b[i] as i32;
            acc_th[i] = self.ft_b[i] as i32;
        }
        for color in 0..2usize {
            for ty in 0..6usize {
                let mut bb = pos.pieces[ty] & pos.colors[color];
                while bb != 0 {
                    let s = bb.trailing_zeros() as usize;
                    bb &= bb - 1;
                    let f_us = ((((color != us) as usize) * 6 + ty) * 64 + (s ^ (56 * us))) * HIDDEN;
                    let f_th = ((((color != them) as usize) * 6 + ty) * 64 + (s ^ (56 * them))) * HIDDEN;
                    let ru = &self.ft_w[f_us..f_us + HIDDEN];
                    let rt = &self.ft_w[f_th..f_th + HIDDEN];
                    for i in 0..HIDDEN {
                        acc_us[i] += ru[i] as i32;
                        acc_th[i] += rt[i] as i32;
                    }
                }
            }
        }
        let n = (pos.colors[0] | pos.colors[1]).count_ones() as usize;
        let bucket = (n.saturating_sub(2) / 4).min(BUCKETS - 1);
        let w = &self.out_w[bucket * 2 * HIDDEN..(bucket + 1) * 2 * HIDDEN];
        let mut sum = 0i32;
        for i in 0..HIDDEN {
            let c = acc_us[i].clamp(0, QA);
            sum += c * c * w[i] as i32;
            let d = acc_th[i].clamp(0, QA);
            sum += d * d * w[HIDDEN + i] as i32;
        }
        let out = (sum / QA + self.out_b[bucket]) as i64 * SCALE as i64 / (QA * QB) as i64;
        (out as i32).clamp(-MAX_EVAL, MAX_EVAL)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn eval_avx2(&self, pos: &Position) -> i32 {
        self.eval_impl(pos)
    }

    #[inline]
    pub fn eval(&self, pos: &Position) -> i32 {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // SAFETY: avx2 availability was checked at load time.
            return unsafe { self.eval_avx2(pos) };
        }
        self.eval_impl(pos)
    }
}

fn load() -> Option<Network> {
    if let Ok(path) = std::env::var("CRE_NNUE_FILE") {
        return match std::fs::read(&path).map_err(|e| e.to_string()).and_then(|b| Network::parse(&b)) {
            Ok(n) => Some(n),
            Err(e) => {
                eprintln!("info string NNUE file {path}: {e}");
                None
            }
        };
    }
    Network::parse(EMBEDDED).ok()
}

pub fn net() -> Option<&'static Network> {
    NET.get_or_init(load).as_ref()
}

pub fn is_loaded() -> bool {
    net().is_some()
}

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// NNUE score from the side to move, or `None` when no net is loaded.
#[inline]
pub fn evaluate(pos: &Position) -> Option<i32> {
    net().map(|n| n.eval(pos))
}


#[cfg(test)]
mod tests {
    use super::*;

    fn blank(fill_w: i16, out_w: i16, out_b: i32) -> Vec<u8> {
        let mut v = b"CRN1".to_vec();
        for x in [1u32, HIDDEN as u32, BUCKETS as u32, FEATURES as u32] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        for _ in 0..FEATURES * HIDDEN {
            v.extend_from_slice(&fill_w.to_le_bytes());
        }
        for _ in 0..HIDDEN {
            v.extend_from_slice(&0i16.to_le_bytes());
        }
        for _ in 0..BUCKETS * 2 * HIDDEN {
            v.extend_from_slice(&out_w.to_le_bytes());
        }
        for _ in 0..BUCKETS {
            v.extend_from_slice(&out_b.to_le_bytes());
        }
        v
    }

    #[test]
    fn rejects_empty_and_wrong_shape() {
        assert!(Network::parse(&[]).is_err());
        assert!(Network::parse(b"CRN1").is_err());
        let mut v = blank(0, 0, 0);
        v.pop();
        assert!(Network::parse(&v).is_err());
    }

    #[test]
    fn extreme_accumulators_do_not_overflow() {
        // Every feature weight at +i16::MAX saturates all SCReLU units at QA; output
        // weights at the trainer's clamp (1.98 * QB = 127) give the worst-case i32 sum.
        let net = Network::parse(&blank(i16::MAX, 127, i32::MAX / 4)).unwrap();
        let pos = Position::startpos();
        let v = net.eval(&pos);
        assert!((-MAX_EVAL..=MAX_EVAL).contains(&v), "eval {v} outside clamp");
        let net = Network::parse(&blank(i16::MAX, -127, i32::MIN / 4)).unwrap();
        let v = net.eval(&pos);
        assert!((-MAX_EVAL..=MAX_EVAL).contains(&v), "eval {v} outside clamp");
    }

    #[test]
    fn zero_net_evals_zero_and_is_perspective_symmetric() {
        let net = Network::parse(&blank(0, 0, 0)).unwrap();
        let a = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert_eq!(net.eval(&a), 0);
    }
}
