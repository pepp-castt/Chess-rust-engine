//! Tapered PeSTO evaluation plus cheap positional terms.
//!
//! PeSTO (Ronald Friederich / RofChade) is the standard HCE starting point
//! used by dozens of engines. Reckless, Viridithas and every elite engine later
//! replaced HCE with self-play NNUE — that is the single largest Elo jump and
//! is documented in ROADMAP.md. This function is the correct base to Texel-tune
//! and then to generate NNUE training data from.

use crate::attacks::{
    bishop_attacks, king_attacks, knight_attacks, pawn_attacks, pawn_attacks_bb, queen_attacks,
    rook_attacks, tables,
};
use crate::bitboard::{ADJACENT_FILES, FILES};
use crate::board::Position;
use crate::types::{Color, PieceType, Square};

const MG_VALUE: [i32; 6] = [82, 337, 365, 477, 1025, 0];
const EG_VALUE: [i32; 6] = [94, 281, 297, 512, 936, 0];
pub const SEE_VALUE: [i32; 6] = [100, 300, 300, 500, 900, 20_000];
const PHASE_INC: [i32; 6] = [0, 1, 1, 2, 4, 0];

// PeSTO tables are printed with a8 at index 0 (diagram order).
// LERF white uses sq^56, black uses sq. See RESEARCH.md.
const MG_PAWN: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 98, 134, 61, 95, 68, 126, 34, -11, -6, 7, 26, 31, 65, 56, 25, -20,
    -14, 13, 6, 21, 23, 12, 17, -23, -27, -2, -5, 12, 17, 6, 10, -25, -26, -4, -4, -10, 3, 3, 33,
    -12, -35, -1, -20, -23, -15, 24, 38, -22, 0, 0, 0, 0, 0, 0, 0, 0,
];
const EG_PAWN: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85, 67, 56, 53, 82,
    84, 32, 24, 13, 5, -2, 4, 17, 17, 13, 9, -3, -7, -7, -8, 3, -1, 4, 7, -6, 1, 0, -5, -1, -8, 13,
    8, 8, 10, 13, 0, 2, -7, 0, 0, 0, 0, 0, 0, 0, 0,
];
const MG_KNIGHT: [i32; 64] = [
    -167, -89, -34, -49, 61, -97, -15, -107, -73, -41, 72, 36, 23, 62, 7, -17, -47, 60, 37, 65, 84,
    129, 73, 44, -9, 17, 19, 53, 37, 69, 18, 22, -13, 4, 16, 13, 28, 19, 21, -8, -23, -9, 12, 10,
    19, 17, 25, -16, -29, -53, -12, -3, -1, 18, -14, -19, -105, -21, -58, -33, -17, -28, -19, -23,
];
const EG_KNIGHT: [i32; 64] = [
    -58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52, -24, -20, 10, 9,
    -1, -9, -19, -41, -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16, 25, 16, 17, 4, -18, -23, -3, -1,
    15, 10, -3, -20, -22, -42, -20, -10, -5, -2, -20, -23, -44, -29, -51, -23, -15, -22, -18, -50,
    -64,
];
const MG_BISHOP: [i32; 64] = [
    -29, 4, -82, -37, -25, -42, 7, -8, -26, 16, -18, -13, 30, 59, 18, -47, -16, 37, 43, 40, 35, 50,
    37, -2, -4, 5, 19, 50, 37, 37, 7, -2, -6, 13, 13, 26, 34, 12, 10, 4, 0, 15, 15, 15, 14, 27, 18,
    10, 4, 15, 16, 0, 7, 21, 33, 1, -33, -3, -14, -21, -13, -12, -39, -21,
];
const EG_BISHOP: [i32; 64] = [
    -14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -3, -13, -4, -14, 2, -8, 0, -1, -2, 6, 0,
    4, -3, 9, 12, 9, 14, 10, 3, 2, -6, 3, 13, 19, 7, 10, -3, -9, -12, -3, 8, 10, 13, 3, -7, -15,
    -14, -18, -7, -1, 4, -9, -15, -27, -23, -9, -23, -5, -9, -16, -5, -17,
];
const MG_ROOK: [i32; 64] = [
    32, 42, 32, 51, 63, 9, 31, 43, 27, 32, 58, 62, 80, 67, 26, 44, -5, 19, 26, 36, 17, 45, 61, 16,
    -24, -11, 7, 26, 24, 35, -8, -20, -36, -26, -12, -1, 9, -7, 6, -23, -45, -25, -16, -17, 3, 0,
    -5, -33, -44, -16, -20, -9, -1, 11, -6, -71, -19, -13, 1, 17, 16, 7, -37, -26,
];
const EG_ROOK: [i32; 64] = [
    13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3, -5, -3, 4, 3, 13,
    1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1, -7, -12, -8, -16, -6, -6, 0, 2, -9,
    -9, -11, -3, -9, 2, 3, -1, -5, -13, 4, -20,
];
const MG_QUEEN: [i32; 64] = [
    -28, 0, 29, 12, 59, 44, 43, 45, -24, -39, -5, 1, -16, 57, 28, 54, -13, -17, 7, 8, 29, 56, 47,
    57, -27, -27, -16, -16, -1, 17, -2, 1, -9, -26, -9, -10, -2, -4, 3, -3, -14, 2, -11, -2, -5, 2,
    14, 5, -35, -8, 11, 2, 8, 15, -3, 1, -1, -18, -9, 10, -15, -25, -31, -50,
];
const EG_QUEEN: [i32; 64] = [
    -9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49, 47, 35, 19, 9,
    3, 22, 24, 45, 57, 40, 57, 36, -18, 28, 19, 47, 31, 34, 39, 23, -16, -27, 15, 6, 9, 17, 10, 5,
    -22, -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41,
];
const MG_KING: [i32; 64] = [
    -65, 23, 16, -15, -56, -34, 2, 13, 29, -1, -20, -7, -8, -4, -38, -29, -9, 24, 2, -16, -20, 6,
    22, -22, -17, -20, -12, -27, -30, -25, -14, -36, -49, -1, -27, -39, -46, -44, -33, -51, -14,
    -14, -22, -46, -44, -30, -15, -27, 1, 7, -8, -64, -43, -16, 9, 8, -15, 36, 12, -54, 8, -28, 24,
    14,
];
const EG_KING: [i32; 64] = [
    -74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17, 23, 15, 20, 45,
    44, 13, -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23, 9, -11, -19, -3, 11, 21, 23, 16,
    7, -9, -27, -11, 4, 13, 14, 4, -5, -17, -53, -34, -21, -11, -28, -14, -24, -43,
];

const MG_PST: [&[i32; 64]; 6] = [
    &MG_PAWN, &MG_KNIGHT, &MG_BISHOP, &MG_ROOK, &MG_QUEEN, &MG_KING,
];
const EG_PST: [&[i32; 64]; 6] = [
    &EG_PAWN, &EG_KNIGHT, &EG_BISHOP, &EG_ROOK, &EG_QUEEN, &EG_KING,
];

const BISHOP_PAIR_MG: i32 = 28;
const BISHOP_PAIR_EG: i32 = 42;
const DOUBLED_MG: i32 = -12;
const DOUBLED_EG: i32 = -20;
const ISOLATED_MG: i32 = -8;
const ISOLATED_EG: i32 = -12;
const ROOK_OPEN_MG: i32 = 18;
const ROOK_OPEN_EG: i32 = 10;
const ROOK_SEMI_MG: i32 = 8;
const ROOK_SEMI_EG: i32 = 4;
const ROOK_SEVENTH_MG: i32 = 12;
const ROOK_SEVENTH_EG: i32 = 22;
const TEMPO: i32 = 10;
const OUTPOST_MG: i32 = 18;
const OUTPOST_EG: i32 = 10;
const KING_ATT_W: [i32; 6] = [0, 2, 2, 3, 5, 0];

const PASSED_MG: [i32; 8] = [0, 2, 6, 12, 24, 48, 80, 0];
const PASSED_EG: [i32; 8] = [0, 8, 14, 28, 52, 90, 140, 0];

#[inline]
fn pst_index(color: Color, sq: Square) -> usize {
    if color == Color::White {
        (sq.0 ^ 56) as usize
    } else {
        sq.index()
    }
}

/// Evaluation from the side to move, in centipawns: NNUE when a net is loaded and
/// `UseNNUE` is on, otherwise the hand-crafted evaluation below.
#[inline]
pub fn evaluate(pos: &Position) -> i32 {
    if crate::nnue::enabled() {
        if let Some(n) = crate::nnue::net() {
            let v = n.eval(pos);
            if n.residual {
                // HCE-anchored correction net: HCE + clamp(net). CRE_DELTA_CAP bounds the damage.
                let cap = delta_cap();
                return evaluate_hce(pos) + v.clamp(-cap, cap);
            }
            return v;
        }
    }
    evaluate_hce(pos)
}

static DELTA_CAP: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// Max |correction| (cp) a residual (version 2) net may add to HCE. UCI `ResidualCap`
/// overrides; `CRE_DELTA_CAP` is the initial value; default 400.
pub fn delta_cap() -> i32 {
    use std::sync::atomic::Ordering::Relaxed;
    let v = DELTA_CAP.load(Relaxed);
    if v >= 0 {
        return v;
    }
    let init = std::env::var("CRE_DELTA_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
    DELTA_CAP.store(init, Relaxed);
    init
}

pub fn set_delta_cap(cap: i32) {
    DELTA_CAP.store(cap.clamp(0, 3000), std::sync::atomic::Ordering::Relaxed);
}

/// Hand-crafted eval (PeSTO + structure). Used when no NNUE net is loaded
/// and as the teacher for the first net.
pub fn evaluate_hce(pos: &Position) -> i32 {
    let mut mg = [0i32; 2];
    let mut eg = [0i32; 2];
    let mut phase = 0i32;
    let occ = pos.occ();
    let t = tables();

    for ty in 0..6 {
        let pt = PieceType::from_index(ty);
        for color_i in 0..2 {
            let color = Color::from_index(color_i);
            let mut bb = pos.bb(color, pt);
            while bb != 0 {
                let sq = Square(bb.trailing_zeros() as u8);
                bb &= bb - 1;
                let idx = pst_index(color, sq);
                mg[color_i] += MG_VALUE[ty] + MG_PST[ty][idx];
                eg[color_i] += EG_VALUE[ty] + EG_PST[ty][idx];
                phase += PHASE_INC[ty];
            }
        }
    }

    for color_i in 0..2 {
        let color = Color::from_index(color_i);
        let pawns = pos.bb(color, PieceType::Pawn);
        let enemy_pawns = pos.bb(color.flip(), PieceType::Pawn);

        // Bishop pair
        if pos.bb(color, PieceType::Bishop).count_ones() >= 2 {
            mg[color_i] += BISHOP_PAIR_MG;
            eg[color_i] += BISHOP_PAIR_EG;
        }

        // Doubled / isolated / passed pawns
        let mut p = pawns;
        while p != 0 {
            let sq = Square(p.trailing_zeros() as u8);
            p &= p - 1;
            let file = sq.file() as usize;
            let ours_file = pawns & FILES[file];
            if ours_file.count_ones() > 1 && (sq.bb() & ours_file) != 0 {
                // penalize every pawn that has a friendly pawn behind it on the file
                let behind = if color == Color::White {
                    t.forward_file[Color::Black.index()][sq.index()]
                } else {
                    t.forward_file[Color::White.index()][sq.index()]
                };
                if ours_file & behind != 0 {
                    mg[color_i] += DOUBLED_MG;
                    eg[color_i] += DOUBLED_EG;
                }
            }
            if pawns & ADJACENT_FILES[file] == 0 {
                mg[color_i] += ISOLATED_MG;
                eg[color_i] += ISOLATED_EG;
            }
            if enemy_pawns & t.passed[color.index()][sq.index()] == 0 {
                let rel_rank = if color == Color::White {
                    sq.rank()
                } else {
                    7 - sq.rank()
                } as usize;
                mg[color_i] += PASSED_MG[rel_rank];
                eg[color_i] += PASSED_EG[rel_rank];
                // Connected / protected passer
                if pawns & ADJACENT_FILES[file] != 0 {
                    mg[color_i] += PASSED_MG[rel_rank] / 2;
                    eg[color_i] += PASSED_EG[rel_rank] / 2;
                }
            }
        }

        // Rooks on open / semi-open files and on the 7th
        let mut rooks = pos.bb(color, PieceType::Rook);
        while rooks != 0 {
            let sq = Square(rooks.trailing_zeros() as u8);
            rooks &= rooks - 1;
            let file = sq.file() as usize;
            let ours = pawns & FILES[file];
            let theirs = enemy_pawns & FILES[file];
            if ours == 0 && theirs == 0 {
                mg[color_i] += ROOK_OPEN_MG;
                eg[color_i] += ROOK_OPEN_EG;
            } else if ours == 0 {
                mg[color_i] += ROOK_SEMI_MG;
                eg[color_i] += ROOK_SEMI_EG;
            }
            let seventh = if color == Color::White { 6 } else { 1 };
            if sq.rank() == seventh {
                mg[color_i] += ROOK_SEVENTH_MG;
                eg[color_i] += ROOK_SEVENTH_EG;
            }
        }

        // Very cheap mobility (safe squares, not occupied by us)
        let us_occ = pos.colors[color_i];
        let mut kn = pos.bb(color, PieceType::Knight);
        while kn != 0 {
            let sq = Square(kn.trailing_zeros() as u8);
            kn &= kn - 1;
            let mob = (knight_attacks(sq) & !us_occ).count_ones() as i32;
            mg[color_i] += (mob - 4) * 3;
            eg[color_i] += (mob - 4) * 2;
        }
        let mut b = pos.bb(color, PieceType::Bishop);
        while b != 0 {
            let sq = Square(b.trailing_zeros() as u8);
            b &= b - 1;
            let mob = (bishop_attacks(sq, occ) & !us_occ).count_ones() as i32;
            mg[color_i] += (mob - 5) * 3;
            eg[color_i] += (mob - 5) * 2;
        }
        let mut r = pos.bb(color, PieceType::Rook);
        while r != 0 {
            let sq = Square(r.trailing_zeros() as u8);
            r &= r - 1;
            let mob = (rook_attacks(sq, occ) & !us_occ).count_ones() as i32;
            mg[color_i] += (mob - 5) * 2;
            eg[color_i] += (mob - 5) * 3;
        }
        let mut q = pos.bb(color, PieceType::Queen);
        while q != 0 {
            let sq = Square(q.trailing_zeros() as u8);
            q &= q - 1;
            let mob = (queen_attacks(sq, occ) & !us_occ).count_ones() as i32;
            mg[color_i] += (mob - 8) * 1;
            eg[color_i] += (mob - 8) * 2;
        }

        // King pawn shield (opening only)
        let king = pos.king_sq(color);
        let shield_rank = if color == Color::White {
            king.rank().saturating_add(1).min(7)
        } else {
            king.rank().saturating_sub(1)
        };
        let mut shield = 0i32;
        for df in [-1i32, 0, 1] {
            let f = king.file() as i32 + df;
            if (0..8).contains(&f) {
                let sq = Square::from_file_rank(f as u8, shield_rank);
                if pawns & sq.bb() != 0 {
                    shield += 1;
                }
            }
        }
        mg[color_i] += (shield - 2) * 10;

        // Pawn storm against the enemy king (opening/middlegame)
        let ek = pos.king_sq(color.flip());
        for df in [-1i32, 0, 1] {
            let f = ek.file() as i32 + df;
            if !(0..8).contains(&f) {
                continue;
            }
            let mut fp = pawns & FILES[f as usize];
            while fp != 0 {
                let sq = Square(fp.trailing_zeros() as u8);
                fp &= fp - 1;
                let rel = if color == Color::White {
                    sq.rank()
                } else {
                    7 - sq.rank()
                } as i32;
                if rel >= 4 {
                    mg[color_i] += 5 * (rel - 3);
                }
            }
        }

        // Knight outposts: supported by a pawn, not attackable by an enemy pawn,
        // on the 5th/6th (relative).
        let mut kn_out = pos.bb(color, PieceType::Knight);
        while kn_out != 0 {
            let sq = Square(kn_out.trailing_zeros() as u8);
            kn_out &= kn_out - 1;
            let rel = if color == Color::White {
                sq.rank()
            } else {
                7 - sq.rank()
            };
            if rel >= 4 {
                let supported = pawn_attacks(color.flip(), sq) & pawns != 0;
                let attackable = pawn_attacks(color, sq) & enemy_pawns != 0;
                if supported && !attackable {
                    mg[color_i] += OUTPOST_MG;
                    eg[color_i] += OUTPOST_EG;
                }
            }
        }

        // King safety: weighted enemy attacks on the king zone.
        let zone = king_attacks(king) | king.bb();
        let enemy = color.flip();
        let mut units = 0i32;
        let mut n_att = 0i32;
        for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
            let mut bb = pos.bb(enemy, pt);
            while bb != 0 {
                let sq = Square(bb.trailing_zeros() as u8);
                bb &= bb - 1;
                let att = match pt {
                    PieceType::Knight => knight_attacks(sq),
                    PieceType::Bishop => bishop_attacks(sq, occ),
                    PieceType::Rook => rook_attacks(sq, occ),
                    PieceType::Queen => queen_attacks(sq, occ),
                    _ => 0,
                };
                if att & zone != 0 {
                    units += KING_ATT_W[pt.index()];
                    n_att += 1;
                }
            }
        }
        if pawn_attacks_bb(enemy, pos.bb(enemy, PieceType::Pawn)) & zone != 0 {
            units += 1;
            n_att += 1;
        }
        if n_att >= 2 {
            mg[color_i] -= (units * units) / 2;
        }
    }

    let us = pos.side.index();
    let them = pos.side.flip().index();
    let mg_score = mg[us] - mg[them];
    let mut eg_score = eg[us] - eg[them];

    // Drive the enemy king to the edge in K+major vs K endings so we actually
    // convert instead of hitting the 200-ply cap.
    let piece_count = occ.count_ones();
    if piece_count <= 4 {
        let our_k = pos.king_sq(pos.side);
        let their_k = pos.king_sq(pos.side.flip());
        let tf = their_k.file() as i32;
        let tr = their_k.rank() as i32;
        let edge = tf.min(7 - tf).min(tr).min(7 - tr);
        let cd = (our_k.file() as i32 - tf).abs().max((our_k.rank() as i32 - tr).abs());
        let majors = pos.bb(pos.side, PieceType::Queen) | pos.bb(pos.side, PieceType::Rook);
        let their_majors = pos.bb(pos.side.flip(), PieceType::Queen) | pos.bb(pos.side.flip(), PieceType::Rook);
        if majors != 0 && their_majors == 0 {
            eg_score += (3 - edge) * 18 + (14 - cd) * 6;
        } else if their_majors != 0 && majors == 0 {
            eg_score -= (3 - edge) * 18 + (14 - cd) * 6;
        }
    }

    let mg_phase = phase.min(24);
    let eg_phase = 24 - mg_phase;
    (mg_score * mg_phase + eg_score * eg_phase) / 24 + TEMPO
}

pub fn material_total(pos: &Position) -> i32 {
    let mut m = 0;
    for ty in 0..5 {
        m += pos.pieces[ty].count_ones() as i32 * SEE_VALUE[ty];
    }
    m
}
