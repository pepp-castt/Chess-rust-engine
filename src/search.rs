//! Fail-soft PVS with the pruning / ordering stack used by Reckless, Viridithas
//! and Stockfish. Conservative margins first; tune on OpenBench (see ROADMAP).

use crate::params;
use crate::board::Position;
use crate::eval::evaluate;
use crate::movegen::{legal_moves, noisy_moves, pseudo_legal};
use crate::see::see_ge;
use crate::time::TimeManager;
use crate::tt::{from_tt, to_tt, TranspositionTable, BOUND_EXACT, BOUND_LOWER, BOUND_UPPER};
use crate::types::{
    is_mate_score, mated_in, mate_in, Color, Move, PieceType, DRAW, INF, MAX_MOVES, MAX_PLY, MATE,
};
use std::sync::Arc;

const MVV_LVA: [[i32; 6]; 6] = [
    [105, 205, 305, 405, 505, 605],
    [104, 204, 304, 404, 504, 604],
    [103, 203, 303, 403, 503, 603],
    [102, 202, 302, 402, 502, 602],
    [101, 201, 301, 401, 501, 601],
    [100, 200, 300, 400, 500, 600],
];

pub struct ThreadData {
    pub nodes: u64,
    pub seldepth: i32,
    pub tt: Arc<TranspositionTable>,
    pub time: Arc<TimeManager>,
    pub killers: [[Move; 2]; MAX_PLY],
    pub history: [[[i16; 64]; 64]; 2],
    pub hash_hist: Vec<u64>,
    pub pv: [[Move; MAX_PLY]; MAX_PLY],
    pub pv_len: [usize; MAX_PLY],
    pub eval_stack: [i32; MAX_PLY],
    pub best: Move,
    pub score: i32,
    pub check_nodes: u64,
    pub silent: bool,
    /// capture history [moving piece 12][to 64][victim 6]
    pub cap_hist: Vec<i16>,
    /// continuation history [prev piece 13][prev to 64][piece 12][to 64]
    pub cont_hist: Vec<i16>,
    pub stack_pc: [u8; MAX_PLY],
    pub stack_to: [u8; MAX_PLY],
    pub root_depth: i32,
    pub counter: [[Move; 64]; 64],
    pub ply_move: [Move; MAX_PLY],
}

const HIST_MAX: i32 = 16384;

#[inline]
fn cont_idx(pp: usize, pt: usize, pc: usize, to: usize) -> usize {
    ((pp * 64 + pt) * 12 + pc) * 64 + to
}
#[inline]
fn cap_idx(pc: usize, to: usize, victim: usize) -> usize {
    (pc * 64 + to) * 6 + victim
}

#[inline]
fn gravity(e: &mut i16, bonus: i32) {
    let b = bonus.clamp(-HIST_MAX, HIST_MAX);
    let c = *e as i32;
    *e = (c + b - c * b.abs() / HIST_MAX) as i16;
}

impl ThreadData {
    pub fn new(tt: Arc<TranspositionTable>, time: Arc<TimeManager>) -> ThreadData {
        ThreadData {
            nodes: 0,
            seldepth: 0,
            tt,
            time,
            killers: [[Move::NONE; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            hash_hist: Vec::with_capacity(512),
            pv: [[Move::NONE; MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
            eval_stack: [0; MAX_PLY],
            best: Move::NONE,
            score: 0,
            check_nodes: 0,
            silent: false,
            cap_hist: vec![0; 12 * 64 * 6],
            cont_hist: vec![0; 13 * 64 * 12 * 64],
            stack_pc: [12; MAX_PLY],
            stack_to: [0; MAX_PLY],
            root_depth: 0,
            counter: [[Move::NONE; 64]; 64],
            ply_move: [Move::NONE; MAX_PLY],
        }
    }

    /// Keep butterfly / capture / continuation / counter history across `go`
    /// commands of one game. Stockfish and Reckless only clear those on
    /// `ucinewgame`; we were throwing them away every move. Agent N measured
    /// this independently (EXP-N1). Scratch (killers, PV, stacks, nodes) resets.
    pub fn prepare(&mut self, tt: Arc<TranspositionTable>, time: Arc<TimeManager>) {
        self.tt = tt;
        self.time = time;
        self.nodes = 0;
        self.seldepth = 0;
        self.killers = [[Move::NONE; 2]; MAX_PLY];
        self.pv = [[Move::NONE; MAX_PLY]; MAX_PLY];
        self.pv_len = [0; MAX_PLY];
        self.eval_stack = [0; MAX_PLY];
        self.best = Move::NONE;
        self.score = 0;
        self.check_nodes = 0;
        self.stack_pc = [12; MAX_PLY];
        self.stack_to = [0; MAX_PLY];
        self.root_depth = 0;
        self.ply_move = [Move::NONE; MAX_PLY];
        self.hash_hist.clear();
    }

    #[inline]
    fn should_stop(&self) -> bool {
        if self.time.stopped() {
            return true;
        }
        if self.nodes >= self.time.max_nodes {
            self.time.request_stop();
            return true;
        }
        if self.nodes & 2047 == 0 && self.time.hard_expired() {
            self.time.request_stop();
            return true;
        }
        false
    }
}

#[inline]
fn quiet_hist(td: &ThreadData, pos: &Position, mv: Move, ply: usize) -> i32 {
    let us = pos.side.index();
    let from = mv.from().index();
    let to = mv.to().index();
    let mut h = td.history[us][from][to] as i32;
    let pc = pos.piece_on(mv.from()).index();
    if pc < 12 {
        if ply >= 1 {
            h += td.cont_hist[cont_idx(td.stack_pc[ply - 1] as usize, td.stack_to[ply - 1] as usize, pc, to)] as i32;
        }
        if ply >= 2 {
            h += td.cont_hist[cont_idx(td.stack_pc[ply - 2] as usize, td.stack_to[ply - 2] as usize, pc, to)] as i32;
        }
    }
    h
}

fn score_move(td: &ThreadData, pos: &Position, mv: Move, tt_move: Move, ply: usize) -> i32 {
    if mv == tt_move {
        return 10_000_000;
    }
    if mv.is_capture() || mv.is_promo() {
        let victim = if mv.is_ep() {
            PieceType::Pawn
        } else {
            let p = pos.piece_on(mv.to());
            if p.is_none() {
                PieceType::Pawn
            } else {
                p.ty()
            }
        };
        let attacker = pos.piece_on(mv.from()).ty();
        let see_bonus = if see_ge(pos, mv, 0) { 1_000_000 } else { -500_000 };
        let pc = pos.piece_on(mv.from()).index();
        let ch = if pc < 12 { td.cap_hist[cap_idx(pc, mv.to().index(), victim.index())] as i32 / 8 } else { 0 };
        return 2_000_000 + see_bonus + MVV_LVA[attacker.index()][victim.index()] + ch;
    }
    if mv == td.killers[ply][0] {
        return 900_000;
    }
    if mv == td.killers[ply][1] {
        return 800_000;
    }
    if ply > 0 {
        let prev = td.ply_move[ply - 1];
        if !prev.is_none() && td.counter[prev.from().index()][prev.to().index()] == mv {
            return 850_000;
        }
    }
    quiet_hist(td, pos, mv, ply)
}

fn sort_moves(
    td: &ThreadData,
    pos: &Position,
    moves: &mut [Move],
    n: usize,
    tt_move: Move,
    ply: usize,
) {
    let mut scores = [0i32; MAX_MOVES];
    for i in 0..n {
        scores[i] = score_move(td, pos, moves[i], tt_move, ply);
    }
    for i in 1..n {
        let key_m = moves[i];
        let key_s = scores[i];
        let mut j = i;
        while j > 0 && scores[j - 1] < key_s {
            moves[j] = moves[j - 1];
            scores[j] = scores[j - 1];
            j -= 1;
        }
        moves[j] = key_m;
        scores[j] = key_s;
    }
}

fn is_repetition(td: &ThreadData, pos: &Position) -> bool {
    let h = pos.hash;
    let len = td.hash_hist.len();
    let look = pos.halfmove as usize;
    let start = len.saturating_sub(look);
    let mut hits = 0;
    let mut i = len;
    while i > start {
        i -= 1;
        if td.hash_hist[i] == h {
            hits += 1;
            if hits >= 1 {
                return true;
            }
        }
    }
    false
}

fn is_draw(td: &ThreadData, pos: &Position, ply: i32) -> bool {
    if pos.halfmove >= 100 {
        return true;
    }
    if pos.is_insufficient_material() {
        return true;
    }
    ply > 0 && is_repetition(td, pos)
}

fn update_history(td: &mut ThreadData, pos: &Position, mv: Move, ply: usize, bonus: i32) {
    gravity(&mut td.history[pos.side.index()][mv.from().index()][mv.to().index()], bonus);
    let pc = pos.piece_on(mv.from()).index();
    let to = mv.to().index();
    if pc < 12 {
        if ply >= 1 {
            let i = cont_idx(td.stack_pc[ply - 1] as usize, td.stack_to[ply - 1] as usize, pc, to);
            gravity(&mut td.cont_hist[i], bonus);
        }
        if ply >= 2 {
            let i = cont_idx(td.stack_pc[ply - 2] as usize, td.stack_to[ply - 2] as usize, pc, to);
            gravity(&mut td.cont_hist[i], bonus);
        }
    }
}

fn update_cap_history(td: &mut ThreadData, pos: &Position, mv: Move, bonus: i32) {
    let pc = pos.piece_on(mv.from()).index();
    if pc >= 12 {
        return;
    }
    let victim = if mv.is_ep() {
        PieceType::Pawn
    } else {
        let p = pos.piece_on(mv.to());
        if p.is_none() { PieceType::Pawn } else { p.ty() }
    };
    gravity(&mut td.cap_hist[cap_idx(pc, mv.to().index(), victim.index())], bonus);
}

fn update_killers(td: &mut ThreadData, ply: usize, mv: Move) {
    if td.killers[ply][0] != mv {
        td.killers[ply][1] = td.killers[ply][0];
        td.killers[ply][0] = mv;
    }
}

fn qsearch(pos: &Position, td: &mut ThreadData, mut alpha: i32, beta: i32, ply: i32) -> i32 {
    td.nodes += 1;
    if ply as usize >= MAX_PLY - 1 {
        return evaluate(pos);
    }
    if td.should_stop() {
        return 0;
    }
    td.seldepth = td.seldepth.max(ply);

    if is_draw(td, pos, ply) {
        return DRAW;
    }

    let in_check = pos.in_check();
    let orig_alpha = alpha;
    let q_pv = beta - alpha > 1;
    if let Some((_mv, s, d, b)) = td.tt.probe(pos.hash) {
        let _ = d;
        if !q_pv {
            let ts = from_tt(s, ply);
            if b == BOUND_EXACT
                || (b == BOUND_LOWER && ts >= beta)
                || (b == BOUND_UPPER && ts <= alpha)
            {
                return ts;
            }
        }
    }
    let stand = if in_check {
        -INF
    } else {
        evaluate(pos)
    };
    if !in_check {
        if stand >= beta {
            return stand;
        }
        if stand > alpha {
            alpha = stand;
        }
    }

    let mut moves = [Move::NONE; MAX_MOVES];
    let n = if in_check {
        legal_moves(pos, &mut moves)
    } else {
        noisy_moves(pos, &mut moves)
    };
    if n == 0 {
        return if in_check { mated_in(ply) } else { stand };
    }
    sort_moves(td, pos, &mut moves, n, Move::NONE, ply as usize);

    let mut best = stand;
    for i in 0..n {
        let mv = moves[i];
        if !in_check {
            if !mv.is_promo() && stand + 200 + crate::eval::SEE_VALUE
                [if mv.is_ep() { 0 } else { pos.piece_on(mv.to()).ty().index() }]
                < alpha
            {
                continue;
            }
            if !see_ge(pos, mv, 0) {
                continue;
            }
        }
        let mut np = *pos;
        np.make(mv);
        if np.was_illegal() {
            continue;
        }
        td.tt.prefetch(np.hash);
        td.hash_hist.push(pos.hash);
        let score = -qsearch(&np, td, -beta, -alpha, ply + 1);
        td.hash_hist.pop();
        if td.time.stopped() {
            return 0;
        }
        if score > best {
            best = score;
            if score > alpha {
                alpha = score;
                if score >= beta {
                    break;
                }
            }
        }
    }
    let qb = if best >= beta {
        BOUND_LOWER
    } else if best > orig_alpha {
        BOUND_EXACT
    } else {
        BOUND_UPPER
    };
    td.tt.store(pos.hash, Move::NONE, to_tt(best, ply), 0, qb);
    best
}

fn search(
    pos: &Position,
    td: &mut ThreadData,
    mut alpha: i32,
    mut beta: i32,
    mut depth: i32,
    ply: i32,
    is_pv: bool,
    cut_node: bool,
    excluded: Move,
) -> i32 {
    td.pv_len[ply as usize] = 0;
    td.nodes += 1;
    if ply as usize >= MAX_PLY - 1 {
        return evaluate(pos);
    }
    if td.should_stop() {
        return 0;
    }
    td.seldepth = td.seldepth.max(ply);

    let in_check = pos.in_check();
    // Do not stack check-ext on a singular-excluded re-entry (same ply).
    if in_check && excluded.is_none() {
        depth += 1;
    }

    if depth <= 0 {
        return qsearch(pos, td, alpha, beta, ply);
    }

    if ply > 0 && is_draw(td, pos, ply) {
        return DRAW;
    }

    // Mate distance pruning
    alpha = alpha.max(mated_in(ply));
    beta = beta.min(mate_in(ply + 1));
    if alpha >= beta {
        return alpha;
    }

    let mut tt_move = Move::NONE;
    let mut tt_score = 0;
    let mut tt_bound = 0;
    let mut tt_hit = false;
    let mut tt_depth = 0i32;
    if let Some((mv, s, d, b)) = td.tt.probe(pos.hash) {
        tt_hit = true;
        tt_move = mv;
        tt_score = from_tt(s, ply);
        tt_bound = b;
        tt_depth = d;
        if excluded.is_none() && !is_pv && d >= depth {
            if b == BOUND_EXACT
                || (b == BOUND_LOWER && tt_score >= beta)
                || (b == BOUND_UPPER && tt_score <= alpha)
            {
                return tt_score;
            }
        }
    }

    // Pawn corrhist from PR #5 is *not* applied. `static_eval += corr/32`
    // saturates at ±512 cp (gravity uses HIST_MAX=16384), poisons RFP/NMP, and
    // scored 12.5% vs SF19 UCI_Elo 2400 against 69% with this line gone and
    // 71% on 451d933 (pre-#5). Re-add only with a SPRT vs this baseline.
    let static_eval = if in_check {
        -INF
    } else {
        evaluate(pos)
    };
    td.eval_stack[ply as usize] = static_eval;
    let improving = ply >= 2
        && !in_check
        && static_eval > td.eval_stack[(ply as usize).saturating_sub(2)];

    // Reverse futility pruning
    if excluded.is_none()
        && !is_pv
        && !in_check
        && depth <= params::rfp_depth()
        && static_eval - params::rfp_margin() * depth + (if improving { params::rfp_improving() } else { 0 }) >= beta
        && static_eval.abs() < MATE_IN_MAX_LOCAL
    {
        return static_eval;
    }

    // Null move pruning
    if excluded.is_none()
        && !is_pv
        && !in_check
        && depth >= 3
        && static_eval >= beta
        && pos.bb(pos.side, PieceType::Pawn) != pos.colors[pos.side.index()] ^ pos.bb(pos.side, PieceType::King)
        && !(tt_hit && tt_bound == BOUND_UPPER && tt_score < beta)
    {
        let r = params::nmp_base() + depth / params::nmp_depth_div() + ((static_eval - beta) / params::nmp_eval_div()).clamp(0, 3);
        let mut np = *pos;
        // make null
        if let Some(ep) = np.ep {
            np.hash ^= crate::zobrist::ZOBRIST.ep[ep.file() as usize];
            np.ep = None;
        }
        np.side = np.side.flip();
        np.hash ^= crate::zobrist::ZOBRIST.side;
        td.hash_hist.push(pos.hash);
        td.stack_pc[ply as usize] = 12;
        td.stack_to[ply as usize] = 0;
        td.ply_move[ply as usize] = Move::NONE;
        let score = -search(&np, td, -beta, -beta + 1, depth - 1 - r, ply + 1, false, !cut_node, Move::NONE);
        td.hash_hist.pop();
        if td.time.stopped() {
            return 0;
        }
        if score >= beta {
            return if is_mate_score(score) { beta } else { score };
        }
    }

    // Internal iterative reduction
    if excluded.is_none() && depth >= 4 && tt_move.is_none() {
        depth -= 1;
    }

    // Singular extension: TT move uniquely good → search it one ply deeper.
    // Never stack with in-check extension (bench d10 used to hang).
    let mut se_ext = 0i32;
    if excluded.is_none()
        && !in_check
        && ply > 0
        && depth >= 8
        && !tt_move.is_none()
        && tt_hit
        && tt_depth >= depth - 3
        && tt_bound != BOUND_UPPER
        && tt_score.abs() < MATE_IN_MAX_LOCAL
    {
        let rbeta = (tt_score - (params::se_margin() + depth)).max(-MATE_IN_MAX_LOCAL);
        let sdepth = ((depth - 1) / 2).max(1);
        let s = search(pos, td, rbeta - 1, rbeta, sdepth, ply, false, cut_node, tt_move);
        if td.time.stopped() {
            return 0;
        }
        if s < rbeta {
            se_ext = 1;
        } else if !is_pv && s >= beta {
            return s;
        }
    }

    let mut moves = [Move::NONE; MAX_MOVES];
    let n = pseudo_legal(pos, &mut moves);
    if n == 0 {
        return if in_check { mated_in(ply) } else { DRAW };
    }
    sort_moves(td, pos, &mut moves, n, tt_move, ply as usize);

    let mut best = -INF;
    let mut best_move = Move::NONE;
    let mut legal = 0;
    let mut quiets: [Move; 64] = [Move::NONE; 64];
    let mut quiet_n = 0;
    let mut caps: [Move; 32] = [Move::NONE; 32];
    let mut cap_n = 0;
    let mut bound = BOUND_UPPER;

    for i in 0..n {
        let mv = moves[i];
        if mv == excluded {
            continue;
        }
        let is_quiet = !mv.is_capture() && !mv.is_promo();

        // Late move pruning
        if !is_pv
            && !in_check
            && is_quiet
            && legal > 0
            && depth <= 7
            && legal >= params::lmp_base() + depth * depth / if improving { 1 } else { 2 }
        {
            continue;
        }

        // History pruning of quiets that have consistently failed
        let mv_hist = if is_quiet { quiet_hist(td, pos, mv, ply as usize) } else { 0 };
        if !is_pv && !in_check && is_quiet && legal > 0 && depth <= 4 && mv_hist < -params::hist_prune() * depth {
            continue;
        }

        // SEE pruning of bad captures / quiets
        if !is_pv && !in_check && legal > 0 && depth <= 6 {
            let th = if is_quiet { -params::see_quiet() * depth } else { -params::see_capture() * depth };
            if !see_ge(pos, mv, th) {
                continue;
            }
        }

        // Futility pruning
        if !is_pv
            && !in_check
            && is_quiet
            && depth <= params::futility_depth()
            && static_eval + params::futility_margin() * depth < alpha
            && legal > 0
        {
            continue;
        }

        let mut np = *pos;
        np.make(mv);
        if np.was_illegal() {
            continue;
        }
        td.tt.prefetch(np.hash);
        legal += 1;
        let gives_check = np.in_check();

        // In-check nodes already get +1 at search entry. Do not also extend
        // the checking move — that doubled extensions and exploded the tree
        // (bench depth 10 never finished). Singular ext is at most +1 and
        // never applied in check.
        let new_depth = depth - 1 + if mv == tt_move { se_ext } else { 0 };

        td.hash_hist.push(pos.hash);
        td.stack_pc[ply as usize] = pos.piece_on(mv.from()).index() as u8;
        td.stack_to[ply as usize] = mv.to().index() as u8;
        td.ply_move[ply as usize] = mv;

        let mut score;
        if legal == 1 {
            score = -search(&np, td, -beta, -alpha, new_depth, ply + 1, is_pv, false, Move::NONE);
        } else {
            // LMR
            let mut r = 0;
            if is_quiet && depth >= 3 && legal >= 4 {
                r = lmr_reduction(depth, legal);
                if !is_pv {
                    r += 1;
                }
                if cut_node {
                    r += 1;
                }
                if !improving {
                    r += 1;
                }
                if gives_check {
                    r = r.saturating_sub(1);
                }
                r -= mv_hist / params::lmr_hist_div();
                r = r.clamp(0, new_depth - 1);
            }
            score = -search(
                &np,
                td,
                -alpha - 1,
                -alpha,
                new_depth - r,
                ply + 1,
                false,
                true,
                Move::NONE,
            );
            if r > 0 && score > alpha {
                score = -search(&np, td, -alpha - 1, -alpha, new_depth, ply + 1, false, !cut_node, Move::NONE);
            }
            if is_pv && score > alpha && score < beta {
                score = -search(&np, td, -beta, -alpha, new_depth, ply + 1, true, false, Move::NONE);
            }
        }

        td.hash_hist.pop();
        if td.time.stopped() {
            return 0;
        }

        if score > best {
            best = score;
            best_move = mv;
            if score > alpha {
                alpha = score;
                bound = BOUND_EXACT;
                td.pv[ply as usize][0] = mv;
                let child = (ply as usize) + 1;
                let len = td.pv_len[child];
                for i in 0..len {
                    td.pv[ply as usize][i + 1] = td.pv[child][i];
                }
                td.pv_len[ply as usize] = len + 1;
                if score >= beta {
                    bound = BOUND_LOWER;
                    let bonus = (170 * depth - 90).clamp(0, 1800);
                    if is_quiet {
                        update_killers(td, ply as usize, mv);
                        update_history(td, pos, mv, ply as usize, bonus);
                        if ply > 0 {
                            let prev = td.ply_move[ply as usize - 1];
                            if !prev.is_none() {
                                td.counter[prev.from().index()][prev.to().index()] = mv;
                            }
                        }
                        for k in 0..quiet_n {
                            update_history(td, pos, quiets[k], ply as usize, -bonus);
                        }
                    } else if mv.is_capture() {
                        update_cap_history(td, pos, mv, bonus);
                    }
                    for k in 0..cap_n {
                        update_cap_history(td, pos, caps[k], -bonus);
                    }
                    break;
                }
            }
        }
        if is_quiet && quiet_n < 64 {
            quiets[quiet_n] = mv;
            quiet_n += 1;
        } else if mv.is_capture() && cap_n < 32 {
            caps[cap_n] = mv;
            cap_n += 1;
        }
    }

    if legal == 0 {
        return if in_check { mated_in(ply) } else { DRAW };
    }

    if excluded.is_none() {
        td.tt.store(pos.hash, best_move, to_tt(best, ply), depth, bound);
    }
    best
}

fn lmr_reduction(depth: i32, move_no: i32) -> i32 {
    // ln(depth)*ln(move_no) style table, integer approx
    let d = depth.max(1) as f32;
    let m = move_no.max(1) as f32;
    ((d.ln() * m.ln()) * (params::lmr_scale() as f32 / 100.0)) as i32
}

const MATE_IN_MAX_LOCAL: i32 = crate::types::MATE_IN_MAX;

pub struct SearchResult {
    pub best: Move,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
    pub pv: Vec<Move>,
}

pub fn iterative_deepening(pos: &Position, td: &mut ThreadData, use_book: bool) -> SearchResult {
    td.tt.new_search();
    td.nodes = 0;
    td.best = Move::NONE;
    if use_book {
        if let Some(mv) = crate::book::probe(pos) {
            let mut legal = [Move::NONE; MAX_MOVES];
            let n = legal_moves(pos, &mut legal);
            if (0..n).any(|i| legal[i] == mv) {
                if !td.silent {
                    println!("info string book {}", mv);
                }
                return SearchResult {
                    best: mv,
                    score: 0,
                    depth: 0,
                    nodes: 0,
                    pv: vec![mv],
                };
            }
        }
    }
    let mut last_score = 0;
    let mut last_pv: Vec<Move> = Vec::new();
    let mut last_depth = 0;
    let mut stable = 0;

    let mut root_moves = [Move::NONE; MAX_MOVES];
    let rn = legal_moves(pos, &mut root_moves);
    if rn == 0 {
        return SearchResult {
            best: Move::NONE,
            score: if pos.in_check() { -MATE } else { 0 },
            depth: 0,
            nodes: 0,
            pv: Vec::new(),
        };
    }
    td.best = root_moves[0];

    for depth in 1..=td.time.max_depth {
        td.seldepth = 0;
        td.root_depth = depth;
        let mut alpha = -INF;
        let mut beta = INF;
        let mut window = params::asp_window();
        if depth >= 5 {
            alpha = last_score - window;
            beta = last_score + window;
        }
        let mut score;
        loop {
            score = search(pos, td, alpha, beta, depth, 0, true, false, Move::NONE);
            if td.time.stopped() || td.time.hard_expired() {
                break;
            }
            if score <= alpha {
                beta = (alpha + beta) / 2;
                alpha = (score - window).max(-INF);
                window += window / 2 + 10;
            } else if score >= beta {
                beta = (score + window).min(INF);
                window += window / 2 + 10;
            } else {
                break;
            }
            if window > 2000 {
                alpha = -INF;
                beta = INF;
            }
        }
        if td.time.stopped() || td.time.hard_expired() {
            break;
        }

        last_depth = depth;
        last_score = score;
        td.score = score;
        if td.pv_len[0] > 0 {
            td.best = td.pv[0][0];
            last_pv = td.pv[0][..td.pv_len[0]].to_vec();
        } else if !td.best.is_none() {
            last_pv = vec![td.best];
        }

        let elapsed = td.time.elapsed().as_millis().max(1);
        let nps = td.nodes as u128 * 1000 / elapsed as u128;
        let mut pv_s = String::new();
        for m in &last_pv {
            pv_s.push(' ');
            pv_s.push_str(&m.to_uci());
        }
        let score_str = if is_mate_score(score) {
            let mate = if score > 0 {
                (MATE - score + 1) / 2
            } else {
                -(MATE + score) / 2
            };
            format!("mate {}", mate)
        } else {
            format!("cp {}", score)
        };
        if !td.silent {
            println!(
                "info depth {} seldepth {} score {} nodes {} nps {} time {} pv{}",
                depth,
                td.seldepth,
                score_str,
                td.nodes,
                nps,
                elapsed,
                pv_s
            );
        }

        if last_pv.first().copied() == Some(td.best) {
            stable += 1;
        } else {
            stable = 0;
        }
        if td.time.soft_expired(stable >= 4) {
            break;
        }
        if td.nodes >= td.time.max_nodes {
            break;
        }
    }

    if last_pv.is_empty() {
        last_pv.push(td.best);
    }
    SearchResult {
        best: td.best,
        score: last_score,
        depth: last_depth,
        nodes: td.nodes,
        pv: last_pv,
    }
}

/// OpenBench / Reckless-style bench: search a fixed set of positions to a
/// fixed depth and print `nodes nps`.
pub fn bench(depth: i32) -> (u64, u64) {
    const FENS: &[&str] = &[
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
        "8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1",
        "8/5k2/8/8/8/8/1K6/1B6 w - - 0 1",
        "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 4",
        "2r3k1/1q1nbppp/r3p3/3pP3/pPpP4/P1Q2N2/2RN1PPP/2R4K b - b3 0 23",
    ];
    let _ = crate::attacks::tables();
    let tt = Arc::new(TranspositionTable::new(16));
    let mut total_nodes = 0u64;
    let start = std::time::Instant::now();
    for fen in FENS {
        let pos = Position::from_fen(fen).expect("bench fen");
        let limits = crate::time::Limits {
            depth: Some(depth),
            infinite: false,
            ..Default::default()
        };
        let tm = Arc::new(TimeManager::new(limits, pos.side == Color::White, 0));
        // force depth-only
        // TimeManager with only depth set becomes infinite-time; we cap via max_depth.
        let mut td = ThreadData::new(tt.clone(), tm);
        td.silent = true;
        let _ = iterative_deepening(&pos, &mut td, false);
        total_nodes += td.nodes;
        tt.clear();
    }
    let ms = start.elapsed().as_millis().max(1);
    let nps = total_nodes as u128 * 1000 / ms as u128;
    println!("{} nodes {} nps", total_nodes, nps);
    (total_nodes, nps as u64)
}
