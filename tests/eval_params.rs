//! Guards for the tunable HCE parameter vector (Agent I).
use chess_rust_engine::board::Position;
use chess_rust_engine::eval::{evaluate_hce, evaluate_hce_with, DEFAULT_PARAMS, NP};

/// Mirror a FEN vertically and swap colours (and the side to move).
fn mirror_fen(fen: &str) -> String {
    let mut it = fen.split_whitespace();
    let board = it.next().unwrap();
    let side = it.next().unwrap();
    let ranks: Vec<String> = board
        .split('/')
        .rev()
        .map(|r| {
            r.chars()
                .map(|c| if c.is_ascii_uppercase() { c.to_ascii_lowercase() } else { c.to_ascii_uppercase() })
                .collect()
        })
        .collect();
    format!("{} {} - - 0 1", ranks.join("/"), if side == "w" { "b" } else { "w" })
}

const FENS: [&str; 8] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r2q1rk1/pp2ppbp/2np1np1/2pP4/2P1P1b1/2N2N2/PP2BPPP/R1BQ1RK1 b - - 0 9",
    "8/8/8/4k3/8/8/4PK2/8 w - - 0 1",
    "4k3/8/8/8/8/8/8/R3K2R w - - 0 1",
    "rnbqkb1r/pp2pppp/3p1n2/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 5",
];

#[test]
fn default_params_have_expected_length() {
    assert_eq!(DEFAULT_PARAMS.len(), NP);
}

#[test]
fn entry_point_matches_explicit_params() {
    for f in FENS {
        let p = Position::from_fen(f).unwrap();
        assert_eq!(evaluate_hce(&p), evaluate_hce_with(&p, &DEFAULT_PARAMS), "{f}");
    }
}

/// Colour symmetry: the eval of a position equals the eval of its colour-mirrored twin.
/// A tuned parameter set must never break this (parameters are shared by both colours).
#[test]
fn eval_is_colour_symmetric() {
    for f in FENS {
        let a = Position::from_fen(f).unwrap();
        let b = Position::from_fen(&mirror_fen(f)).unwrap();
        assert_eq!(evaluate_hce(&a), evaluate_hce(&b), "mirror of {f}");
    }
}
