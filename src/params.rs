//! Search tunables (SPSA targets).
//!
//! Every constant here used to be a hand-set literal in `search.rs`. They are
//! now UCI spin options so a tuner (see `tools/spsa.py`) can change them per
//! engine process without recompiling, the way Stockfish / Reckless are tuned
//! on a game farm.
//!
//! Invariants:
//! * Defaults are **exactly** the previous literals, so `bench` is
//!   node-identical to the pre-refactor engine unless an option is changed.
//! * Reads are `Relaxed` atomic loads (a plain `mov` on x86-64). Options are
//!   only written from the UCI thread between searches.
//! * `min`/`max`/`step` are the SPSA search box; they keep the tuner from
//!   wandering into nonsense (e.g. negative depths).
//!
//! To add a tunable: add one line to `tunables!`, then call `params::name()`.

use std::sync::atomic::{AtomicI32, Ordering};

macro_rules! tunables {
    ($( $name:ident, $uci:literal, $def:expr, $min:expr, $max:expr, $step:expr; )*) => {
        /// Static description of one tunable.
        pub struct Spec {
            pub uci: &'static str,
            pub default: i32,
            pub min: i32,
            pub max: i32,
            /// Typical SPSA perturbation size `c_end` for this parameter.
            pub step: i32,
        }

        #[allow(non_camel_case_types)]
        #[repr(usize)]
        enum Idx { $( $name, )* __COUNT }

        pub const COUNT: usize = Idx::__COUNT as usize;

        pub static SPECS: [Spec; COUNT] = [
            $( Spec { uci: $uci, default: $def, min: $min, max: $max, step: $step }, )*
        ];

        static VALUES: [AtomicI32; COUNT] = [ $( AtomicI32::new($def), )* ];

        $(
            #[inline(always)]
            pub fn $name() -> i32 {
                VALUES[Idx::$name as usize].load(Ordering::Relaxed)
            }
        )*
    };
}

tunables! {
    //  fn name            UCI name          default  min    max    step
    // --- reverse futility pruning ---
    rfp_margin,         "RfpMargin",          70,     20,   200,    6;
    rfp_improving,      "RfpImproving",       40,      0,   150,    5;
    rfp_depth,          "RfpDepth",            8,      3,    12,    1;
    // --- null move pruning ---
    nmp_base,           "NmpBase",             3,      1,     6,    1;
    nmp_depth_div,      "NmpDepthDiv",         4,      2,    10,    1;
    nmp_eval_div,       "NmpEvalDiv",        200,     60,   500,   20;
    // --- move-count / history / SEE / futility pruning ---
    lmp_base,           "LmpBase",             4,      2,    10,    1;
    hist_prune,         "HistPrune",        2500,    500,  6000,  250;
    see_quiet,          "SeeQuiet",           20,      5,    80,    4;
    see_capture,        "SeeCapture",        100,     30,   250,   10;
    futility_margin,    "FutilityMargin",    120,     40,   300,   10;
    futility_depth,     "FutilityDepth",       6,      2,    10,    1;
    // --- singular extension ---
    se_margin,          "SeMargin",           40,     10,   120,    5;
    // --- late move reductions (scale is x100: 45 == 0.45) ---
    lmr_scale,          "LmrScale",           45,     20,    90,    3;
    lmr_hist_div,       "LmrHistDiv",       6000,   2000, 16000,  600;
    // --- aspiration windows ---
    asp_window,         "AspWindow",          25,      8,    80,    3;
}

/// Set a tunable by its UCI name. Returns `false` if the name is unknown.
/// Values are clamped into the declared range.
pub fn set(name: &str, value: i32) -> bool {
    for (i, s) in SPECS.iter().enumerate() {
        if s.uci.eq_ignore_ascii_case(name) {
            VALUES[i].store(value.clamp(s.min, s.max), Ordering::Relaxed);
            return true;
        }
    }
    false
}

/// Current value by UCI name (used by tests / the `params` command).
pub fn get(name: &str) -> Option<i32> {
    SPECS
        .iter()
        .position(|s| s.uci.eq_ignore_ascii_case(name))
        .map(|i| VALUES[i].load(Ordering::Relaxed))
}

/// Reset every tunable to its default.
pub fn reset() {
    for (i, s) in SPECS.iter().enumerate() {
        VALUES[i].store(s.default, Ordering::Relaxed);
    }
}

/// `option name ...` lines for the UCI `uci` handshake.
pub fn uci_options() -> Vec<String> {
    SPECS
        .iter()
        .map(|s| {
            format!(
                "option name {} type spin default {} min {} max {}",
                s.uci, s.default, s.min, s.max
            )
        })
        .collect()
}

/// SPSA spec lines: `name, int, default, min, max, c_end, r_end`
/// (OpenBench-shaped). Printed by the `params` UCI command so `tools/spsa.py`
/// has exactly one source of truth.
pub fn spsa_spec() -> Vec<String> {
    SPECS
        .iter()
        .map(|s| {
            format!(
                "{}, int, {}.0, {}.0, {}.0, {}.0, 0.002",
                s.uci, s.default, s.min, s.max, s.step
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_inside_bounds() {
        for s in SPECS.iter() {
            assert!(s.min <= s.default && s.default <= s.max, "{}", s.uci);
            assert!(s.step > 0, "{}", s.uci);
        }
    }

    #[test]
    fn clamp_is_pure_and_unknown_is_false() {
        // Never mutates live values: other tests run searches in parallel.
        assert!(!set("NoSuchThing", 1));
        let i = SPECS.iter().position(|s| s.uci == "RfpMargin").unwrap();
        assert_eq!(10_000.clamp(SPECS[i].min, SPECS[i].max), SPECS[i].max);
        // Setting a value to its own default is a no-op and must round-trip.
        assert!(set("rfpmargin", SPECS[i].default));
        assert_eq!(get("RfpMargin"), Some(SPECS[i].default));
    }
}
