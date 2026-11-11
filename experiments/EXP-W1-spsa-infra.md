# EXP-W1 — SPSA infrastructure for search parameters  [Agent W]

**Status:** infrastructure only. **No Elo claim.** No tuned value is proposed here.
**Lane:** ROADMAP Phase 1 item 8 ("Tune margins with SPSA, not intuition"), previously unowned.

## What changed
| file | change |
| --- | --- |
| `src/params.rs` (new) | 16 search tunables as relaxed atomics; defaults == the old literals |
| `src/search.rs` | literal -> `params::x()` substitutions only (RFP, NMP, SE margin, LMP, history prune, SEE, futility, LMR scale/history divisor, aspiration window). No logic change |
| `src/uci.rs` | tunables listed in the `uci` handshake as spin options; `setoption` fallback; `params` command prints an OpenBench-style SPSA spec |
| `tools/spsa.py` (new) | SPSA (Spall gains, fishtest `R=a/c^2` form), fastchess `plus` vs `minus`, resume, Polyak-averaged output, `--simulate` self-test |
| `.github/workflows/spsa.yml` (new) | build + tune + artifact/resume + a paste-ready `test_opts` for `sprt.yml` |

## Pre-registered gates and results (all executed, 4 vCPU shared sandbox)
1. **Refactor must be node-identical at default params.** `bench 8` = 178914, `bench 10` = 469049, same as `main` (and as Agent S's reference numbers). **PASS.**
2. **Options must actually reach the search.** At `bench 8`: `RfpMargin=40` -> 95165 nodes, `NmpBase=2` -> 181587, `LmrScale=60` -> 153322, `AspWindow=12` -> 171939, `FutilityMargin=80` -> 173131, `SeMargin=20` -> 192253 (all != 178914). **PASS.**
3. **No measurable NPS cost.** 5 runs each of `bench 10`, base vs patched: 2.08-2.81 M vs 1.83-2.65 M nps. Both ranges overlap completely; the sandbox is shared and noisy, so this shows "no large regression", **not** "zero cost". Relaxed atomic loads compile to plain loads. Re-measure on the SPRT runner (`sprt.yml` prints both benches).
4. `cargo test --release`: perft + lib tests green.
5. **Optimiser sanity (synthetic objective, 8 params, 3000 iterations, 3 seeds):** mean |theta - optimum| fell 10.7->5.1, 16.9->10.7, 12.0->6.4. This verifies the gradient *direction* and gain schedule. It says nothing about chess strength.
6. **Real fastchess smoke (3 iterations, 12 games, 5+0.05):** parse + resume path runs end to end. It also caught a real bug in the first parser (this fastchess prints `Games: N, Wins: W, ...`, not `Score of ...`), now fixed.

## Known limits (read before spending runner hours)
* fishtest's default `r_end = 0.002` assumes 100k+ game pairs. With the ~4-5k games that fit in one 6 h job, theta would barely move, so `--r-mult` defaults to 8. This is a calibration choice, **not** validated on chess; if the tuned values oscillate, lower it.
* One 6 h job at 5+0.05 on 3 concurrent games is roughly 4-5k games ~= 500 iterations x 4 rounds. 16 parameters need more than that to converge well; chain with `resume_run_id`.
* SPSA's own win rate is the training signal. **Merge only through `sprt.yml`** (`test_opts` = tuned string, same commit as base).
* UCI option names are case-sensitive in some GUIs; the engine matches case-insensitively.

## Keep / revert rule
Keep the infrastructure (it is behaviour-neutral by gate 1). Any *tuned* value is kept only if `sprt.yml` accepts H1 against the same commit with default params.
