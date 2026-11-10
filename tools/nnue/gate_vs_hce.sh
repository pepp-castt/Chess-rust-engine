#!/usr/bin/env bash
# Head-to-head gate: (HCE + residual net) vs pure HCE, same binary, same book, colours reversed.
#   tools/nnue/gate_vs_hce.sh ENGINE NET [ROUNDS=50] [TC=10+0.1] [CONC=4] [DELTA_CAP=400] [OUT=gate]
# ROUNDS*2 games. TEST uses UseNNUE=true with CRE_NNUE_FILE=NET; BASE uses UseNNUE=false (HCE).
# Floor is 5+0.05 (repo rule: nothing faster teaches/measures anything).
set -euo pipefail
ENG=$1; NET=$2; ROUNDS=${3:-50}; TC=${4:-10+0.1}; CONC=${5:-4}; CAP=${6:-400}; OUT=${7:-gate}
BOOK=${BOOK:-tools/openings_book.epd}
FC=${FASTCHESS:-fastchess}
mkdir -p "$OUT"
# TC like "10+0.1" -> time control; TC like "nodes=25000" -> fixed nodes (isolates eval quality from NPS).
case "$TC" in nodes=*|depth=*) EACH="$TC";; *) EACH="tc=$TC";; esac
export CRE_NNUE_FILE="$NET" CRE_DELTA_CAP="$CAP"
"$FC" \
  -engine cmd="$ENG" name=test option.Hash=16 option.Threads=1 option.UseNNUE=true \
  -engine cmd="$ENG" name=hce  option.Hash=16 option.Threads=1 option.UseNNUE=false \
  -each $EACH -rounds "$ROUNDS" -games 2 -repeat -concurrency "$CONC" \
  -openings file="$BOOK" format=epd order=random \
  -draw movenumber=40 movecount=8 score=10 -resign movecount=4 score=800 twosided=true \
  -maxmoves 300 -ratinginterval 10 -pgnout file="$OUT/games.pgn" 2>&1 \
  | grep -vE '^(Warning|Info|Position|Moves);' | tee "$OUT/gate.log" | grep -E 'Elo:|Games:|Score of|Ptnml|WL/DD' | tail -n 12
