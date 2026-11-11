#!/usr/bin/env bash
# Net-vs-HCE gate: same binary, only the evaluation differs. Reproduces E's 100-game gate.
#   tools/gate_net.sh <engine> <net.bin> [games=100] [tc=10+0.1] [concurrency=3] [out_dir=gate_out]
# Verdict is the W-D-L / Elo line fastchess prints. A positive Elo with a CI that excludes 0 is
# required before a net is allowed anywhere near `main`.  Never promote on validation loss.
set -euo pipefail
ENG=$(readlink -f "$1"); NET=$(readlink -f "$2")
GAMES=${3:-100}; TC=${4:-10+0.1}; CONC=${5:-3}; OUT=${6:-gate_out}
FASTCHESS=${FASTCHESS:-fastchess}
BOOK=${BOOK:-$(dirname "$0")/openings_book.epd}
mkdir -p "$OUT"
cat > "$OUT/net.sh" <<WRAP
#!/usr/bin/env bash
export CRE_NNUE_FILE="$NET"
exec "$ENG" "\$@"
WRAP
chmod +x "$OUT/net.sh"
"$FASTCHESS" \
  -engine name=NET cmd="$OUT/net.sh" option.UseNNUE=true option.Hash=64 \
  -engine name=HCE cmd="$ENG" option.UseNNUE=false option.Hash=64 \
  -each tc="$TC" \
  -openings file="$BOOK" format=epd order=random -srand 7 \
  -rounds $((GAMES / 2)) -games 2 -repeat -concurrency "$CONC" \
  -pgnout file="$OUT/games.pgn" -recover -ratinginterval 10 2>&1 | tee "$OUT/gate.log" | tail -n 25
echo "config: engine_sha=$(sha256sum "$ENG" | cut -c1-12) net_sha=$(sha256sum "$NET" | cut -c1-12) tc=$TC games=$GAMES book=$(basename "$BOOK") hash=64 threads=1" | tee "$OUT/config.txt"
