#!/bin/bash
# Full-budget BBOB runs on the development (1-5, 71-80) and validation
# (101-115) instances, every dimension, one job per function and instance
# chunk (cocopp merges the chunks), longest jobs first.
# Usage: bash final.sh OUT [P]   (results in exdata/OUT/{P,U}{dim}, P jobs in parallel)
# Env: BIN (driver, default ../target/release/dfo-bbob), DIMS, SETS (default
# "P U") and EXTRA (extra driver flags) narrow or vary a run; DRY=1 only
# writes the job list.
cd "$(dirname "$0")"
OUT=$1; P=${2:-8}
BIN=${BIN:-../target/release/dfo-bbob}
mkdir -p logs
# COCO does not overwrite a result folder (it adds a suffix), and cocopp
# would then read both: refuse to run into an existing output.
if [ -z "$DRY" ] && [ -e "exdata/$OUT" ]; then echo "exdata/$OUT exists; remove it first" >&2; exit 1; fi
chunks() {  # instance chunks per set and dimension
  local set=$1 d=$2
  if [ "$set" = P ]; then all="1 2 3 4 5 71 72 73 74 75 76 77 78 79 80"; else all=$(seq -s ' ' 101 115); fi
  local size=15; [ "$d" -ge 20 ] && size=5; [ "$d" -ge 40 ] && size=2
  echo $all | tr ' ' '\n' | awk -v s=$size '{a[NR]=$1} END {for (i=1;i<=NR;i+=s) {c=a[i]; for (j=i+1;j<i+s&&j<=NR;j++) c=c","a[j]; print c}}'
}
for d in ${DIMS:-40 20 10 5 3 2}; do
  for f in 24 23 22 21 20 19 18 17 16 15 7 6 8 9 10 11 12 13 14 1 2 3 4 5; do
    for set in ${SETS:-P U}; do
      for c in $(chunks $set $d); do
        echo "$BIN --dims $d --functions $f --instances $c --budget 100000 --out $OUT/$set$d/f${f}_${c//,/-} $EXTRA"
      done
    done
  done
done > logs/$OUT.jobs
wc -l < logs/$OUT.jobs
[ -n "$DRY" ] && exit 0
xargs -P $P -I{} bash -c '{} > /dev/null 2>&1' < logs/$OUT.jobs
# A failed job leaves its folder missing: list any.
missing=0
while read -r job; do
  dir=${job##*--out }; dir=${dir%% *}
  [ -d "exdata/$dir" ] || { echo "missing: exdata/$dir" >&2; missing=$((missing + 1)); }
done < logs/$OUT.jobs
echo "finished ($missing missing)"
