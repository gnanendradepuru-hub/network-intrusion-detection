#!/usr/bin/env bash
# Fetch NSL-KDD (train + test) into ./data
set -euo pipefail
mkdir -p data && cd data
base="https://raw.githubusercontent.com/defcom17/NSL_KDD/master"
curl -fsSL -o "KDDTrain+.txt" "$base/KDDTrain%2B.txt"
curl -fsSL -o "KDDTest+.txt"  "$base/KDDTest%2B.txt"
wc -l KDD*.txt
