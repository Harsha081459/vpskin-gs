#!/usr/bin/env bash
# GPU smoke test of the WHOLE pipeline with a MOCK SMPL and FAKE AMASS (no licensed data).
# Copies the repo to a scratch dir, so a real assets/SMPL_NEUTRAL.pkl is never touched.
# Usage: bash tools/smoke_gpu.sh [python] [scratch_dir]
set -euo pipefail
PY="${1:-python}"
WD="${2:-$HOME/vpskin-smoke}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
rm -rf "$WD"; mkdir -p "$WD"
# keep data/dna_rendering (GauHuman imports code from it); drop datasets and outputs
tar -C "$REPO" --exclude=./output --exclude=./results --exclude=./.git --exclude=./data/synthetic --exclude=./data/amass --exclude=./data/zju_mocap_refine --exclude=./data/people_snapshot --exclude=./data/fake_amass -cf - . | tar -C "$WD" -xf -
cd "$WD"
mkdir -p assets data
"$PY" - <<'EOF'
from tests.mock_smpl import make_mock_smpl
from tests.test_data_pipeline import fake_amass
make_mock_smpl("assets/SMPL_NEUTRAL.pkl")
fake_amass("data/fake_amass", ["CMU", "ACCAD", "BMLmovi"])
EOF
"$PY" tools/make_pose_bank.py --amass_root data/fake_amass --out data/amass --n_test 2 --test_len 60 --train_len 150
"$PY" tools/make_synthetic.py --smpl assets/SMPL_NEUTRAL.pkl --motions data/amass --out data/synthetic/subj0 \
      --res 256 --n_train 30 --n_views 4 --n_test_motions 1 --test_stride 20 --n_val 2
"$PY" tools/run_experiments.py --datasets synthetic --variants ${VARIANTS:-B0 B1 A1 A3 A4 A6} --iters ${ITERS:-200}
"$PY" tools/aggregate.py
echo "SMOKE OK -> $WD/results/tables.md"
