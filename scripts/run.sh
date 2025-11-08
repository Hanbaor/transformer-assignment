#!/usr/bin/env bash
set -euo pipefail

# Detect project root (the parent directory of this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYTHON="${PYTHON:-python}"

TRAIN_SCRIPT="${ROOT_DIR}/src/train.py"
TRANSLATE_SCRIPT="${ROOT_DIR}/src/translate.py"
CONFIG_FILE="${ROOT_DIR}/configs/transformer.yml"
OUT_DIR="${ROOT_DIR}/results"

# Customizable parameters via env vars
SENTENCE="${SENTENCE:-Eine Gruppe von Menschen steht vor einem Gebäude.}"
BEAM_SIZE="${BEAM_SIZE:-5}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.6}"
KEEP_UNK="${KEEP_UNK:-0}"        # 1 to keep <unk>, otherwise 0
UNK_REPLACE="${UNK_REPLACE:-0}"  # 1 to enable UNK replacement via attention

# Resolve flags
KEEP_UNK_FLAG=""
UNK_REPLACE_FLAG=""
[[ "${KEEP_UNK}" == "1" ]] && KEEP_UNK_FLAG="--keep_unk"
[[ "${UNK_REPLACE}" == "1" ]] && UNK_REPLACE_FLAG="--unk_replace"

echo "== DE→EN Transformer (IWSLT17) =="
echo "Project root: ${ROOT_DIR}"
echo "Python: $(command -v "${PYTHON}")"
echo "Config: ${CONFIG_FILE}"
echo "Results dir: ${OUT_DIR}"
echo "-------------------------------------------"

# Pre-run checks
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_FILE}"
  exit 1
fi

DATA_DIR="${ROOT_DIR}/data/en-de"
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ERROR] Data directory not found: ${DATA_DIR}"
  echo "Make sure IWSLT17 EN-DE files are placed under 'data/en-de/'."
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "[1/3] Training..."
"${PYTHON}" "${TRAIN_SCRIPT}" --config "${CONFIG_FILE}"
echo "Training done."
echo "-------------------------------------------"

echo "[2/3] Single-sentence translation..."
echo "Sentence: ${SENTENCE}"
"${PYTHON}" "${TRANSLATE_SCRIPT}" \
  --config "${CONFIG_FILE}" \
  --sentence "${SENTENCE}" \
  --beam_size "${BEAM_SIZE}" \
  --length_penalty "${LENGTH_PENALTY}" \
  ${KEEP_UNK_FLAG} \
  ${UNK_REPLACE_FLAG}
echo "Single-sentence translation done."
echo "-------------------------------------------"

echo "[3/3] Test set translation with BLEU..."
PAIR_OUT="${OUT_DIR}/test_pairs.txt"
"${PYTHON}" "${TRANSLATE_SCRIPT}" \
  --config "${CONFIG_FILE}" \
  --dataset test \
  --eval \
  --output "${PAIR_OUT}" \
  --beam_size "${BEAM_SIZE}" \
  --length_penalty "${LENGTH_PENALTY}" \
  ${KEEP_UNK_FLAG} \
  ${UNK_REPLACE_FLAG}
echo "Saved pairs to: ${PAIR_OUT}"
echo "Results saved under '${OUT_DIR}'."
echo "All done!"