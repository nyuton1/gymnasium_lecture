#!/usr/bin/env bash
# =============================================================================
# ローカル実行環境セットアップスクリプト
# -----------------------------------------------------------------------------
# Colab の `!apt install ...` / `!pip install ...` / `!gdown ...` を置き換え、
# ローカル（macOS / Linux）で深層強化学習スクリプトを動かすための venv を作ります。
#
# 使い方:
#   bash setup.sh
# 実行後、仮想環境を有効化してからスクリプトを実行してください:
#   source .venv/bin/activate
#   python a2c.py
# =============================================================================
set -euo pipefail

# このスクリプトが置かれているディレクトリへ移動（どこから実行しても動くように）
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> Python: $("${PYTHON_BIN}" --version)"

# -----------------------------------------------------------------------------
# 1. ffmpeg の確認（RecordVideo の mp4 書き出しに必要）
#    無い場合は警告だけ出して続行（録画以外は動くため）。
# -----------------------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNING: ffmpeg が見つかりません。動画の録画には ffmpeg が必要です。"
  echo "         macOS: brew install ffmpeg / Ubuntu: sudo apt install ffmpeg"
fi

# -----------------------------------------------------------------------------
# 2. 仮想環境（venv）の作成
# -----------------------------------------------------------------------------
if [ ! -d "${VENV_DIR}" ]; then
  echo "==> 仮想環境を作成: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "==> 既存の仮想環境を使用: ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# -----------------------------------------------------------------------------
# 3. pip 更新
# -----------------------------------------------------------------------------
echo "==> pip を更新"
pip install --upgrade pip

# -----------------------------------------------------------------------------
# 4. swig（システム版）を用意
#    gymnasium[box2d] が依存する box2d-py のビルドに swig が必須。
#    PyPI の `swig` パッケージは環境によってビルド時に壊れることがあるため、
#    macOS は Homebrew、Linux は apt のシステム版 swig を使う。
# -----------------------------------------------------------------------------
if command -v swig >/dev/null 2>&1; then
  echo "==> swig は既に利用可能: $(swig -version | grep -i version | head -1)"
else
  echo "==> swig が見つかりません。インストールを試みます（box2d-py のビルド用）"
  if [ "$(uname)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    brew install swig
  elif command -v apt >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y swig
  else
    echo "ERROR: swig を自動インストールできませんでした。手動で導入してください。"
    echo "       macOS: brew install swig / Ubuntu: sudo apt install swig"
    exit 1
  fi
fi

# -----------------------------------------------------------------------------
# 5. 依存パッケージ一括インストール
# -----------------------------------------------------------------------------
echo "==> requirements.txt をインストール"
pip install -r requirements.txt

# -----------------------------------------------------------------------------
# 6. 動作確認: BipedalWalkerHardcore-v3 が生成できるか
# -----------------------------------------------------------------------------
echo "==> 環境生成の確認 (BipedalWalkerHardcore-v3)"
python - <<'PY'
import gymnasium as gym
env = gym.make("BipedalWalkerHardcore-v3", render_mode="rgb_array")
obs, info = env.reset(seed=0)
print("OK: BipedalWalkerHardcore-v3 を生成できました。観測次元 =", obs.shape)
env.close()
PY

echo ""
echo "============================================================"
echo "セットアップ完了 🎉"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python a2c.py      # 学習 → 録画 → 再生"
echo "============================================================"
