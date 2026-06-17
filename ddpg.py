"""DDPG (Deep Deterministic Policy Gradient) を BipedalWalkerHardcore-v3 で学習・再生するスクリプト。

DDPG は連続行動空間向けのオフポリシー型アクター–クリティック手法です。
アクターが状態から決定論的な行動を出力し、それに行動ノイズを加えて探索します。
クリティックが Q 値を推定し、経験再生バッファとターゲットネットワークで
学習を安定化させます。

実行例:
    python ddpg.py                      # 学習(2000step) → ベストモデル録画・再生
    python ddpg.py --timesteps 50000    # 学習ステップ数を増やす
    python ddpg.py --mode play          # 学習をスキップし、保存済みモデルを再生
    python ddpg.py --mode train         # 学習のみ（録画・再生しない）

学習過程は TensorBoard で確認できます:
    tensorboard --logdir tensorboard/
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_utils import record_agent_video

# =============================================================================
# 設定（元 Colab ノートブックの DDPG セルから移植）
# =============================================================================
ENV_ID = "BipedalWalkerHardcore-v3"
LOG_DIR = "./ddpg_logs_bipedalwalkerhardcore/"
VIDEO_FOLDER = "ddpg_bipedalwalkerhardcore_videos_practice"
FINAL_MODEL = "ddpg_bipedalwalkerhardcore"


def train(timesteps: int) -> None:
    """DDPG モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。"""
    os.makedirs(LOG_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. 学習用環境と評価用環境
    # -------------------------------------------------------------------------
    env = gym.make(ENV_ID, render_mode="rgb_array")
    eval_env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="rgb_array")])

    # -------------------------------------------------------------------------
    # 2. コールバック（チェックポイント保存 + 定期評価でベストモデル保存）
    # -------------------------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=LOG_DIR,
        name_prefix="ddpg_model",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "results"),
        eval_freq=500,
        deterministic=True,
        render=False,
    )
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # -------------------------------------------------------------------------
    # 3. 行動ノイズ（探索用。平均0・標準偏差0.1の正規分布ノイズ）
    # -------------------------------------------------------------------------
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    # -------------------------------------------------------------------------
    # 4. DDPG モデルの構築
    # -------------------------------------------------------------------------
    model = DDPG(
        policy="MlpPolicy",
        env=env,
        action_noise=action_noise,
        gamma=0.99,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        train_freq=(1, "step"),
        gradient_steps=1,
        tensorboard_log="tensorboard/",
        verbose=1,
    )

    # -------------------------------------------------------------------------
    # 5. 学習 → 最終モデル保存 → 後片付け
    # -------------------------------------------------------------------------
    model.learn(total_timesteps=timesteps, callback=callbacks)
    model.save(FINAL_MODEL)
    env.close()
    eval_env.close()


def play() -> None:
    """保存済みモデルをロードし、1エピソードを録画して再生する。"""
    best_model_path = os.path.join(LOG_DIR, "best_model", "best_model")
    if os.path.exists(best_model_path + ".zip"):
        model_path = best_model_path
    elif os.path.exists(FINAL_MODEL + ".zip"):
        model_path = FINAL_MODEL
    else:
        raise FileNotFoundError(
            "学習済みモデルが見つかりません。先に `python ddpg.py --mode train` を実行してください。"
        )

    agent = DDPG.load(model_path)
    record_agent_video(agent, ENV_ID, VIDEO_FOLDER, deterministic=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="DDPG を BipedalWalkerHardcore-v3 で学習・再生")
    parser.add_argument("--timesteps", type=int, default=2000, help="総学習ステップ数（既定: 2000）")
    parser.add_argument(
        "--mode",
        choices=["train", "play", "both"],
        default="both",
        help="train=学習のみ / play=再生のみ / both=学習して再生（既定）",
    )
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        train(args.timesteps)
    if args.mode in ("play", "both"):
        play()


if __name__ == "__main__":
    main()
