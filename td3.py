"""TD3 (Twin Delayed DDPG) を BipedalWalker-v3 で学習・再生するスクリプト。

TD3 は DDPG の改良版で、Q 値の過大推定を抑えるために
(1) 2 つのクリティックの小さい方をターゲットに使う、
(2) ポリシー更新を遅らせる（policy_delay）、
(3) ターゲットポリシーにノイズを加えて平滑化する、
という 3 つの工夫を導入しています。これにより DDPG より安定して学習できます。

実行例:
    python td3.py                      # 学習(2000step) → ベストモデル録画・再生
    python td3.py --timesteps 50000    # 学習ステップ数を増やす
    python td3.py --mode play          # 学習をスキップし、保存済みモデルを再生
    python td3.py --mode train         # 学習のみ（録画・再生しない）

学習過程は TensorBoard で確認できます:
    tensorboard --logdir tensorboard/
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_utils import record_agent_video

# =============================================================================
# 設定（元 Colab ノートブックの TD3 セルから移植）
# =============================================================================
ENV_ID = "BipedalWalker-v3"
LOG_DIR = "./td3_logs_bipedalwalker/"
VIDEO_FOLDER = "td3_bipedalwalker_videos_practice"
FINAL_MODEL = "td3_bipedalwalker"


def train(timesteps: int) -> None:
    """TD3 モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。"""
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
        name_prefix="td3_model",
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
    # 4. TD3 モデルの構築（DDPG と同様 + policy_delay でポリシー更新を遅延）
    # -------------------------------------------------------------------------
    model = TD3(
        policy="MlpPolicy",
        env=env,
        action_noise=action_noise,
        gamma=0.99,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_delay=2,
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
            "学習済みモデルが見つかりません。先に `python td3.py --mode train` を実行してください。"
        )

    agent = TD3.load(model_path)
    record_agent_video(agent, ENV_ID, VIDEO_FOLDER, deterministic=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="TD3 を BipedalWalker-v3 で学習・再生")
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
