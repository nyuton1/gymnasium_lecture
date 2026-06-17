"""SAC (Soft Actor-Critic) を BipedalWalkerHardcore-v3 で学習・再生するスクリプト。

SAC は「報酬の最大化」に加えて「方策のエントロピー最大化」を目的に組み込んだ
オフポリシー型アクター–クリティック手法です。確率的な方策とエントロピー項により
探索と活用のバランスを自動調整し、高いサンプル効率とハイパーパラメータへの
頑健性を実現します。

実行例:
    python sac.py                      # 学習(2000step) → ベストモデル録画・再生
    python sac.py --timesteps 50000    # 学習ステップ数を増やす（元ノートブック値）
    python sac.py --mode play          # 学習をスキップし、保存済みモデルを再生
    python sac.py --mode train         # 学習のみ（録画・再生しない）

注意:
    既定の learning_starts=3000 は「ランダム行動で経験を集めてから学習を開始する」
    ステップ数です。--timesteps を 3000 以下にすると勾配更新が一度も走らず、
    パイプラインの動作確認はできても方策はほぼ学習されません。実際に学習させる
    場合は --timesteps を大きくしてください（元ノートブックでは 50000）。

学習過程は TensorBoard で確認できます:
    tensorboard --logdir tensorboard/
"""

import argparse
import os

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_utils import record_agent_video

# =============================================================================
# 設定（元 Colab ノートブックの SAC セルから移植）
# =============================================================================
ENV_ID = "BipedalWalkerHardcore-v3"
LOG_DIR = "./sac_logs_bipedalwalkerhardcore/"
VIDEO_FOLDER = "sac_bipedalwalkerhardcore_videos_practice"
FINAL_MODEL = "sac_bipedalwalkerhardcore"


def train(timesteps: int) -> None:
    """SAC モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。"""
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
        name_prefix="sac_model",
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
    # 3. SAC モデルの構築
    #    ent_coef='auto': エントロピー係数を自動調整 / オフポリシー + リプレイバッファ
    # -------------------------------------------------------------------------
    model = SAC(
        policy="MlpPolicy",
        env=env,
        gamma=0.99,
        learning_rate=0.00035,
        buffer_size=100_000,
        learning_starts=3000,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        verbose=1,
        tensorboard_log="tensorboard/",
    )

    # -------------------------------------------------------------------------
    # 4. 学習 → 最終モデル保存 → 後片付け
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
            "学習済みモデルが見つかりません。先に `python sac.py --mode train` を実行してください。"
        )

    agent = SAC.load(model_path)
    record_agent_video(agent, ENV_ID, VIDEO_FOLDER, deterministic=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAC を BipedalWalkerHardcore-v3 で学習・再生")
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
