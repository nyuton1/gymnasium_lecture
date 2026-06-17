"""A2C (Advantage Actor-Critic) を BipedalWalkerHardcore-v3 で学習・再生するスクリプト。

A3C の同期版にあたる A2C は、複数の環境を並列に動かして経験をまとめて収集し、
アクター（方策）とクリティック（価値関数）を一括更新するオンポリシー手法です。
実装がシンプルで再現性が高く、Gymnasium 環境のベースラインとしてよく使われます。

実行例:
    python a2c.py                      # 学習(2000step) → ベストモデル録画・再生
    python a2c.py --timesteps 50000    # 学習ステップ数を増やす
    python a2c.py --mode play          # 学習をスキップし、保存済みモデルを再生
    python a2c.py --mode train         # 学習のみ（録画・再生しない）

学習過程は TensorBoard で確認できます:
    tensorboard --logdir tensorboard/
"""

import argparse
import os

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_utils import record_agent_video

# =============================================================================
# 設定（元 Colab ノートブックの A2C セルから移植）
# =============================================================================
ENV_ID = "BipedalWalkerHardcore-v3"                       # 使用する Gym 環境
LOG_DIR = "./a2c_logs_bipedalwalkerhardcore/"             # ログ・モデルの保存先
VIDEO_FOLDER = "a2c-bipedalwalkerhardcore_videos_practice"  # 再生動画の保存先
FINAL_MODEL = "a2c_bipedalwalkerhardcore"                 # 学習後の最終モデル保存名(.zip)


def train(timesteps: int) -> None:
    """A2C モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。"""
    # -------------------------------------------------------------------------
    # 1. ログディレクトリの準備
    # -------------------------------------------------------------------------
    os.makedirs(LOG_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. ベクトル化環境の作成
    #    複数環境を並列実行して学習効率を上げる。make_vec_env が内部で Monitor を
    #    適用し、各エピソードの統計を monitor_dir に記録する。
    # -------------------------------------------------------------------------
    vec_env = make_vec_env(ENV_ID, n_envs=4, monitor_dir=LOG_DIR)

    # -------------------------------------------------------------------------
    # 3. チェックポイントコールバック（一定ステップごとにモデルを保存）
    # -------------------------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=LOG_DIR,
        name_prefix="a2c_model",
    )

    # -------------------------------------------------------------------------
    # 4. 評価環境と評価コールバック
    #    定期的に別環境で評価し、最高性能のモデルを best_model として保存する。
    # -------------------------------------------------------------------------
    def make_eval_env():
        return Monitor(gym.make(ENV_ID, render_mode="rgb_array"))

    eval_env = DummyVecEnv([make_eval_env])
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
    # 5. A2C モデルの構築
    #    MlpPolicy: ベクトル観測用 / n_steps: 更新までの収集ステップ数
    # -------------------------------------------------------------------------
    model = A2C(
        policy="MlpPolicy",
        env=vec_env,
        gamma=0.99,
        learning_rate=3e-4,
        n_steps=5,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log="tensorboard/",
    )

    # -------------------------------------------------------------------------
    # 6. 学習 → 最終モデル保存 → 後片付け
    # -------------------------------------------------------------------------
    model.learn(total_timesteps=timesteps, callback=callbacks)
    model.save(FINAL_MODEL)
    vec_env.close()
    eval_env.close()


def play() -> None:
    """保存済みモデルをロードし、1エピソードを録画して再生する。"""
    # best_model があればそれを、無ければ最終モデルをロードする
    best_model_path = os.path.join(LOG_DIR, "best_model", "best_model")
    if os.path.exists(best_model_path + ".zip"):
        model_path = best_model_path
    elif os.path.exists(FINAL_MODEL + ".zip"):
        model_path = FINAL_MODEL
    else:
        raise FileNotFoundError(
            "学習済みモデルが見つかりません。先に `python a2c.py --mode train` を実行してください。"
        )

    agent = A2C.load(model_path)
    record_agent_video(agent, ENV_ID, VIDEO_FOLDER, deterministic=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="A2C を BipedalWalkerHardcore-v3 で学習・再生")
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
