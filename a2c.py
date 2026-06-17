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
import time
from datetime import timedelta

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_utils import FallPenaltyWrapper, record_agent_video

# =============================================================================
# 設定（元 Colab ノートブックの A2C セルから移植）
# =============================================================================
ENV_ID = "BipedalWalkerHardcore-v3"                       # 使用する Gym 環境
LOG_DIR = "./a2c_logs_bipedalwalkerhardcore/"             # ログ・モデルの保存先
VIDEO_FOLDER = "a2c-bipedalwalkerhardcore_videos_practice"  # 再生動画の保存先
FINAL_MODEL = "a2c_bipedalwalkerhardcore"                 # 学習後の最終モデル保存名(.zip)


def train(timesteps: int, n_envs: int, fall_penalty: float) -> None:
    """A2C モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。"""
    # -------------------------------------------------------------------------
    # 1. ログディレクトリの準備
    # -------------------------------------------------------------------------
    os.makedirs(LOG_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. ベクトル化環境の作成（SubprocVecEnv でサブプロセス並列）
    #    make_vec_env の既定は DummyVecEnv（逐次）なので、真の並列化には
    #    vec_env_cls=SubprocVecEnv を明示する。n_envs=1 のときは subprocess
    #    起動コストを避けて DummyVecEnv にフォールバック。
    #    make_vec_env が内部で Monitor を適用し、統計を monitor_dir に記録する。
    #    A2C は実効バッチ = n_steps × n_envs で、n_envs を増やすほど更新が安定する。
    # -------------------------------------------------------------------------
    vec_env = make_vec_env(
        ENV_ID,
        n_envs=n_envs,
        monitor_dir=LOG_DIR,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
        # 転倒ペナルティ(-100)を緩和する報酬整形を学習用 env にのみ適用する。
        # wrapper は Monitor の外側に入るため、Monitor が記録する ep_rew_mean は
        # 素の報酬（-100 込み）のまま。緩和後の報酬は勾配にのみ効く。
        # fall_penalty == -100 のときはラッパーを付けず元の挙動にする。
        wrapper_class=FallPenaltyWrapper if fall_penalty != -100 else None,
        wrapper_kwargs={"fall_penalty": fall_penalty},
    )

    # -------------------------------------------------------------------------
    # 3. チェックポイントコールバック（一定ステップごとにモデルを保存）
    # -------------------------------------------------------------------------
    # save_freq は VecEnv では vec-step 単位で数えられるため、
    # 総タイムステップ基準の頻度を保つよう n_envs で割る。
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000 // n_envs, 1),
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
        eval_freq=max(500 // n_envs, 1),
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
        "--n-envs",
        type=int,
        default=4,
        help="並列環境数（既定: 4 / 推奨上限: 論理コア数。1 で逐次=DummyVecEnv）",
    )
    parser.add_argument(
        "--fall-penalty",
        type=float,
        default=-40.0,
        help="転倒時の報酬(-100)を緩和する置換値（既定: -40.0 / -100 で緩和なし=元の挙動）",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "play", "both"],
        default="both",
        help="train=学習のみ / play=再生のみ / both=学習して再生（既定）",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    if args.mode in ("train", "both"):
        train(args.timesteps, args.n_envs, args.fall_penalty)
    if args.mode in ("play", "both"):
        play()
    elapsed = time.perf_counter() - start_time
    print(f"経過時間: {timedelta(seconds=round(elapsed))}（{elapsed:.1f} 秒）")


if __name__ == "__main__":
    main()
