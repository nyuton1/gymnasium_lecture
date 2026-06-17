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
import time
from datetime import timedelta

import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_utils import FallPenaltyWrapper, record_agent_video

# =============================================================================
# 設定（元 Colab ノートブックの DDPG セルから移植）
# =============================================================================
ENV_ID = "BipedalWalkerHardcore-v3"
LOG_DIR = "./ddpg_logs_bipedalwalkerhardcore/"
VIDEO_FOLDER = "ddpg_bipedalwalkerhardcore_videos_practice"
FINAL_MODEL = "ddpg_bipedalwalkerhardcore"


def train(timesteps: int, n_envs: int, fall_penalty: float) -> None:
    """DDPG モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。"""
    os.makedirs(LOG_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. 学習用環境（SubprocVecEnv で並列収集）と評価用環境
    #    学習に描画は不要なので render_mode は付けない（高速化）。
    #    n_envs=1 のときは subprocess 起動コストを避けて DummyVecEnv にフォールバック。
    #    make_vec_env が各 env を Monitor で自動ラップし ep_rew_mean を記録する。
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
    eval_env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="rgb_array")])

    # -------------------------------------------------------------------------
    # 2. コールバック（チェックポイント保存 + 定期評価でベストモデル保存）
    # -------------------------------------------------------------------------
    # save_freq/eval_freq は VecEnv では vec-step 単位で数えられるため、
    # 総タイムステップ基準の頻度を保つよう n_envs で割る。
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000 // n_envs, 1),
        save_path=LOG_DIR,
        name_prefix="ddpg_model",
    )
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
    # 3. 行動ノイズ（探索用。平均0・標準偏差0.1の正規分布ノイズ）
    # -------------------------------------------------------------------------
    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    # -------------------------------------------------------------------------
    # 4. DDPG モデルの構築
    # -------------------------------------------------------------------------
    #    gradient_steps=-1: 収集ステップ数(=n_envs)ぶん勾配更新し、並列化しても
    #    更新比 1:1 を維持する（n_envs=1 なら 1 回更新で従来と等価）。
    model = DDPG(
        policy="MlpPolicy",
        env=vec_env,
        action_noise=action_noise,
        gamma=0.99,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=1_000,
        train_freq=(1, "step"),
        gradient_steps=-1,
        tensorboard_log="tensorboard/",
        verbose=1,
    )

    # -------------------------------------------------------------------------
    # 5. 学習 → 最終モデル保存 → 後片付け
    # -------------------------------------------------------------------------
    model.learn(total_timesteps=timesteps, callback=callbacks)
    model.save(FINAL_MODEL)
    vec_env.close()
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
