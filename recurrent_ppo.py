"""RecurrentPPO を BipedalWalkerHardcore-v3 で学習・再生するスクリプト。

RecurrentPPO は PPO の方策・価値関数に LSTM（再帰結合）を組み込んだオンポリシー手法です。
過去の観測を隠れ状態として保持できるため、観測だけでは現在の状態が定まらない
部分観測(POMDP)や、時系列依存のあるタスクに強いのが特長です。
（stable-baselines3 本体ではなく拡張パッケージ sb3_contrib に実装されています）

再生（play）では、ステップ間で LSTM の隠れ状態を引き継ぐ必要があるため、
gym_utils.record_agent_video を recurrent=True で呼び出します（毎ステップ隠れ状態を
ゼロに戻すと再帰方策が正しく動かないため）。

実行例:
    python recurrent_ppo.py                      # 学習(2000step) → ベストモデル録画・再生
    python recurrent_ppo.py --timesteps 50000    # 学習ステップ数を増やす
    python recurrent_ppo.py --mode play          # 学習をスキップし、保存済みモデルを再生
    python recurrent_ppo.py --mode train         # 学習のみ（録画・再生しない）

学習過程は TensorBoard で確認できます:
    tensorboard --logdir tensorboard/
"""

import argparse
import os
import time
from datetime import timedelta

import gymnasium as gym
from sb3_contrib import RecurrentPPO
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
# 設定（元 Colab ノートブック 11_ の RecurrentPPO セルから移植）
# =============================================================================
ENV_ID = "BipedalWalkerHardcore-v3"                                  # 使用する Gym 環境
LOG_DIR = "./recurrent_ppo_logs_bipedalwalkerhardcore/"              # ログ・モデルの保存先
VIDEO_FOLDER = "recurrent_ppo-bipedalwalkerhardcore_videos_practice"  # 再生動画の保存先
FINAL_MODEL = "recurrent_ppo_bipedalwalkerhardcore"                 # 学習後の最終モデル保存名(.zip)


def train(timesteps: int, n_envs: int, fall_penalty: float) -> None:
    """RecurrentPPO モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。"""
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
    #    RecurrentPPO は実効バッチ = n_steps × n_envs で、n_envs を増やすほど更新が安定する。
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
        name_prefix="recurrent_ppo_model",
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
    # 5. RecurrentPPO モデルの構築
    #    MlpLstmPolicy: ベクトル観測 + LSTM 隠れ状態を使う再帰方策
    #    n_steps: 各 env が更新前に集める収集ステップ数（シーケンス長）
    #    batch_size: 勾配更新時のミニバッチ / n_epochs: 同じロールアウトの反復回数
    #    clip_range: 方策比のクリッピング幅（PPO の肝）
    # -------------------------------------------------------------------------
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        gamma=0.99,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
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
            "学習済みモデルが見つかりません。先に `python recurrent_ppo.py --mode train` を実行してください。"
        )

    agent = RecurrentPPO.load(model_path)
    # RecurrentPPO は再生時に LSTM 隠れ状態を引き継ぐ必要があるため recurrent=True。
    record_agent_video(agent, ENV_ID, VIDEO_FOLDER, deterministic=True, recurrent=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="RecurrentPPO を BipedalWalkerHardcore-v3 で学習・再生")
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
