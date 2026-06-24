"""SAC (Soft Actor-Critic) を BipedalWalkerHardcore-v3 で学習・再生するスクリプト。

SAC は「報酬の最大化」に加えて「方策のエントロピー最大化」を目的に組み込んだ
オフポリシー型アクター–クリティック手法です。確率的な方策とエントロピー項により
探索と活用のバランスを自動調整し、高いサンプル効率とハイパーパラメータへの
頑健性を実現します。

実行例:
    python sac.py                      # 学習(2000step) → ベストモデル録画・再生
    python sac.py --timesteps 50000    # 学習ステップ数を増やす（元ノートブック値）
    python sac.py --mode play          # 学習をスキップし、最新 run のモデルを再生
    python sac.py --mode train         # 学習のみ（録画・再生しない）
    python sac.py --progress-video-every 20000  # best更新時の進捗動画を最低2万step間隔で録画
    python sac.py --no-progress-video  # 学習中の進捗動画録画を無効化

各学習は上書きを避けてタイムスタンプ別フォルダ
``runs/sac/<YYYYMMDD-HHMMSS-pid>/`` に隔離して保存されるため、
同じ設定で何度学習しても過去の成果（best_model / 最終モデル / 動画）が消えません。

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
import time
from datetime import timedelta
from typing import Optional

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_utils import (
    FallPenaltyWrapper,
    RecordBestVideoCallback,
    new_run_dir,
    record_agent_video,
    resolve_model_path,
    resolve_run_dir,
)

# =============================================================================
# 設定（元 Colab ノートブックの SAC セルから移植）
# =============================================================================
ENV_ID = "BipedalWalkerHardcore-v3"
# 1 回の学習に関わる全成果物を runs/<ALGO>/<run_id>/ に自己完結させる。
# run ディレクトリの作成・解決は gym_utils のヘルパーに集約。
ALGO = "sac"


def train(
    timesteps: int,
    n_envs: int,
    fall_penalty: float,
    progress_video_every: int,
    progress_video: bool,
) -> str:
    """SAC モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。

    出力はタイムスタンプ別の run ディレクトリ ``runs/<ALGO>/<run_id>/`` に
    隔離するため、同じ設定で再学習しても過去の成果物を上書きしない。
    戻り値はこの run ディレクトリのパス（main() が play() に引き継ぐ）。
    """
    # -------------------------------------------------------------------------
    # 0. この学習専用の run ディレクトリ（タイムスタンプ）を作る
    # -------------------------------------------------------------------------
    run_dir = new_run_dir(ALGO)
    run_id = os.path.basename(run_dir)
    logs_dir = os.path.join(run_dir, "logs")        # monitor / checkpoint / results
    best_dir = os.path.join(run_dir, "best_model")  # best_model.zip
    progress_dir = os.path.join(run_dir, "videos", "progress")  # 進捗動画
    final_model_path = os.path.join(run_dir, "final_model")     # -> final_model.zip
    print(f"[run] 出力先: {os.path.abspath(run_dir)}")

    # -------------------------------------------------------------------------
    # 1. 学習用環境（SubprocVecEnv で並列収集）と評価用環境
    #    学習に描画は不要なので render_mode は付けない（高速化）。
    #    n_envs=1 のときは subprocess 起動コストを避けて DummyVecEnv にフォールバック。
    #    make_vec_env が各 env を Monitor で自動ラップし ep_rew_mean を記録する。
    # -------------------------------------------------------------------------
    vec_env = make_vec_env(
        ENV_ID,
        n_envs=n_envs,
        monitor_dir=logs_dir,
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
        save_path=logs_dir,
        name_prefix="sac_model",
    )
    # best_model が更新された瞬間にその時点のモデルから進捗動画を録画する
    # （EvalCallback の callback_on_new_best として発火）。早期は best が頻繁に
    # 更新されるため min_interval_steps で間引く。--no-progress-video で無効化。
    new_best_callback = (
        RecordBestVideoCallback(
            ENV_ID,
            progress_dir,
            min_interval_steps=progress_video_every,
            deterministic=True,
            verbose=1,
        )
        if progress_video
        else None
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=logs_dir,
        eval_freq=max(500 // n_envs, 1),
        n_eval_episodes=3,  # 既定5から軽量化（評価オーバーヘッド削減。best選定はやや粗く）
        callback_on_new_best=new_best_callback,
        deterministic=True,
        render=False,
    )
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # -------------------------------------------------------------------------
    # 3. SAC モデルの構築
    #    ent_coef='auto': エントロピー係数を自動調整 / オフポリシー + リプレイバッファ
    # -------------------------------------------------------------------------
    #    gradient_steps=-1: 収集ステップ数(=n_envs)ぶん勾配更新し、並列化しても
    #    更新比 1:1 を維持する（n_envs=1 なら 1 回更新で従来と等価）。
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        gamma=0.99,
        learning_rate=0.00035,
        buffer_size=1_000_000,
        learning_starts=3000,
        train_freq=(1, "step"),
        gradient_steps=-1,
        ent_coef="auto",
        target_update_interval=1,
        verbose=1,
        tensorboard_log="tensorboard/",
    )

    # -------------------------------------------------------------------------
    # 4. 学習 → 最終モデル保存 → 後片付け
    #    tb_log_name に run_id を使い TensorBoard 上で run を識別できるようにする。
    # -------------------------------------------------------------------------
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=f"{ALGO.upper()}_{run_id}",
    )
    model.save(final_model_path)
    vec_env.close()
    eval_env.close()
    print(f"[run] 最終モデル: {os.path.abspath(final_model_path)}.zip")
    return run_dir


def play(run: Optional[str] = None) -> None:
    """保存済みモデルをロードし、1エピソードを録画して再生する。

    run（id/パス）指定が無ければ runs/<ALGO>/ の最新 run を使い、
    run 内では best_model → final_model の順にフォールバックする。
    """
    run_dir = resolve_run_dir(ALGO, run)
    model_path = resolve_model_path(run_dir)
    video_folder = os.path.join(run_dir, "videos", "play")
    print(f"[play] モデル: {os.path.abspath(model_path)}.zip")
    agent = SAC.load(model_path)
    record_agent_video(agent, ENV_ID, video_folder, deterministic=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAC を BipedalWalkerHardcore-v3 で学習・再生")
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
        "--progress-video-every",
        type=int,
        default=20000,
        help="学習中、best更新時に進捗動画を録る最低step間隔（既定: 20000 / 0 で更新ごと毎回）",
    )
    parser.add_argument(
        "--no-progress-video",
        action="store_true",
        help="学習中の進捗動画録画を無効化する",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="--mode play で再生する run（id またはパス。既定: 最新 run）",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "play", "both"],
        default="both",
        help="train=学習のみ / play=再生のみ / both=学習して再生（既定）",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    trained_run = None
    if args.mode in ("train", "both"):
        trained_run = train(
            args.timesteps,
            args.n_envs,
            args.fall_penalty,
            args.progress_video_every,
            not args.no_progress_video,
        )
    if args.mode in ("play", "both"):
        # both のときは今学習した run を、play 単独のときは --run（無ければ最新）を再生
        play(args.run if args.run is not None else trained_run)
    elapsed = time.perf_counter() - start_time
    print(f"経過時間: {timedelta(seconds=round(elapsed))}（{elapsed:.1f} 秒）")


if __name__ == "__main__":
    main()
