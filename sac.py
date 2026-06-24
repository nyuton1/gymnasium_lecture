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
``sac_runs_bipedalwalkerhardcore/<YYYYMMDD-HHMMSS>/`` に隔離して保存されるため、
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
import glob
import os
import time
from datetime import datetime, timedelta
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

from gym_utils import FallPenaltyWrapper, RecordBestVideoCallback, record_agent_video

# =============================================================================
# 設定（元 Colab ノートブックの SAC セルから移植）
# =============================================================================
ENV_ID = "BipedalWalkerHardcore-v3"
# 各学習を上書きせずタイムスタンプ別フォルダに隔離する run ルート。
# 1 回の学習 = RUN_ROOT/<YYYYMMDD-HHMMSS>/ 配下に
# logs / best_model / videos/{progress,play} / final_model.zip をまとめる。
RUN_ROOT = "sac_runs_bipedalwalkerhardcore"
# 以下は旧構造（固定パス）で学習した成果物を play() で再生するためのレガシー・
# フォールバック先。新しい学習はすべて RUN_ROOT 配下に書き出す。
LOG_DIR = "./sac_logs_bipedalwalkerhardcore/"
VIDEO_FOLDER = "sac_bipedalwalkerhardcore_videos_practice"
FINAL_MODEL = "sac_bipedalwalkerhardcore"


def train(
    timesteps: int,
    n_envs: int,
    fall_penalty: float,
    progress_video_every: int,
    progress_video: bool,
) -> str:
    """SAC モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。

    出力はタイムスタンプ別の run ディレクトリ ``RUN_ROOT/<YYYYMMDD-HHMMSS>/`` に
    隔離するため、同じ設定で再学習しても過去の成果物を上書きしない。
    戻り値はこの run ディレクトリのパス（main() が play() に引き継ぐ）。
    """
    # -------------------------------------------------------------------------
    # 0. この学習専用の run ディレクトリ（タイムスタンプ）を作る
    # -------------------------------------------------------------------------
    # 並列に複数 run を起動しても衝突しないよう PID を付けて run_id を一意化する
    # （タイムスタンプが同じ秒でも PID で区別され、辞書順ソートの「最新」判定も保たれる）。
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    run_dir = os.path.join(RUN_ROOT, run_id)
    logs_dir = os.path.join(run_dir, "logs")        # monitor / checkpoint / results
    best_dir = os.path.join(run_dir, "best_model")  # best_model.zip
    progress_dir = os.path.join(run_dir, "videos", "progress")  # 進捗動画
    final_model_path = os.path.join(run_dir, "final_model")     # -> final_model.zip
    os.makedirs(logs_dir, exist_ok=True)
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
        tb_log_name=f"SAC_{run_id}",
    )
    model.save(final_model_path)
    vec_env.close()
    eval_env.close()
    print(f"[run] 最終モデル: {os.path.abspath(final_model_path)}.zip")
    return run_dir


def _latest_run_dir() -> Optional[str]:
    """RUN_ROOT 内で最新（タイムスタンプ名の辞書順で最大）の run ディレクトリを返す。

    タイムスタンプは ``YYYYMMDD-HHMMSS`` 形式で辞書順 = 時系列順になるため、
    sorted の末尾が最新の run。RUN_ROOT が無い／空なら None。
    """
    if not os.path.isdir(RUN_ROOT):
        return None
    runs = sorted(d for d in glob.glob(os.path.join(RUN_ROOT, "*")) if os.path.isdir(d))
    return runs[-1] if runs else None


def play(run: Optional[str] = None) -> None:
    """保存済みモデルをロードし、1エピソードを録画して再生する。

    モデルの解決順:
      1. 引数 run（run id またはパス）で指定された run
      2. RUN_ROOT 内の最新 run
      3. レガシー固定パス（旧構造で学習した成果物の互換フォールバック）
    各 run 内では best_model → final_model の順にフォールバックする。
    """
    run_dir = None
    if run is not None:
        run_dir = run if os.path.isdir(run) else os.path.join(RUN_ROOT, run)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"指定された run が見つかりません: {run}")
    else:
        run_dir = _latest_run_dir()

    if run_dir is not None:
        best_model_path = os.path.join(run_dir, "best_model", "best_model")
        final_in_run = os.path.join(run_dir, "final_model")
        if os.path.exists(best_model_path + ".zip"):
            model_path = best_model_path
        elif os.path.exists(final_in_run + ".zip"):
            model_path = final_in_run
        else:
            raise FileNotFoundError(f"run 内に学習済みモデルがありません: {run_dir}")
        video_folder = os.path.join(run_dir, "videos", "play")
    else:
        # レガシー固定パス（旧構造で学習した成果物）にフォールバック
        legacy_best = os.path.join(LOG_DIR, "best_model", "best_model")
        if os.path.exists(legacy_best + ".zip"):
            model_path = legacy_best
        elif os.path.exists(FINAL_MODEL + ".zip"):
            model_path = FINAL_MODEL
        else:
            raise FileNotFoundError(
                "学習済みモデルが見つかりません。先に `python sac.py --mode train` を実行してください。"
            )
        video_folder = VIDEO_FOLDER

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
