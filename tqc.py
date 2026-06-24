"""TQC (Truncated Quantile Critics) を BipedalWalkerHardcore-v3 で学習・再生するスクリプト。

TQC は SAC を「分布型クリティック（複数の分位点で価値分布を推定し、上位の分位点を
切り捨てて過大評価を抑える）」へ拡張したオフポリシー手法です（sb3_contrib 提供）。
RL-Baselines3-Zoo では本環境（BipedalWalkerHardcore-v3）の実績がトップクラスで、
高いサンプル効率と安定性から「ハードモードを解く」定番アルゴリズムとして知られます。
このスクリプトの狙いは**最も速くゴールする**方策で、学習用 env にのみ前進速度ボーナス
（SpeedRewardWrapper）を加えてゴール所要時間を縮める方向へ誘導します。

実行例:
    python tqc.py                       # 学習(2000step) → ベストモデル録画・再生＋所要時間計測
    python tqc.py --timesteps 2000000   # しっかり学習（RL-Zoo 既定は 2e6 step）
    python tqc.py --max-episodes 5000   # 5000 エピソードに達したら学習を打ち切る
    python tqc.py --speed-coef 0.3      # 前進速度ボーナス係数（0 で速度ボーナス無効）
    python tqc.py --mode play           # 学習をスキップし、最新 run のモデルを再生・計測
    python tqc.py --no-progress-video   # 学習中の進捗動画録画を無効化

各学習は上書きを避けてタイムスタンプ別フォルダ
``runs/tqc/<YYYYMMDD-HHMMSS-pid>/`` に隔離して保存されます。

注意:
    learning_starts=10000（SAC の 3000 より大きい）ため、--timesteps を 10000 以下に
    すると勾配更新が一度も走りません。パイプライン確認のスモークは --timesteps 12000
    以上で行ってください。意味のある方策には数百万ステップ規模が必要です。

学習過程は TensorBoard で確認できます:
    tensorboard --logdir tensorboard/
"""

import argparse
import os
import time
from datetime import timedelta
from typing import Optional

import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnMaxEpisodes,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_utils import (
    RecordBestVideoCallback,
    SpeedRewardWrapper,
    linear_schedule,
    measure_goal_time,
    new_run_dir,
    record_agent_video,
    resolve_model_path,
    resolve_run_dir,
)

# =============================================================================
# 設定（sac.py の構造を踏襲。アルゴリズム固有部分のみ差し替え）
# =============================================================================
ENV_ID = "BipedalWalkerHardcore-v3"
# 1 回の学習に関わる全成果物を runs/<ALGO>/<run_id>/ に自己完結させる。
# run ディレクトリの作成・解決は gym_utils のヘルパーに集約。
ALGO = "tqc"


def train(
    timesteps: int,
    n_envs: int,
    fall_penalty: float,
    speed_coef: float,
    max_episodes: int,
    progress_video_every: int,
    progress_video: bool,
) -> str:
    """TQC モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。

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
    #    n_envs=1 のときは subprocess 起動コストを避けて DummyVecEnv にフォールバック。
    #    make_vec_env が各 env を Monitor で自動ラップし ep_rew_mean を記録する。
    # -------------------------------------------------------------------------
    # 報酬整形（転倒罰の緩和 + 前進速度ボーナス）を学習用 env にのみ適用する。
    # wrapper は Monitor の外側に入るため、Monitor が記録する ep_rew_mean は素の報酬の
    # まま（整形後の報酬は勾配にのみ効く）。eval_env と play() は素の報酬を使うので、
    # best_model 選定と所要時間計測は真の性能を反映する。
    # fall_penalty == -100 かつ speed_coef == 0 のときはラッパーを付けず元の挙動にする。
    apply_wrapper = (fall_penalty != -100) or (speed_coef != 0)
    vec_env = make_vec_env(
        ENV_ID,
        n_envs=n_envs,
        monitor_dir=logs_dir,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
        wrapper_class=SpeedRewardWrapper if apply_wrapper else None,
        wrapper_kwargs={"fall_penalty": fall_penalty, "speed_coef": speed_coef},
    )
    eval_env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="rgb_array")])

    # -------------------------------------------------------------------------
    # 2. コールバック（チェックポイント保存 + 定期評価 + エピソード数で打ち切り）
    #    save_freq/eval_freq は VecEnv では vec-step 単位のため n_envs で割る。
    # -------------------------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000 // n_envs, 1),
        save_path=logs_dir,
        name_prefix="tqc_model",
    )
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
        n_eval_episodes=3,
        callback_on_new_best=new_best_callback,
        deterministic=True,
        render=False,
    )
    # 「5000 エピソード以内」を守るための打ち切り。StopTrainingOnMaxEpisodes は
    # _init_callback で _total_max_episodes = max_episodes * num_envs と解釈する
    # （= max_episodes は「1 env あたり」）。総エピソード数を max_episodes に保つため
    # n_envs で割って渡す（max(1, ...) は n_envs > max_episodes の保険）。
    stop_callback = StopTrainingOnMaxEpisodes(
        max_episodes=max(1, max_episodes // n_envs), verbose=1
    )
    callbacks = CallbackList([checkpoint_callback, eval_callback, stop_callback])

    # -------------------------------------------------------------------------
    # 3. TQC モデルの構築（RL-Baselines3-Zoo の BipedalWalkerHardcore-v3 レシピ）
    #    learning_rate は線形減衰（lin_7.3e-4）/ net_arch=[400,300] / tau=0.01。
    #    gradient_steps=1（収集ステップごとに 1 回更新）。learning_starts=10000。
    # -------------------------------------------------------------------------
    model = TQC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=linear_schedule(7.3e-4),
        buffer_size=1_000_000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.01,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000,
        policy_kwargs=dict(net_arch=[400, 300]),
        verbose=1,
        tensorboard_log="tensorboard/",
    )

    # -------------------------------------------------------------------------
    # 4. 学習 → 最終モデル保存 → 後片付け
    #    本番は --timesteps を大きく（例 2_000_000）し、--max-episodes(=5000) を
    #    実質の打ち切り条件にする。スモークは小さい --timesteps が先に止める。
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
    """保存済みモデルをロードし、1エピソードを録画・再生し、ゴール所要時間を計測する。

    run（id/パス。既定は最新 run）を解決し、run 内では best_model → final_model の
    順にフォールバックする。再生後に measure_goal_time で所要時間（秒）と成功率を表示する。
    """
    run_dir = resolve_run_dir(ALGO, run)
    model_path = resolve_model_path(run_dir)
    video_folder = os.path.join(run_dir, "videos", "play")
    print(f"[play] モデル: {os.path.abspath(model_path)}.zip")
    agent = TQC.load(model_path)
    record_agent_video(agent, ENV_ID, video_folder, deterministic=True)
    # ゴールまでの所要時間（ステップ数→秒）と成功率を計測して表示する。
    measure_goal_time(agent, ENV_ID, n_episodes=10, deterministic=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="TQC を BipedalWalkerHardcore-v3 で学習・再生")
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
        "--speed-coef",
        type=float,
        default=0.3,
        help="前進速度ボーナス係数（obs[2]>0 のとき speed_coef*obs[2] を加算。既定 0.3 / 0 で無効）",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=5000,
        help="学習を止める総エピソード数の上限（既定 5000。内部で // n_envs して per-env 値に換算）",
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
            args.speed_coef,
            args.max_episodes,
            args.progress_video_every,
            not args.no_progress_video,
        )
    if args.mode in ("play", "both"):
        play(args.run if args.run is not None else trained_run)
    elapsed = time.perf_counter() - start_time
    print(f"経過時間: {timedelta(seconds=round(elapsed))}（{elapsed:.1f} 秒）")


if __name__ == "__main__":
    main()
