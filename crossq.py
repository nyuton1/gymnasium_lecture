"""CrossQ を BipedalWalkerHardcore-v3 で学習・再生するスクリプト。

CrossQ は「Batch Normalization in Deep Reinforcement Learning」由来のオフポリシー手法
（sb3_contrib 2.4.0 以降）。クリティックに Batch Normalization を導入し、**ターゲット
ネットワークを撤廃**することで、低い update-to-data 比でも安定した価値推定を可能にし、
SAC/TQC より高いサンプル効率を狙います。学習エピソード数に制約がある本課題
（5000 エピソード以内）で有利になりうる候補として TQC と比較するために用意しました。
このスクリプトの狙いは**最も速くゴールする**方策で、学習用 env にのみ前進速度ボーナス
（SpeedRewardWrapper）を加えてゴール所要時間を縮める方向へ誘導します。

実行例:
    python crossq.py                       # 学習(2000step) → 録画・再生＋所要時間計測
    python crossq.py --timesteps 1000000   # しっかり学習
    python crossq.py --max-episodes 5000   # 5000 エピソードに達したら学習を打ち切る
    python crossq.py --speed-coef 0.3      # 前進速度ボーナス係数（0 で速度ボーナス無効）
    python crossq.py --mode play           # 学習をスキップし、最新 run のモデルを再生・計測
    python crossq.py --no-progress-video   # 学習中の進捗動画録画を無効化

各学習は上書きを避けてタイムスタンプ別フォルダ
``runs/crossq/<YYYYMMDD-HHMMSS-pid>/`` に隔離して保存されます。

注意:
    learning_starts=10000 のため、--timesteps を 10000 以下にすると勾配更新が一度も
    走りません。パイプライン確認のスモークは --timesteps 12000 以上で行ってください。
    意味のある方策には数百万ステップ規模が必要です。

学習過程は TensorBoard で確認できます:
    tensorboard --logdir tensorboard/
"""

import argparse
import os
import time
from datetime import timedelta
from typing import Optional

import gymnasium as gym

try:
    from sb3_contrib import CrossQ
except ImportError as exc:  # CrossQ は sb3-contrib 2.4.0 以降で追加
    raise ImportError(
        "CrossQ をインポートできません。CrossQ は sb3-contrib 2.4.0 以降が必要です。"
        "`pip install -U sb3-contrib`、または requirements.txt の "
        "sb3-contrib==2.4.0 が入っているか確認してください。"
    ) from exc

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
    measure_goal_time,
    new_run_dir,
    record_agent_video,
    resolve_model_path,
    resolve_run_dir,
)

# =============================================================================
# 設定（sac.py / tqc.py の構造を踏襲。アルゴリズム固有部分のみ差し替え）
# =============================================================================
ENV_ID = "BipedalWalkerHardcore-v3"
# 1 回の学習に関わる全成果物を runs/<ALGO>/<run_id>/ に自己完結させる。
ALGO = "crossq"


def train(
    timesteps: int,
    n_envs: int,
    fall_penalty: float,
    speed_coef: float,
    max_episodes: int,
    progress_video_every: int,
    progress_video: bool,
) -> str:
    """CrossQ モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。

    出力はタイムスタンプ別の run ディレクトリ ``runs/<ALGO>/<run_id>/`` に
    隔離するため、同じ設定で再学習しても過去の成果物を上書きしない。
    戻り値はこの run ディレクトリのパス（main() が play() に引き継ぐ）。
    """
    # -------------------------------------------------------------------------
    # 0. この学習専用の run ディレクトリ（タイムスタンプ）を作る
    # -------------------------------------------------------------------------
    run_dir = new_run_dir(ALGO)
    run_id = os.path.basename(run_dir)
    logs_dir = os.path.join(run_dir, "logs")
    best_dir = os.path.join(run_dir, "best_model")
    progress_dir = os.path.join(run_dir, "videos", "progress")
    final_model_path = os.path.join(run_dir, "final_model")
    print(f"[run] 出力先: {os.path.abspath(run_dir)}")

    # -------------------------------------------------------------------------
    # 1. 学習用環境（並列収集）と評価用環境
    #    報酬整形（転倒罰の緩和 + 前進速度ボーナス）は学習用 env にのみ適用する。
    #    eval_env と play() は素の報酬を使うため、best_model 選定と所要時間計測は
    #    真の性能を反映する。fall_penalty==-100 かつ speed_coef==0 なら整形しない。
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000 // n_envs, 1),
        save_path=logs_dir,
        name_prefix="crossq_model",
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
    # _total_max_episodes = max_episodes * num_envs と解釈する（max_episodes は
    # 1 env あたり）。総エピソード数を max_episodes に保つため n_envs で割って渡す。
    stop_callback = StopTrainingOnMaxEpisodes(
        max_episodes=max(1, max_episodes // n_envs), verbose=1
    )
    callbacks = CallbackList([checkpoint_callback, eval_callback, stop_callback])

    # -------------------------------------------------------------------------
    # 3. CrossQ モデルの構築
    #    RL-Baselines3-Zoo の BipedalWalker-v3 CrossQ レシピをハードモード向けに調整:
    #    非対称な net_arch（方策 [256,256] / 批評家 [1024,1024]）、buffer_size は
    #    300k→1M に拡大。CrossQ はターゲットネットを撤廃し BatchNorm を使うため、
    #    tau / train_freq / gradient_steps / ent_coef / learning_rate は SAC とは
    #    意味が異なる。ここでは渡さず CrossQ の既定値に委ねる（最小引数で頑健に）。
    # -------------------------------------------------------------------------
    model = CrossQ(
        policy="MlpPolicy",
        env=vec_env,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        learning_starts=10000,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[1024, 1024])),
        verbose=1,
        tensorboard_log="tensorboard/",
    )

    # -------------------------------------------------------------------------
    # 4. 学習 → 最終モデル保存 → 後片付け
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
    agent = CrossQ.load(model_path)
    record_agent_video(agent, ENV_ID, video_folder, deterministic=True)
    # ゴールまでの所要時間（ステップ数→秒）と成功率を計測して表示する。
    measure_goal_time(agent, ENV_ID, n_episodes=10, deterministic=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CrossQ を BipedalWalkerHardcore-v3 で学習・再生")
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
