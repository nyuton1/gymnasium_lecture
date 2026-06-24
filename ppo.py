"""PPO (Proximal Policy Optimization) を BipedalWalkerHardcore-v3 で学習・再生するスクリプト。

PPO は方策勾配法の一種で、更新前後の方策比をクリッピング(clip_range)して
1 回の更新が大きくなりすぎないよう抑えることで、実装の単純さと学習の安定性を
両立させたオンポリシー手法です。連続・離散の双方で広く使われる定番アルゴリズムです。

このスクリプトは rl-baselines3-zoo の BipedalWalkerHardcore-v3 チューニング済み設定に
合わせてある（VecNormalize で観測・報酬を正規化、ent_coef/gae_lambda 設定、learning_rate と
clip_range は線形減衰、device="cpu"）。MlpPolicy の PPO は CPU が最速で、速度は
SubprocVecEnv 並列で稼ぐ。なお Hardcore は非常に難しく、PPO 単体で「解く」(平均+300)のは
事実上不可。本格学習でも歩き始め〜部分的な前進の観察が現実的なゴール。

実行例:
    python ppo.py                       # 学習(既定100万step) → ベストモデル録画・再生
    python ppo.py --timesteps 3000000   # しっかり学習（数百万〜）
    python ppo.py --mode play           # 学習をスキップし、保存済みモデルを再生
    python ppo.py --mode train          # 学習のみ（録画・再生しない）

学習過程は TensorBoard で確認できます:
    tensorboard --logdir tensorboard/
"""

import argparse
import os
import time
from datetime import timedelta
from typing import Callable, Optional

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from gym_utils import (
    FallPenaltyWrapper,
    new_run_dir,
    record_agent_video,
    resolve_model_path,
    resolve_run_dir,
)

# =============================================================================
# 設定（元 Colab ノートブック 11_ の PPO セルから移植）
# =============================================================================
ENV_ID = "BipedalWalkerHardcore-v3"                       # 使用する Gym 環境
# 1 回の学習に関わる全成果物を runs/<ALGO>/<run_id>/ に自己完結させる。
ALGO = "ppo"


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """学習進行に応じて initial_value → 0 へ線形減衰するスケジュールを返す。

    SB3 はスカラの代わりに `progress_remaining (1.0→0.0) -> 値` のコールバックを
    learning_rate / clip_range に渡せる。rl-baselines3-zoo の `lin_` プレフィックス
    （Hardcore の PPO 既定 `lin_2.5e-4` / `lin_0.2`）と同等の線形減衰を実装する。
    終盤に学習率とクリップ幅を絞ることで方策の崩壊を防ぎ収束を安定させる。
    """

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


def train(timesteps: int, n_envs: int, fall_penalty: float) -> str:
    """PPO モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。

    出力はタイムスタンプ別 run ディレクトリ runs/<ALGO>/<run_id>/ に隔離する。
    戻り値はこの run ディレクトリのパス（main() が play() に引き継ぐ）。
    """
    # -------------------------------------------------------------------------
    # 1. この学習専用の run ディレクトリ（タイムスタンプ）を作る
    # -------------------------------------------------------------------------
    run_dir = new_run_dir(ALGO)
    run_id = os.path.basename(run_dir)
    logs_dir = os.path.join(run_dir, "logs")
    best_dir = os.path.join(run_dir, "best_model")
    final_model_path = os.path.join(run_dir, "final_model")
    print(f"[run] 出力先: {os.path.abspath(run_dir)}")

    # -------------------------------------------------------------------------
    # 2. ベクトル化環境の作成（SubprocVecEnv でサブプロセス並列）
    #    make_vec_env の既定は DummyVecEnv（逐次）なので、真の並列化には
    #    vec_env_cls=SubprocVecEnv を明示する。n_envs=1 のときは subprocess
    #    起動コストを避けて DummyVecEnv にフォールバック。
    #    make_vec_env が内部で Monitor を適用し、統計を monitor_dir に記録する。
    #    PPO は実効バッチ = n_steps × n_envs で、n_envs を増やすほど更新が安定する。
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
    # VecNormalize で観測と報酬を移動平均で正規化する（rl-zoo3 の normalize: true 相当）。
    # 連続制御 PPO の成否を分ける必須級の処理。VecEnv 全体の外側に被せるため、
    # 各 env 内の Monitor は素の報酬を記録し続け、rollout/ep_rew_mean は honest なまま。
    # gamma は PPO と揃える（報酬正規化が割引リターンの分散を使うため）。
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99
    )

    # -------------------------------------------------------------------------
    # 3. チェックポイントコールバック（一定ステップごとにモデルを保存）
    # -------------------------------------------------------------------------
    # save_freq は VecEnv では vec-step 単位で数えられるため、
    # 総タイムステップ基準の頻度を保つよう n_envs で割る。
    # 本格学習（数百万ステップ）ではチェックポイントが溜まりすぎないよう ~1M ごとにする
    # （best_model / final_model が主役なのでチェックポイントは粗くてよい）。
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1_000_000 // n_envs, 1),
        save_path=logs_dir,
        name_prefix="ppo_model",
    )

    # -------------------------------------------------------------------------
    # 4. 評価環境と評価コールバック
    #    定期的に別環境で評価し、最高性能のモデルを best_model として保存する。
    # -------------------------------------------------------------------------
    def make_eval_env():
        return Monitor(gym.make(ENV_ID, render_mode="rgb_array"))

    # 評価 env も VecNormalize で包むが、norm_reward=False / training=False にする。
    # EvalCallback が評価直前に sync_envs_normalization で学習 env の観測統計を同期するため、
    # 観測は学習時と同じ分布に正規化されつつ、報酬は素のまま＝best 選定と eval/mean_reward を
    # 真の性能で行える（best_model を honest な指標で選ぶ）。
    eval_env = VecNormalize(
        DummyVecEnv([make_eval_env]),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
        training=False,
    )
    # 本格学習向けに評価頻度を ~10万ステップごとに緩め、1 回の評価エピソード数を増やして
    # （Hardcore は報酬の分散が大きい）best 選定のノイズを抑える。
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=logs_dir,
        eval_freq=max(100_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # -------------------------------------------------------------------------
    # 5. PPO モデルの構築（rl-baselines3-zoo の BipedalWalkerHardcore-v3 チューニング済み値）
    #    MlpPolicy: ベクトル観測用 / n_steps: 各 env が更新前に集める収集ステップ数
    #    batch_size: 勾配更新時のミニバッチ / n_epochs: 同じロールアウトの反復回数
    #    gae_lambda: GAE の平滑化 / ent_coef: 探索を促す少量のエントロピー項
    #    learning_rate / clip_range: 学習終盤に向け線形に 0 へ減衰（lin_ スケジュール）
    #    device="cpu": MlpPolicy の PPO は CPU が最速（SB3 公式・実測）。Mac の MPS は
    #    不具合で失敗するため使わない。並列化（SubprocVecEnv）で速度を稼ぐ。
    #    vf_coef / max_grad_norm / policy_kwargs は SB3 既定（=zoo も未指定）。
    # -------------------------------------------------------------------------
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=linear_schedule(2.5e-4),
        clip_range=linear_schedule(0.2),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.001,
        device="cpu",
        verbose=1,
        tensorboard_log="tensorboard/",
    )

    # -------------------------------------------------------------------------
    # 6. 学習 → 最終モデル保存 → 後片付け
    # -------------------------------------------------------------------------
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=f"{ALGO.upper()}_{run_id}",
    )
    model.save(final_model_path)
    # 観測・報酬の正規化統計を保存する。play() はこの統計で観測を正規化して再生する
    # （統計を合わせないと学習時と入力分布がずれて方策が正しく動かない）。
    vec_env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    vec_env.close()
    eval_env.close()
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
    agent = PPO.load(model_path)
    # 学習時の VecNormalize 統計があれば、観測を同じ統計で正規化してから推論する。
    # 無い run（旧構造や正規化なしの学習）では None のまま＝素の観測で従来どおり再生。
    obs_transform = None
    stats_path = os.path.join(run_dir, "vecnormalize.pkl")
    if os.path.exists(stats_path):
        vn = VecNormalize.load(stats_path, DummyVecEnv([lambda: gym.make(ENV_ID)]))
        vn.training = False        # 統計を更新しない
        vn.norm_reward = False     # 報酬は素のまま（再生表示用）
        obs_transform = vn.normalize_obs
        print(f"[play] VecNormalize 統計を適用: {os.path.abspath(stats_path)}")
    record_agent_video(
        agent, ENV_ID, video_folder, deterministic=True, obs_transform=obs_transform
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO を BipedalWalkerHardcore-v3 で学習・再生")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help=(
            "総学習ステップ数（PPO 既定: 1,000,000 = 意味のある最小規模。"
            "本格学習は数百万〜。他スクリプトの動作確認既定 2000 とは異なる）"
        ),
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="並列環境数（PPO 既定: 8 / 推奨上限: 物理コア数。1 で逐次=DummyVecEnv）",
    )
    parser.add_argument(
        "--fall-penalty",
        type=float,
        default=-40.0,
        help="転倒時の報酬(-100)を緩和する置換値（既定: -40.0 / -100 で緩和なし=元の挙動）",
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
        trained_run = train(args.timesteps, args.n_envs, args.fall_penalty)
    if args.mode in ("play", "both"):
        play(args.run if args.run is not None else trained_run)
    elapsed = time.perf_counter() - start_time
    print(f"経過時間: {timedelta(seconds=round(elapsed))}（{elapsed:.1f} 秒）")


if __name__ == "__main__":
    main()
