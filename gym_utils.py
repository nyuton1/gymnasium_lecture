"""gym_utils — 学習済みエージェントの動作を録画・再生するためのユーティリティ。

このモジュールは、元の Colab ノートブックが Google Drive から取得していた
`gym_utils.py`（`display_video` 関数）を **ローカル実行向けに再作成** したものです。

主な公開関数・クラス:
    display_video(env, video_folder)
        指定フォルダに保存された動画(mp4)を表示する。
        - Jupyter / IPython カーネル上では HTML5 <video> として埋め込み表示。
        - 通常の Python スクリプト実行時は保存先パスを表示し、
          可能なら OS 既定のプレーヤで動画を開く。

    record_agent_video(agent, env_id, video_folder, deterministic=True)
        学習済みエージェントを 1 エピソード動かして動画を録画し、
        最後に display_video で表示する。
        （各アルゴリズムスクリプトで共通だった「再生ループ」を集約したもの）

    FallPenaltyWrapper(env, fall_penalty=-40.0)
        転倒時の報酬 -100 を小さい値に置き換える報酬整形ラッパー。
        各学習スクリプトが make_vec_env(wrapper_class=...) で学習用 env にのみ適用する。

    RecordBestVideoCallback(env_id, video_folder, min_interval_steps=0)
        EvalCallback の callback_on_new_best として使い、ベストモデル更新時に
        その時点のモデルから進捗動画を録画して履歴に残す SB3 コールバック。

元ノートブックとの互換性のため、`display_video(env, video_folder)` の
シグネチャ（第1引数に環境/ラッパー、第2引数に動画フォルダ）を維持しています。
第1引数は本実装では使用しませんが、互換性のために受け取ります。
"""

from __future__ import annotations

import base64
import glob
import os
import subprocess
import sys
from datetime import datetime
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class FallPenaltyWrapper(gym.Wrapper):
    """転倒時の報酬 -100 を緩和する報酬整形(reward shaping)ラッパー。

    BipedalWalker(Hardcore) は胴体が地面に触れて転倒すると、その 1 ステップで
    報酬がちょうど -100 に置き換えられてエピソードが終了する（gymnasium の
    ``bipedal_walker.py``: ``if self.game_over or pos[0] < 0: reward = -100``）。
    この -100 は前進報酬（1 エピソード累計でも +300 程度）に比べて一度に入る罰が
    大きすぎ、「動かない方がマシ」とエージェントが萎縮して学習が進みにくくなる。

    そこで転倒ステップの報酬(-100)を、より小さい ``fall_penalty`` に置き換える。
    -100 はこの転倒時にしか発生しないため ``reward <= -100`` で転倒を判定でき、
    前進報酬やトルクコストには一切手を加えない。学習用 env にのみ適用し、
    評価・再生では素の報酬を使う（真の性能で best_model を選ぶため）。
    """

    def __init__(self, env, fall_penalty: float = -40.0):
        super().__init__(env)
        self.fall_penalty = fall_penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward <= -100:  # 転倒（報酬がちょうど -100）のときだけ緩和
            reward = self.fall_penalty
        return obs, reward, terminated, truncated, info


def _latest_video(video_folder: str) -> Optional[str]:
    """video_folder 内で最後に更新された mp4 のパスを返す。無ければ None。

    gymnasium の RecordVideo は ``rl-video-episode-0.mp4`` のような名前で
    動画を保存するため、フォルダ内の mp4 を更新時刻でソートして最新を選ぶ。
    """
    if not os.path.isdir(video_folder):
        return None
    mp4_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    if not mp4_files:
        return None
    return max(mp4_files, key=os.path.getmtime)


def _in_notebook() -> bool:
    """Jupyter / IPython の対話カーネル上で実行されているかを判定する。"""
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        if shell is None:
            return False
        # ZMQInteractiveShell = Jupyter Notebook / Lab / Colab
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _open_with_os_player(video_path: str) -> None:
    """OS 既定のプレーヤで動画ファイルを開く（失敗しても致命的にしない）。"""
    try:
        if sys.platform == "darwin":          # macOS
            subprocess.run(["open", video_path], check=False)
        elif sys.platform.startswith("linux"):  # Linux（GUI 環境のみ有効）
            subprocess.run(["xdg-open", video_path], check=False)
        elif sys.platform.startswith("win"):   # Windows
            os.startfile(video_path)  # type: ignore[attr-defined]
    except Exception as exc:  # プレーヤ起動の失敗は無視（パスは別途表示済み）
        print(f"(動画の自動再生に失敗しました: {exc})")


def display_video(env, video_folder: str, open_player: bool = True) -> Optional[object]:
    """保存済みの動画を表示する。

    Args:
        env: 互換性のために受け取る環境/ラッパー（本実装では未使用）。
        video_folder: RecordVideo が動画を書き出したフォルダ。
        open_player: 通常実行時に OS 既定プレーヤで動画を開くか（既定 True）。
            学習中の進捗録画など、毎回プレーヤを開きたくない場合に False を渡す。

    Returns:
        Jupyter 上では IPython の表示オブジェクト（HTML）。それ以外は None。
    """
    video_path = _latest_video(video_folder)
    if video_path is None:
        print(
            f"[display_video] '{video_folder}' に mp4 が見つかりませんでした。\n"
            "  ・エピソードが途中で終了して録画されなかった可能性があります。\n"
            "  ・動画書き出しには ffmpeg と moviepy が必要です。"
        )
        return None

    if _in_notebook():
        # --- Jupyter / Colab: HTML5 <video> として埋め込み表示 ---
        from IPython.display import HTML, display

        with open(video_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        html = (
            '<video width="480" controls autoplay loop>'
            f'<source src="data:video/mp4;base64,{encoded}" type="video/mp4">'
            "お使いの環境では動画を再生できません。"
            "</video>"
        )
        return display(HTML(html))

    # --- 通常のスクリプト実行: パスを表示し OS 既定プレーヤで開く ---
    print(f"[display_video] 動画を保存しました: {os.path.abspath(video_path)}")
    # 環境変数 GYM_UTILS_NO_OPEN を設定すると自動再生を抑止できる
    # （ヘッドレス環境や、まとめて実行してプレーヤを開きたくない場合に便利）。
    # 呼び出し側が open_player=False を渡した場合も同様に抑止する。
    if open_player and not os.environ.get("GYM_UTILS_NO_OPEN"):
        _open_with_os_player(video_path)
    return None


def record_agent_video(
    agent,
    env_id: str,
    video_folder: str,
    deterministic: bool = True,
    name_prefix: str = "rl-video",
    open_player: bool = True,
    recurrent: bool = False,
    obs_transform: Optional[Callable] = None,
):
    """学習済みエージェントを 1 エピソード実行して動画に録画し、表示する。

    元ノートブックで各アルゴリズム（A2C / DDPG / TD3 / SAC）に共通だった
    「reset → predict → step →（done/truncated）→ close → display_video」
    という再生ループを 1 箇所に集約したもの。

    Args:
        agent: stable-baselines3 の学習済みモデル（predict を持つもの）。
        env_id: Gym 環境 ID（例: "BipedalWalker-v3"）。
        video_folder: 動画の保存先フォルダ。
        deterministic: True なら確定的な行動を選択（評価向き）。
        name_prefix: RecordVideo の出力ファイル名 prefix（既定 "rl-video"）。
            step 入りの prefix を渡すと同じフォルダでも上書きせず履歴を残せる。
        open_player: 通常実行時に OS 既定プレーヤで動画を開くか（既定 True）。
        recurrent: RecurrentPPO のような再帰方策（LSTM）かどうか（既定 False）。
            True のとき、predict に LSTM 隠れ状態(state)とエピソード開始フラグ
            (episode_start)を渡し、ステップ間で隠れ状態を引き継ぐ。これを渡さないと
            毎ステップ隠れ状態がゼロにリセットされ、再帰方策が正しく動かない。
            非再帰方策（A2C/PPO/TRPO/DDPG/TD3/SAC）では False のままでよい。
        obs_transform: 観測を predict に渡す前に変換する関数（既定 None＝変換しない）。
            VecNormalize で学習したモデル（例: ppo.py）の再生で、学習時の統計に合わせて
            観測を正規化するために使う（VecNormalize.normalize_obs を渡す）。env へ返す
            obs 自体は素のまま保持し、推論入力だけを変換する。
    """
    # rgb_array モードで環境を作り、RecordVideo でラップして録画する
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        disable_logger=True,
    )

    obs, info = env.reset()
    # 再帰方策用の LSTM 隠れ状態。最初のステップはエピソード開始として渡す。
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    while True:
        # 必要なら推論直前に観測を正規化する（VecNormalize 学習モデルの再生など）。
        pred_obs = obs_transform(obs) if obs_transform is not None else obs
        if recurrent:
            # LSTM 隠れ状態を引き継ぎながら推論する（RecurrentPPO 等）。
            action, lstm_states = agent.predict(
                pred_obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
        else:
            action, _states = agent.predict(pred_obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)
        episode_starts = np.array([done or truncated])
        if done or truncated:
            # エピソード終了でループを抜ける。
            # （元ノートブックはここで reset() を呼んでいたが、それは空の2本目の
            #  動画を生成してしまうため呼ばない。close() で1本目が書き出される。）
            break

    env.close()
    return display_video(env, video_folder, open_player=open_player)


class RecordBestVideoCallback(BaseCallback):
    """ベストモデル更新時に進捗動画を録画する SB3 コールバック。

    ``EvalCallback`` の ``callback_on_new_best`` として渡して使う。EvalCallback が
    新記録で best_model を保存した**直後**に呼ばれ、その時点の ``self.model``
    （＝今ベストになった現行モデル）から 1 エピソードを録画して、学習の進捗を
    履歴として残す。best_model.zip を再ロードしないので、保存との書き込み競合を
    気にする必要がない（SB3 2.4.0 で確認: EvalCallback._init_callback が
    ``callback_on_new_best.init_callback(self.model)`` を呼ぶため self.model 参照可）。

    学習の早い段階では best が頻繁に更新されるため、``min_interval_steps`` で
    録画の最低 step 間隔を設けて間引ける（``0`` なら best 更新ごとに毎回録画）。
    動画は ``best-step<総step数>-episode-0.mp4`` のように step 入りの名前で
    保存され、上書きされずに溜まっていく。録画時はプレーヤを開かない。

    Args:
        env_id: 録画に使う Gym 環境 ID。
        video_folder: 進捗動画の保存先フォルダ。
        min_interval_steps: 録画の最低 step 間隔（既定 0 = best 更新ごとに毎回）。
        deterministic: 録画時に確定的行動を取るか（評価向きは True）。
        verbose: 1 で録画ログを出力。
    """

    def __init__(
        self,
        env_id: str,
        video_folder: str,
        min_interval_steps: int = 0,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_id = env_id
        self.video_folder = video_folder
        self.min_interval_steps = min_interval_steps
        self.deterministic = deterministic
        self._last_record_step: Optional[int] = None

    def _on_step(self) -> bool:
        # callback_on_new_best として呼ばれる時点で best_model は保存済み、
        # self.model は今ベストになった現行モデル、self.num_timesteps は最新。
        if (
            self._last_record_step is not None
            and self.num_timesteps - self._last_record_step < self.min_interval_steps
        ):
            return True  # スロットル中（早期の頻繁な best 更新を間引く）

        try:
            os.makedirs(self.video_folder, exist_ok=True)
            record_agent_video(
                self.model,
                self.env_id,
                self.video_folder,
                deterministic=self.deterministic,
                name_prefix=f"best-step{self.num_timesteps:08d}",
                open_player=False,  # 学習中はプレーヤを開かない
            )
            self._last_record_step = self.num_timesteps
            if self.verbose:
                print(f"[progress-video] step={self.num_timesteps} の進捗動画を保存しました")
        except Exception as exc:  # 録画失敗で長時間学習を止めない
            print(f"[progress-video] 録画に失敗しました（学習は継続）: {exc}")
        return True


# =============================================================================
# run ディレクトリ管理（全アルゴリズム共通）
# =============================================================================
# 1 回の学習に関わる全成果物（logs / best_model / videos / final_model）を、
# 上書きを避けてタイムスタンプ別フォルダ runs/<algo>/<YYYYMMDD-HHMMSS-pid>/ に
# 自己完結させる。各スクリプトの train()/play() はここを呼んで run を作成・解決する。
RUNS_ROOT = "runs"


def new_run_dir(algo: str) -> str:
    """この学習専用の run ディレクトリ runs/<algo>/<run_id>/ を作成して返す。

    run_id は ``YYYYMMDD-HHMMSS-<pid>``。並列に複数 run を起動しても衝突しないよう
    PID を付ける（同じ秒でも PID で区別され、辞書順ソートの「最新」判定も保たれる）。
    monitor/checkpoint 用に ``logs/`` も同時に作成する。
    """
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    run_dir = os.path.join(RUNS_ROOT, algo, run_id)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir


def latest_run_dir(algo: str) -> Optional[str]:
    """runs/<algo>/ 内で辞書順（=時系列）最大の run ディレクトリを返す。無ければ None。

    run_id は ``YYYYMMDD-HHMMSS-...`` 形式で辞書順 = 時系列順になる。``_`` や ``.`` で
    始まるエントリ（runs/_stdout_logs や .DS_Store 等、run ではないもの）は除外する。
    """
    algo_root = os.path.join(RUNS_ROOT, algo)
    if not os.path.isdir(algo_root):
        return None
    runs = sorted(
        d
        for d in glob.glob(os.path.join(algo_root, "*"))
        if os.path.isdir(d) and not os.path.basename(d).startswith(("_", "."))
    )
    return runs[-1] if runs else None


def resolve_run_dir(algo: str, run: Optional[str] = None) -> str:
    """再生対象の run ディレクトリを解決する。

    run が id（runs/<algo>/ 配下のフォルダ名）またはパスならそれを、None なら
    最新 run を返す。指定 run が見つからない／run が1つも無い場合は FileNotFoundError。
    """
    if run is not None:
        run_dir = run if os.path.isdir(run) else os.path.join(RUNS_ROOT, algo, run)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"指定された run が見つかりません: {run}")
        return run_dir
    run_dir = latest_run_dir(algo)
    if run_dir is None:
        raise FileNotFoundError(
            f"runs/{algo}/ に run が見つかりません。"
            f"先に `python {algo}.py --mode train` を実行してください。"
        )
    return run_dir


def resolve_model_path(run_dir: str) -> str:
    """run 内のモデルパス（拡張子なし）を best_model → final_model の順に返す。

    どちらの .zip も無ければ FileNotFoundError。
    """
    best = os.path.join(run_dir, "best_model", "best_model")
    final = os.path.join(run_dir, "final_model")
    if os.path.exists(best + ".zip"):
        return best
    if os.path.exists(final + ".zip"):
        return final
    raise FileNotFoundError(f"run 内に学習済みモデルがありません: {run_dir}")
