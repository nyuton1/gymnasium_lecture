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
from typing import Optional

import gymnasium as gym


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


def display_video(env, video_folder: str) -> Optional[object]:
    """保存済みの動画を表示する。

    Args:
        env: 互換性のために受け取る環境/ラッパー（本実装では未使用）。
        video_folder: RecordVideo が動画を書き出したフォルダ。

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
    if not os.environ.get("GYM_UTILS_NO_OPEN"):
        _open_with_os_player(video_path)
    return None


def record_agent_video(
    agent,
    env_id: str,
    video_folder: str,
    deterministic: bool = True,
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
    """
    # rgb_array モードで環境を作り、RecordVideo でラップして録画する
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        disable_logger=True,
    )

    obs, info = env.reset()
    while True:
        action, _states = agent.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            # エピソード終了でループを抜ける。
            # （元ノートブックはここで reset() を呼んでいたが、それは空の2本目の
            #  動画を生成してしまうため呼ばない。close() で1本目が書き出される。）
            break

    env.close()
    return display_video(env, video_folder)
