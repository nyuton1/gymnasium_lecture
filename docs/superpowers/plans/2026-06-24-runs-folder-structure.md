# 成果物を runs/ に統一するフォルダ構造の整理 実装プラン

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 7アルゴリズムの学習成果物を、run 単位で自己完結する単一ルート `runs/<algo>/<timestamp>/` に統一して管理する。

**Architecture:** run のパス計算を `gym_utils.py` に共通ヘルパーとして集約し（`RecordBestVideoCallback` と同じ「共通部品」枠）、各スクリプトはモジュール定数を `ALGO` 1つに減らして `train()` が run ディレクトリ配下に書き出し・`play()` がそれを解決する。SAC 先行構造を全7本へ展開し、既存成果物を `runs/` へ移行する。

**Tech Stack:** Python 3 / Gymnasium 1.0.0 / stable-baselines3 2.4.0 / sb3-contrib 2.4.0。テストスイートは無く、検証は `py_compile` + 短い `--timesteps` のスモーク実行で行う。

## Global Constraints

- 検証は `.venv/bin/python` を直接呼ぶ（venv 有効化不要）。
- 構文チェック: `for f in sac a2c ddpg td3 trpo ppo recurrent_ppo gym_utils; do .venv/bin/python -m py_compile $f.py; done`
- アルゴのハイパーパラメータ・学習ロジックは一切変更しない（パス/run 周りのみ）。
- 進捗動画コールバック（`--progress-video-*` / `RecordBestVideoCallback`）は SAC 限定のまま。他6本に追加しない。
- macOS の multiprocessing は spawn のため `if __name__ == "__main__"` ガードを維持（既存）。
- `ENV_ID = "BipedalWalkerHardcore-v3"` は全7本で維持。env 名は run パスに含めない。
- run_id 形式は `YYYYMMDD-HHMMSS-<pid>`（SAC 既存規約を踏襲）。
- コミットメッセージは `<type>: <説明>` 形式（feat/refactor/docs 等）。属性表示なし。

---

### Task 1: gym_utils.py に run パス・ヘルパーを追加 + .gitignore を runs/ に集約

**Files:**
- Modify: `gym_utils.py`（import 追加 + 末尾近くにヘルパー追加）
- Modify: `.gitignore`

**Interfaces:**
- Produces:
  - `RUNS_ROOT: str = "runs"`
  - `new_run_dir(algo: str) -> str` — `runs/<algo>/<YYYYMMDD-HHMMSS-pid>/` を作成（`logs/` も作る）し run_dir を返す。
  - `latest_run_dir(algo: str) -> Optional[str]` — `runs/<algo>/` 内の辞書順最大 run（`_`/`.` 始まりは除外）。無ければ None。
  - `resolve_run_dir(algo: str, run: Optional[str] = None) -> str` — run 指定（id/パス）or 最新 run を返す。見つからなければ FileNotFoundError。
  - `resolve_model_path(run_dir: str) -> str` — run 内で best_model → final_model の順に存在するモデルパス（拡張子なし）。無ければ FileNotFoundError。

- [ ] **Step 1: import に datetime を追加**

`gym_utils.py` の import 群（`import sys` の次行、`from typing import Optional` の前）に追加する。
既存:

```python
import subprocess
import sys
from typing import Optional
```

変更後:

```python
import subprocess
import sys
from datetime import datetime
from typing import Optional
```

- [ ] **Step 2: ファイル末尾にヘルパーを追加**

`gym_utils.py` の末尾（`RecordBestVideoCallback` クラスの後）に以下をそのまま追記する。

```python


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
```

- [ ] **Step 3: .gitignore を runs/ に集約**

`.gitignore` の「学習・実行で生成される成果物」ブロックを置き換える。
既存:

```
# 学習・実行で生成される成果物
*_logs_bipedalwalker*/
*_runs_bipedalwalker*/
*_videos_practice/
tensorboard/
*.zip
*.log
```

変更後:

```
# 学習・実行で生成される成果物
runs/
tensorboard/
*.zip
*.log
```

- [ ] **Step 4: 構文チェックとヘルパーの機能スモーク**

Run:
```bash
.venv/bin/python -m py_compile gym_utils.py && \
.venv/bin/python -c '
import os, gym_utils as g
d = g.new_run_dir("_smoketest")
print("new_run_dir:", d, "logs exists:", os.path.isdir(os.path.join(d, "logs")))
print("latest_run_dir:", g.latest_run_dir("_smoketest"))
print("resolve_run_dir:", g.resolve_run_dir("_smoketest"))
try:
    g.resolve_model_path(d)
except FileNotFoundError as e:
    print("resolve_model_path raised as expected:", type(e).__name__)
'
rm -rf runs/_smoketest
```
Expected: `new_run_dir` が `runs/_smoketest/<ts-pid>` を表示、`logs exists: True`、`latest_run_dir`/`resolve_run_dir` が同じパス、`resolve_model_path raised as expected: FileNotFoundError`。最後に `runs/_smoketest` を削除。

- [ ] **Step 5: コミット**

```bash
git add gym_utils.py .gitignore
git commit -m "feat: run ディレクトリ管理ヘルパーを gym_utils に追加し .gitignore を runs/ に集約"
```

---

### Task 2: sac.py を共通ヘルパーへ載せ替え（レガシー・フォールバック削除）

**Files:**
- Modify: `sac.py`

**Interfaces:**
- Consumes: `new_run_dir`, `resolve_run_dir`, `resolve_model_path`（Task 1）。
- Produces: `train(...) -> str`（run_dir を返す。既存と同じ）。

- [ ] **Step 1: import を差し替え**

既存:

```python
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
```

変更後（`glob` と `datetime` を除去、ヘルパー import を追加）:

```python
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
```

- [ ] **Step 2: 定数ブロックを ALGO に置換**

既存:

```python
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
```

変更後:

```python
ENV_ID = "BipedalWalkerHardcore-v3"
# 1 回の学習に関わる全成果物を runs/<ALGO>/<run_id>/ に自己完結させる。
# run ディレクトリの作成・解決は gym_utils のヘルパーに集約。
ALGO = "sac"
```

- [ ] **Step 3: train() の run ディレクトリ生成を差し替え**

既存（関数先頭の run_id/run_dir 生成ブロック）:

```python
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
```

変更後:

```python
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
```

- [ ] **Step 4: tb_log_name を ALGO 由来に統一**

既存:

```python
        tb_log_name=f"SAC_{run_id}",
```

変更後:

```python
        tb_log_name=f"{ALGO.upper()}_{run_id}",
```

- [ ] **Step 5: `_latest_run_dir` 関数を削除**

`def _latest_run_dir() -> Optional[str]:` の関数定義全体（docstring 含む def 〜 `return runs[-1] if runs else None` まで）を削除する。解決は `resolve_run_dir` が担う。

- [ ] **Step 6: play() をヘルパーで書き換え（レガシー・フォールバック削除）**

既存の `play()` 関数全体（`def play(run: Optional[str] = None) -> None:` 〜 `record_agent_video(agent, ENV_ID, video_folder, deterministic=True)`）を、以下で置き換える。

```python
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
```

- [ ] **Step 7: 構文チェック + スモーク（学習→再生→最新run再生）**

Run:
```bash
.venv/bin/python -m py_compile sac.py && \
GYM_UTILS_NO_OPEN=1 .venv/bin/python sac.py --timesteps 2000 --n-envs 2 --no-progress-video && \
GYM_UTILS_NO_OPEN=1 .venv/bin/python sac.py --mode play && \
ls -R runs/sac | head -30
```
Expected: エラーなく完走。`runs/sac/<ts-pid>/` に `logs/` `best_model/best_model.zip` `final_model.zip` `videos/play/*.mp4` が生成され、`--mode play` が最新 run を解決して再生する。
（注: timesteps 2000 は learning_starts=3000 未満なので勾配は走らないが、run 構造・保存・再生の疎通確認には十分。EvalCallback は初回評価で best_model を保存する。）

- [ ] **Step 8: コミット**

```bash
git add sac.py
git commit -m "refactor: sac.py を共通 run ヘルパーへ載せ替えレガシー固定パスを削除"
```

---

### Task 3: ddpg.py と td3.py を runs/ 構造へ移行（オフポリシー）

**Files:**
- Modify: `ddpg.py`, `td3.py`

**Interfaces:**
- Consumes: `new_run_dir`, `resolve_run_dir`, `resolve_model_path`（Task 1）。
- Produces: 各 `train(...) -> str`、`play(run=None)`。

両ファイルに同一の変換を適用する。`<algo>` と `<MODEL>` は下表で置換（他は同一）。

| ファイル | `<algo>` | `<MODEL>` |
|---|---|---|
| `ddpg.py` | `ddpg` | `DDPG` |
| `td3.py` | `td3` | `TD3` |

- [ ] **Step 1: import に typing.Optional とヘルパーを追加**

各ファイルで、既存:

```python
import argparse
import os
import time
from datetime import timedelta
```

変更後（`Optional` を追加）:

```python
import argparse
import os
import time
from datetime import timedelta
from typing import Optional
```

そして既存:

```python
from gym_utils import FallPenaltyWrapper, record_agent_video
```

変更後:

```python
from gym_utils import (
    FallPenaltyWrapper,
    new_run_dir,
    record_agent_video,
    resolve_model_path,
    resolve_run_dir,
)
```

- [ ] **Step 2: 定数ブロックを ALGO に置換**

各ファイルで、既存（`<algo>` を表の値に読み替え）:

```python
ENV_ID = "BipedalWalkerHardcore-v3"
LOG_DIR = "./<algo>_logs_bipedalwalkerhardcore/"
VIDEO_FOLDER = "<algo>_bipedalwalkerhardcore_videos_practice"
FINAL_MODEL = "<algo>_bipedalwalkerhardcore"
```

変更後:

```python
ENV_ID = "BipedalWalkerHardcore-v3"
# 1 回の学習に関わる全成果物を runs/<ALGO>/<run_id>/ に自己完結させる。
ALGO = "<algo>"
```

- [ ] **Step 3: train() シグネチャと run ディレクトリ生成**

各ファイルで、既存:

```python
def train(timesteps: int, n_envs: int, fall_penalty: float) -> None:
    """<MODEL> モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。"""
    os.makedirs(LOG_DIR, exist_ok=True)
```

変更後:

```python
def train(timesteps: int, n_envs: int, fall_penalty: float) -> str:
    """<MODEL> モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。

    出力はタイムスタンプ別 run ディレクトリ runs/<ALGO>/<run_id>/ に隔離する。
    戻り値はこの run ディレクトリのパス（main() が play() に引き継ぐ）。
    """
    run_dir = new_run_dir(ALGO)
    run_id = os.path.basename(run_dir)
    logs_dir = os.path.join(run_dir, "logs")
    best_dir = os.path.join(run_dir, "best_model")
    final_model_path = os.path.join(run_dir, "final_model")
    print(f"[run] 出力先: {os.path.abspath(run_dir)}")
```

- [ ] **Step 4: train() 内のパス参照を run ディレクトリへ振り替え**

各ファイルで以下の4箇所を置換する。

`monitor_dir=LOG_DIR,` → `monitor_dir=logs_dir,`

`save_path=LOG_DIR,` → `save_path=logs_dir,`

既存:
```python
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "results"),
```
変更後:
```python
        best_model_save_path=best_dir,
        log_path=logs_dir,
```

- [ ] **Step 5: learn() に tb_log_name、保存先・戻り値を変更**

各ファイルで、既存:

```python
    model.learn(total_timesteps=timesteps, callback=callbacks)
    model.save(FINAL_MODEL)
    vec_env.close()
    eval_env.close()
```

変更後:

```python
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=f"{ALGO.upper()}_{run_id}",
    )
    model.save(final_model_path)
    vec_env.close()
    eval_env.close()
    return run_dir
```

- [ ] **Step 6: play() を書き換え**

各ファイルで `play()` 関数全体（`def play() -> None:` 〜 `record_agent_video(agent, ENV_ID, VIDEO_FOLDER, deterministic=True)`）を、以下で置換（`<MODEL>` は表の値）:

```python
def play(run: Optional[str] = None) -> None:
    """保存済みモデルをロードし、1エピソードを録画して再生する。

    run（id/パス）指定が無ければ runs/<ALGO>/ の最新 run を使い、
    run 内では best_model → final_model の順にフォールバックする。
    """
    run_dir = resolve_run_dir(ALGO, run)
    model_path = resolve_model_path(run_dir)
    video_folder = os.path.join(run_dir, "videos", "play")
    print(f"[play] モデル: {os.path.abspath(model_path)}.zip")
    agent = <MODEL>.load(model_path)
    record_agent_video(agent, ENV_ID, video_folder, deterministic=True)
```

- [ ] **Step 7: main() に --run を追加し play 呼び出しを変更**

各ファイルで、既存の `--mode` 引数定義の**前**に `--run` を追加する。既存:

```python
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
```

変更後:

```python
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
```

- [ ] **Step 8: 構文チェック + スモーク（オフポリシー代表 ddpg）**

Run:
```bash
.venv/bin/python -m py_compile ddpg.py td3.py && \
GYM_UTILS_NO_OPEN=1 .venv/bin/python ddpg.py --timesteps 2000 --n-envs 2 && \
ls -R runs/ddpg | head -20
```
Expected: エラーなく完走。`runs/ddpg/<ts-pid>/` に `logs/` `best_model/best_model.zip` `final_model.zip` `videos/play/*.mp4` が生成。

- [ ] **Step 9: コミット**

```bash
git add ddpg.py td3.py
git commit -m "refactor: ddpg/td3 を runs/<algo>/<ts> 構造へ移行"
```

---

### Task 4: a2c.py / trpo.py / ppo.py / recurrent_ppo.py を runs/ 構造へ移行（オンポリシー）

**Files:**
- Modify: `a2c.py`, `trpo.py`, `ppo.py`, `recurrent_ppo.py`

**Interfaces:**
- Consumes: `new_run_dir`, `resolve_run_dir`, `resolve_model_path`（Task 1）。
- Produces: 各 `train(...) -> str`、`play(run=None)`。

4ファイルに同一の変換を適用する。`<algo>` と `<MODEL>` は下表で置換。**`recurrent_ppo.py` のみ Step 6 の play() で `recurrent=True` を付ける**（後述）。

| ファイル | `<algo>` | `<MODEL>` | play の recurrent |
|---|---|---|---|
| `a2c.py` | `a2c` | `A2C` | なし |
| `trpo.py` | `trpo` | `TRPO` | なし |
| `ppo.py` | `ppo` | `PPO` | なし |
| `recurrent_ppo.py` | `recurrent_ppo` | `RecurrentPPO` | あり |

- [ ] **Step 1: import に typing.Optional とヘルパーを追加**

各ファイルで、既存:

```python
import argparse
import os
import time
from datetime import timedelta
```

変更後:

```python
import argparse
import os
import time
from datetime import timedelta
from typing import Optional
```

そして既存:

```python
from gym_utils import FallPenaltyWrapper, record_agent_video
```

変更後:

```python
from gym_utils import (
    FallPenaltyWrapper,
    new_run_dir,
    record_agent_video,
    resolve_model_path,
    resolve_run_dir,
)
```

- [ ] **Step 2: 定数ブロックを ALGO に置換**

各ファイルで、既存（行末コメントの有無は無視して4行まとめて置換。`<algo>` は表の値）:

```python
ENV_ID = "BipedalWalkerHardcore-v3"                       # 使用する Gym 環境
LOG_DIR = "./<algo>_logs_bipedalwalkerhardcore/"             # ログ・モデルの保存先
VIDEO_FOLDER = "<algo>...bipedalwalkerhardcore_videos_practice"  # 再生動画の保存先
FINAL_MODEL = "<algo>_bipedalwalkerhardcore"                 # 学習後の最終モデル保存名(.zip)
```

変更後:

```python
ENV_ID = "BipedalWalkerHardcore-v3"
# 1 回の学習に関わる全成果物を runs/<ALGO>/<run_id>/ に自己完結させる。
ALGO = "<algo>"
```

（注: `VIDEO_FOLDER` の元値はファイルにより `<algo>-...`（a2c/trpo/ppo/recurrent_ppo はハイフン）。定数ごと削除するので元の綴りは不問。`ENV_ID` 行のコメントも消えるが問題ない。）

- [ ] **Step 3: train() シグネチャと run ディレクトリ生成**

各ファイルで、既存:

```python
    # -------------------------------------------------------------------------
    # 1. ログディレクトリの準備
    # -------------------------------------------------------------------------
    os.makedirs(LOG_DIR, exist_ok=True)
```

変更後:

```python
    # -------------------------------------------------------------------------
    # 1. この学習専用の run ディレクトリ（タイムスタンプ）を作る
    # -------------------------------------------------------------------------
    run_dir = new_run_dir(ALGO)
    run_id = os.path.basename(run_dir)
    logs_dir = os.path.join(run_dir, "logs")
    best_dir = os.path.join(run_dir, "best_model")
    final_model_path = os.path.join(run_dir, "final_model")
    print(f"[run] 出力先: {os.path.abspath(run_dir)}")
```

そして関数シグネチャと docstring を更新する。既存:

```python
def train(timesteps: int, n_envs: int, fall_penalty: float) -> None:
    """<MODEL> モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。"""
```

変更後:

```python
def train(timesteps: int, n_envs: int, fall_penalty: float) -> str:
    """<MODEL> モデルを学習し、チェックポイント・ベストモデル・最終モデルを保存する。

    出力はタイムスタンプ別 run ディレクトリ runs/<ALGO>/<run_id>/ に隔離する。
    戻り値はこの run ディレクトリのパス（main() が play() に引き継ぐ）。
    """
```

- [ ] **Step 4: train() 内のパス参照を run ディレクトリへ振り替え**

各ファイルで以下を置換する。

`monitor_dir=LOG_DIR,` → `monitor_dir=logs_dir,`

`save_path=LOG_DIR,` → `save_path=logs_dir,`

既存:
```python
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "results"),
```
変更後:
```python
        best_model_save_path=best_dir,
        log_path=logs_dir,
```

- [ ] **Step 5: learn() に tb_log_name、保存先・戻り値を変更**

各ファイルで、既存:

```python
    model.learn(total_timesteps=timesteps, callback=callbacks)
    model.save(FINAL_MODEL)
    vec_env.close()
    eval_env.close()
```

変更後:

```python
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=f"{ALGO.upper()}_{run_id}",
    )
    model.save(final_model_path)
    vec_env.close()
    eval_env.close()
    return run_dir
```

- [ ] **Step 6: play() を書き換え**

**a2c / trpo / ppo**（recurrent なし）は `play()` 全体を以下で置換（`<MODEL>` は表の値）:

```python
def play(run: Optional[str] = None) -> None:
    """保存済みモデルをロードし、1エピソードを録画して再生する。

    run（id/パス）指定が無ければ runs/<ALGO>/ の最新 run を使い、
    run 内では best_model → final_model の順にフォールバックする。
    """
    run_dir = resolve_run_dir(ALGO, run)
    model_path = resolve_model_path(run_dir)
    video_folder = os.path.join(run_dir, "videos", "play")
    print(f"[play] モデル: {os.path.abspath(model_path)}.zip")
    agent = <MODEL>.load(model_path)
    record_agent_video(agent, ENV_ID, video_folder, deterministic=True)
```

**recurrent_ppo** は最終行の record_agent_video に `recurrent=True` を付けた版で置換:

```python
def play(run: Optional[str] = None) -> None:
    """保存済みモデルをロードし、1エピソードを録画して再生する。

    run（id/パス）指定が無ければ runs/<ALGO>/ の最新 run を使い、
    run 内では best_model → final_model の順にフォールバックする。
    """
    run_dir = resolve_run_dir(ALGO, run)
    model_path = resolve_model_path(run_dir)
    video_folder = os.path.join(run_dir, "videos", "play")
    print(f"[play] モデル: {os.path.abspath(model_path)}.zip")
    agent = RecurrentPPO.load(model_path)
    # RecurrentPPO は再生時に LSTM 隠れ状態を引き継ぐ必要があるため recurrent=True。
    record_agent_video(agent, ENV_ID, video_folder, deterministic=True, recurrent=True)
```

- [ ] **Step 7: main() に --run を追加し play 呼び出しを変更**

各ファイルで Task 3 Step 7 と同一の変換を適用する（`--mode` の前に `--run` を追加、`trained_run` を捕捉して `play(args.run if args.run is not None else trained_run)` に変更）。

- [ ] **Step 8: 構文チェック + スモーク（オンポリシー代表 ppo + recurrent_ppo の再帰再生）**

Run:
```bash
.venv/bin/python -m py_compile a2c.py trpo.py ppo.py recurrent_ppo.py && \
GYM_UTILS_NO_OPEN=1 .venv/bin/python ppo.py --timesteps 2000 --n-envs 2 && \
GYM_UTILS_NO_OPEN=1 .venv/bin/python recurrent_ppo.py --timesteps 256 --n-envs 2 && \
ls -R runs/ppo runs/recurrent_ppo | head -30
```
Expected: 両方エラーなく完走。`runs/ppo/<ts-pid>/` と `runs/recurrent_ppo/<ts-pid>/` に run 一式が生成。recurrent_ppo の play が `recurrent=True` で LSTM 隠れ状態を引き継いで録画する。

- [ ] **Step 9: コミット**

```bash
git add a2c.py trpo.py ppo.py recurrent_ppo.py
git commit -m "refactor: a2c/trpo/ppo/recurrent_ppo を runs/<algo>/<ts> 構造へ移行"
```

---

### Task 5: 既存成果物を runs/ へ移行（ワンショット）

**Files:**
- 移動のみ（コミット対象外＝すべて gitignore 済み）。

7アルゴの旧構造成果物・SAC 既存 run・散らかりログを `runs/` 配下へ移す。`<ts>` は移行元 zip の mtime から生成し `-legacy` を付ける。

- [ ] **Step 1: 移行前の状態を記録**

Run:
```bash
ls -d *_logs_bipedalwalkerhardcore *_bipedalwalkerhardcore.zip *videos_practice sac_runs_bipedalwalkerhardcore sac_train_*.log 2>/dev/null
```
Expected: 旧構造のフォルダ・zip・SAC run ルート・stdout ログが列挙される（移行対象の確認）。

- [ ] **Step 2: 旧固定パス6アルゴを runs/<algo>/<ts>-legacy/ へ移行**

Run:
```bash
set -e
for algo in a2c ddpg td3 trpo ppo recurrent_ppo; do
  zip="${algo}_bipedalwalkerhardcore.zip"
  logs="${algo}_logs_bipedalwalkerhardcore"
  [ -e "$zip" ] || { echo "skip $algo (no final zip)"; continue; }
  ts=$(date -r "$zip" +%Y%m%d-%H%M%S)
  dst="runs/${algo}/${ts}-legacy"
  mkdir -p "$dst/logs" "$dst/videos/play"
  # best_model を先に退避してから残りの logs 内容を移す
  [ -d "$logs/best_model" ] && mv "$logs/best_model" "$dst/best_model"
  if [ -d "$logs" ]; then mv "$logs"/* "$dst/logs/" 2>/dev/null || true; rmdir "$logs" 2>/dev/null || true; fi
  mv "$zip" "$dst/final_model.zip"
  # 動画フォルダ名は - / _ 混在。glob で拾って中身を play/ へ。
  for v in ${algo}*videos_practice; do
    [ -d "$v" ] || continue
    mv "$v"/* "$dst/videos/play/" 2>/dev/null || true
    rmdir "$v" 2>/dev/null || true
  done
  echo "migrated $algo -> $dst"
done
```
Expected: 各アルゴについて `migrated <algo> -> runs/<algo>/<ts>-legacy` が表示され、旧 `<algo>_logs_...`・`<algo>_....zip`・`<algo>...videos_practice` がルートから消える。
（注: `recurrent_ppo*videos_practice` は `recurrent_ppo-...` を拾う。`ppo*videos_practice` は `ppo-...` を拾い recurrent_ppo は別 prefix なので二重移動は起きない。`recurrent_ppo` ループ時に対象 zip/logs が既に消えていれば skip される。)

- [ ] **Step 3: SAC を runs/sac/ へ移行**

Run:
```bash
set -e
mkdir -p runs/sac
# 3a. 既存 run（構造一致）をそのまま移動
if [ -d sac_runs_bipedalwalkerhardcore ]; then
  find sac_runs_bipedalwalkerhardcore -maxdepth 1 -mindepth 1 -type d -exec mv {} runs/sac/ \;
  rm -f sac_runs_bipedalwalkerhardcore/.DS_Store
  rmdir sac_runs_bipedalwalkerhardcore 2>/dev/null || true
fi
# 3b. SAC レガシー固定パス3点を 1 つの -legacy run にまとめる
if [ -e sac_bipedalwalkerhardcore.zip ]; then
  ts=$(date -r sac_bipedalwalkerhardcore.zip +%Y%m%d-%H%M%S)
  dst="runs/sac/${ts}-legacy"
  mkdir -p "$dst/logs" "$dst/videos/play"
  [ -d sac_logs_bipedalwalkerhardcore/best_model ] && mv sac_logs_bipedalwalkerhardcore/best_model "$dst/best_model"
  if [ -d sac_logs_bipedalwalkerhardcore ]; then mv sac_logs_bipedalwalkerhardcore/* "$dst/logs/" 2>/dev/null || true; rmdir sac_logs_bipedalwalkerhardcore 2>/dev/null || true; fi
  mv sac_bipedalwalkerhardcore.zip "$dst/final_model.zip"
  for v in sac_bipedalwalkerhardcore_videos_practice; do
    [ -d "$v" ] || continue
    mv "$v"/* "$dst/videos/play/" 2>/dev/null || true
    rmdir "$v" 2>/dev/null || true
  done
  echo "migrated sac legacy -> $dst"
fi
# 3c. stray stdout ログを保全
if ls sac_train_*.log >/dev/null 2>&1; then
  mkdir -p runs/_stdout_logs
  mv sac_train_*.log runs/_stdout_logs/
fi
```
Expected: `sac_runs_bipedalwalkerhardcore/` 配下の run が `runs/sac/` 直下へ、SAC レガシー3点が `runs/sac/<ts>-legacy/` へ、`sac_train_*.log` が `runs/_stdout_logs/` へ移動。

- [ ] **Step 4: .DS_Store 掃除と検証**

Run:
```bash
find . -name .DS_Store -not -path './.venv/*' -not -path './.git/*' -delete
echo "--- runs/ ツリー ---"; find runs -maxdepth 2 | sort
echo "--- 旧構造の残骸（空であるべき）---"; ls -d *_logs_bipedalwalkerhardcore *_bipedalwalkerhardcore.zip *videos_practice sac_runs_bipedalwalkerhardcore 2>/dev/null || echo "残骸なし"
echo "--- git status（runs/ は無視され clean であるべき）---"; git status --short
```
Expected: `runs/` に `sac/ a2c/ ddpg/ td3/ trpo/ ppo/ recurrent_ppo/ _stdout_logs/` が並ぶ。「残骸なし」。`git status --short` に `runs/` 由来の差分が出ない（.gitignore で無視）。

- [ ] **Step 5: 移行物のロード確認**

Run:
```bash
GYM_UTILS_NO_OPEN=1 .venv/bin/python a2c.py --mode play
```
Expected: `runs/a2c/<ts>-legacy/` を最新 run として解決し、移行した best_model または final_model をロードして1エピソード録画・再生できる。

（このタスクは移動のみ。コミット対象は無い。）

---

### Task 6: ドキュメント更新（CLAUDE.md / README.md / AGENTS.md）

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `AGENTS.md`

**Interfaces:** なし（ドキュメントのみ）。

- [ ] **Step 1: 現行ドキュメントの該当箇所を確認**

Run:
```bash
grep -n "runs_bipedalwalker\|_logs_bipedalwalker\|_videos_practice\|RUN_ROOT\|sac.py だけ先行\|成果物とログ\|LOG_DIR" CLAUDE.md README.md AGENTS.md
```
Expected: パス・構造に言及する箇所が列挙される（更新対象の特定）。

- [ ] **Step 2: CLAUDE.md を更新**

`CLAUDE.md` の以下を新構造に合わせて書き換える（実ファイルの現行記述を読み、該当段落を置換する）:
- 「成果物とログ」節: SAC（新構造）と旧6アルゴ（旧構造）の分裂記述を、**全7アルゴ共通**の
  `runs/<algo>/<YYYYMMDD-HHMMSS-pid>/`（配下 `logs/` `best_model/best_model.zip`
  `videos/{progress,play}/` `final_model.zip`）に統一した記述へ。`runs/` は gitignore、
  `tensorboard/` は全アルゴ共有のままと明記。
- 「sac.py だけ先行している機能」: run 隔離は**全7共通になった**旨へ更新。SAC 限定で残るのは
  進捗動画コールバック（`--progress-video-*` / `--run` は全7共通）であることを明記。
- コマンド節: 出力先の説明や `--run` の記述を runs/ ベースへ更新（`--run` は全7で利用可に）。
- 「録画・再生は gym_utils に集約」付近に、run パス管理ヘルパー（`new_run_dir` /
  `latest_run_dir` / `resolve_run_dir` / `resolve_model_path`）も gym_utils の共通部品である旨を追記。

- [ ] **Step 3: README.md を更新**

`README.md` のコマンド例・ディレクトリ構造の記述を、`runs/<algo>/<ts>/` レイアウトと
`--run` オプション（全7共通）に合わせて更新する。旧 `<algo>_logs_...` / `*_videos_practice` /
`sac_runs_...` への言及を runs/ ベースへ置換する。

- [ ] **Step 4: AGENTS.md を更新**

`AGENTS.md` にパス・構造への言及があれば CLAUDE.md と同じ方針で更新する（無ければ変更不要）。

- [ ] **Step 5: 整合性チェック**

Run:
```bash
grep -rn "runs_bipedalwalker\|_logs_bipedalwalker\|_videos_practice\|RUN_ROOT" CLAUDE.md README.md AGENTS.md || echo "旧構造への言及は残っていない"
```
Expected: `旧構造への言及は残っていない`（or 残った箇所を Step 2-4 で修正）。

- [ ] **Step 6: コミット**

```bash
git add CLAUDE.md README.md AGENTS.md
git commit -m "docs: フォルダ構造を runs/<algo>/<ts> 統一に合わせて更新"
```

---

### Task 7: 全体最終検証

**Files:** なし（検証のみ）。

- [ ] **Step 1: 全スクリプト構文チェック**

Run:
```bash
for f in sac a2c ddpg td3 trpo ppo recurrent_ppo gym_utils; do .venv/bin/python -m py_compile $f.py && echo "OK $f"; done
```
Expected: 8ファイルすべて `OK`。

- [ ] **Step 2: 学習スモークの網羅（オフ/オン両系統 + 再帰）**

Run:
```bash
GYM_UTILS_NO_OPEN=1 .venv/bin/python td3.py --timesteps 2000 --n-envs 2 && \
GYM_UTILS_NO_OPEN=1 .venv/bin/python a2c.py --timesteps 2000 --n-envs 2 && \
GYM_UTILS_NO_OPEN=1 .venv/bin/python trpo.py --timesteps 1000 --n-envs 2
```
Expected: 3本ともエラーなく完走し、それぞれ `runs/<algo>/<ts-pid>/` が新規生成される。

- [ ] **Step 3: --run による特定 run 再生**

Run:
```bash
RUN_ID=$(ls -1 runs/ppo | grep -v '^_' | sort | tail -1)
echo "再生する run: $RUN_ID"
GYM_UTILS_NO_OPEN=1 .venv/bin/python ppo.py --mode play --run "$RUN_ID"
```
Expected: 指定した `runs/ppo/$RUN_ID/` を解決して再生できる。

- [ ] **Step 4: SAC のフル学習スモーク（勾配が走ることの確認・任意）**

Run:
```bash
GYM_UTILS_NO_OPEN=1 .venv/bin/python sac.py --timesteps 8000 --n-envs 2 --no-progress-video
```
Expected: `learning_starts=3000` を超えて勾配更新が走り、`runs/sac/<ts-pid>/` に run 一式が生成（教材の実学習パスの最終確認。時間がかかるため任意）。

- [ ] **Step 5: 作業ツリーの最終確認**

Run:
```bash
git status --short && echo "--- runs/ 直下 ---" && ls runs/
```
Expected: トラッキング対象に `runs/` 由来の差分が無く（gitignore 済み）、`runs/` 直下に7アルゴ + `_stdout_logs/` が揃っている。

---

## Self-Review

**1. Spec coverage（スペック §ごと）:**
- 目標レイアウト `runs/<algo>/<ts>/` → Task 2–5。✓
- run 単位の整理基準 → 全 train/play がヘルパー経由（Task 1–4）。✓
- 既存成果物の移行・保全 → Task 5（6アルゴ + SAC既存run + SACレガシー + stdoutログ + .DS_Store）。✓
- `_stdout_logs` を `runs/` 直下（run 解決対象外）に置く → Task 5 Step 3c + Task 1 の `latest_run_dir` 除外フィルタ。✓
- `<ts>` は最終モデル zip の mtime + `-legacy` → Task 5 Step 2/3。✓
- gym_utils ヘルパー4種 + RUNS_ROOT → Task 1。✓
- 各スクリプト: 定数→ALGO / train が run_dir 返す / play が resolve / `--run` 追加 / SAC レガシー削除 → Task 2–4。✓
- 進捗動画は SAC 限定のまま → 他6本に追加しない（Task 3/4 はコールバック非追加）。✓
- recurrent_ppo の play は `recurrent=True` 維持 → Task 4 Step 6。✓
- `.gitignore` を runs/ に集約 → Task 1 Step 3。✓
- docs（CLAUDE/README/AGENTS）更新 → Task 6。✓
- 検証（py_compile + スモーク + 移行物ロード + --run）→ 各タスク末 + Task 7。✓

**2. Placeholder scan:** コードステップは実コードを記載。docs（Task 6）はプローズ更新のため対象文書の現行を読んで該当段落を置換する指示で、変更内容（伝えるべき事実）を列挙済み。プレースホルダ無し。

**3. Type consistency:** ヘルパー名は全タスクで一致（`new_run_dir` / `latest_run_dir` / `resolve_run_dir` / `resolve_model_path`）。`train(...) -> str`（run_dir 返却）と `play(run: Optional[str] = None)`、`tb_log_name=f"{ALGO.upper()}_{run_id}"`、`run_id = os.path.basename(run_dir)` を全スクリプトで統一。✓
