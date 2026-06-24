# 設計: 成果物を `runs/` に統一するフォルダ構造の整理

- 日付: 2026-06-24
- 対象: `a2c.py` / `ddpg.py` / `td3.py` / `sac.py` / `trpo.py` / `ppo.py` / `recurrent_ppo.py` / `gym_utils.py` / 各ドキュメント / `.gitignore`
- 目的: ルート直下に約20個散乱している学習成果物を、run 単位で自己完結する単一ルート `runs/` に統一して管理する。

## 背景と問題

現状、成果物の管理方式が SAC と他6アルゴで分裂している。

- **SAC（新構造・先行）**: `sac_runs_bipedalwalkerhardcore/<YYYYMMDD-HHMMSS-pid>/` に
  `logs/` `best_model/` `videos/{progress,play}/` `final_model.zip` を run 単位で自己完結。
- **他6アルゴ（旧構造）**: ルート直下に3点が散乱。
  - `<algo>_logs_bipedalwalkerhardcore/`（monitor csv・checkpoint zip・`best_model/`・`results/`）
  - `<algo>_bipedalwalkerhardcore.zip`（最終モデル）
  - `<algo>...videos_practice/`（再生動画。フォルダ名は `-` と `_` が混在し不統一）
- 加えてルートに散らかり物: `sac_train_*.log`（1.3MB×4）、`.DS_Store`、SAC のレガシー固定パス3点、共有 `tensorboard/`。

結果、ルート直下に成果物フォルダ・ファイルが約20個ぶら下がり、どの成果がどの学習に属するか追いにくい。

## 目標レイアウト（最終形）

```
gymnasium_lecture/
├── *.py / gym_utils.py          # コード（場所は不変）
├── runs/                        # 全成果物の単一ルート（.gitignore 済み）
│   ├── sac/<YYYYMMDD-HHMMSS-pid>/
│   │   ├── logs/                # checkpoint zip / monitor csv / eval ログ
│   │   ├── best_model/best_model.zip
│   │   ├── videos/
│   │   │   ├── progress/        # 学習途中の進捗動画（SAC のみ生成）
│   │   │   └── play/            # 再生動画
│   │   └── final_model.zip
│   ├── a2c/<ts>/   ddpg/   td3/   trpo/   ppo/   recurrent_ppo/   # 全7アルゴ同形
│   └── _stdout_logs/            # 移行した sac_train_*.log の保全先（run ではない）
├── tensorboard/                 # 全アルゴ共有（従来通り・変更なし）
└── docs / notebooks / *.html    # 教材ソース（現状維持）
```

7スクリプト全部が同一の run 構造を共有する。非 SAC には `videos/progress/` が無いだけで、他は同形。

## 設計判断（確定事項）

1. **整理の基準 = run 単位**。1回の学習に関わる全成果物を1つのタイムスタンプ・フォルダに自己完結させる。
   run 丸ごとコピー・削除・比較が容易で、各 run が再現単位として独立する。
2. **既存成果物 = `runs/` へ移行して保全**（データロスなし）。gitignore 済みだが、SAC の 764000 step 等
   実体ある学習結果やデモ用モデルを失わないため。
3. **トップ階層名 = `runs/`**。env 名（`bipedalwalkerhardcore`）は path から省く（1環境リポジトリのため）。
4. **`tensorboard/` は共有のまま**（変更しない）。
5. **進捗動画コールバックは SAC 限定のまま据え置く**（フォルダ構造の整理とは別機能。YAGNI）。
6. **アルゴのハイパーパラメータ・学習ロジックは一切変更しない**。

## 移行（ワンショット・gitignore 済み成果物のため非コミット）

レビュー可能な `mv` で以下を実施する。`<ts>` は移行元の最終モデル `<algo>_bipedalwalkerhardcore.zip`
の更新時刻（mtime。無ければ `best_model/best_model.zip` の mtime）から `YYYYMMDD-HHMMSS` を生成し、
移行物とわかるよう接尾辞 `-legacy` を付ける。各旧アルゴは1つの `-legacy` run にまとまる。

| 移行元（旧構造） | 移行先 |
|---|---|
| `<algo>_logs_.../`（`best_model/` を除く全内容: monitor csv・checkpoint・`results/`） | `runs/<algo>/<ts>-legacy/logs/` |
| `<algo>_logs_.../best_model/` | `runs/<algo>/<ts>-legacy/best_model/` |
| `<algo>_bipedalwalkerhardcore.zip` | `runs/<algo>/<ts>-legacy/final_model.zip` |
| `<algo>...videos_practice/`（中身） | `runs/<algo>/<ts>-legacy/videos/play/` |
| `sac_runs_bipedalwalkerhardcore/<ts>/`（既存 run） | `runs/sac/<ts>/`（構造一致のためそのまま移動） |
| SAC レガシー固定パス3点（`sac_logs_.../` `sac_bipedalwalkerhardcore.zip` `sac_..._videos_practice/`） | `runs/sac/<ts>-legacy/`（上記同型で1 run にまとめる） |
| `sac_train_*.log`（4本） | `runs/_stdout_logs/`（保全。`runs/sac/` 配下に置かない＝run 解決の対象外にするため） |
| `.DS_Store`（各所） | 削除 |

対象アルゴ: `a2c` `ddpg` `td3` `trpo` `ppo` `recurrent_ppo`（旧固定パス）＋ `sac`（既存 run はそのまま、レガシー固定パスは `-legacy` run へ）。

移行後、旧構造のフォルダ・zip はルートから消える。

## コード変更

プロジェクト方針（共通部品は `gym_utils.py`、スクリプト骨格は意図的に重複）を踏襲する。
run のパス計算は7本で完全に同一なので、`RecordBestVideoCallback` と同じ「共通部品」として `gym_utils.py` に集約する。

### `gym_utils.py` に追加するヘルパー

```python
RUNS_ROOT = "runs"

def new_run_dir(algo: str) -> str:
    """runs/<algo>/<YYYYMMDD-HHMMSS-pid>/ を作成し、その run ディレクトリを返す。
    並列 run 衝突回避のため pid を付ける（SAC 既存実装と同じ run_id 規約）。
    logs/ も同時に作る。"""

def latest_run_dir(algo: str) -> Optional[str]:
    """runs/<algo>/ 内で辞書順（=時系列）最大の run を返す。無ければ None。
    `_` や `.` で始まるエントリ（_stdout_logs / .DS_Store 等）は run 候補から除外する。"""

def resolve_run_dir(algo: str, run: Optional[str]) -> Optional[str]:
    """run が id/パスならそれを、None なら latest_run_dir(algo) を返す。
    指定 run が見つからなければ FileNotFoundError。"""

def resolve_model_path(run_dir: str) -> str:
    """run 内で best_model → final_model の順に存在するモデルパス（拡張子なし）を返す。
    どちらも無ければ FileNotFoundError。"""
```

`run_id` は `os.path.basename(run_dir)` で取得し、`tb_log_name=f"{ALGO.upper()}_{run_id}"` に使う。

### 各スクリプト（7本を同形に編集）

- モジュール定数 `LOG_DIR` / `VIDEO_FOLDER` / `FINAL_MODEL` の3つ → `ALGO = "<algo>"` 1つに置換
  （`ENV_ID` は維持）。
- `train(...)`:
  - 先頭で `run_dir = new_run_dir(ALGO)` を作り、`logs_dir = run_dir/logs`、
    `best_dir = run_dir/best_model`、`final = run_dir/final_model` を導出。
  - `make_vec_env(monitor_dir=logs_dir, ...)`、`CheckpointCallback(save_path=logs_dir)`、
    `EvalCallback(best_model_save_path=best_dir, log_path=logs_dir)`（SAC の参照実装に合わせ
    eval ログは `logs/` 直下。旧 `results/` サブフォルダは廃止）。
  - `model.save(final)`、末尾で `return run_dir`。
- `play(run=None)`:
  - `run_dir = resolve_run_dir(ALGO, run)`、`model_path = resolve_model_path(run_dir)`、
    `video_folder = run_dir/videos/play` に録画。
  - SAC のレガシー固定パス・フォールバック分岐は**削除**（移行で全成果物が `runs/` 配下に入り不要）。
  - `recurrent_ppo.py` のみ `record_agent_video(..., recurrent=True)` 維持。
- `main()`: 全7本に `--run`（id/パス。既定=最新 run）を追加。`both` のときは学習した run を、
  `play` 単独のときは `--run`（無ければ最新）を再生。
- SAC は既存の `--progress-video-every` / `--no-progress-video` / `RecordBestVideoCallback` 連携を維持。
  これらは SAC 限定のまま（他6本には追加しない）。

## ドキュメント・無視設定の更新

- `.gitignore`: 旧パターン（`*_logs_bipedalwalker*/` `*_runs_bipedalwalker*/` `*_videos_practice/`）を
  `runs/` に集約。`tensorboard/` `*.zip` `*.log` `.venv/` `__pycache__/` `*.pyc` `.DS_Store` は維持。
- `CLAUDE.md`: 「成果物とログ」節を全面更新（SAC/旧構造の分裂記述を `runs/<algo>/<ts>/` 統一へ書き換え）。
  併せて「sac.py だけ先行している機能」のうち run 隔離は全7共通になった旨を反映（進捗動画は SAC 限定で残す）。
- `README.md`: コマンド例・ディレクトリ構造の記述を新レイアウトへ更新。
- `AGENTS.md`: パス記述があれば同様に更新。

## スコープ境界（やらないこと）

- 進捗動画コールバックの他アルゴへの展開（SAC 限定のまま）。
- アルゴのハイパーパラメータ・学習ロジックの変更。
- `tensorboard/` の移設。
- env 名を path に含めること（1環境リポジトリのため省く）。
- 教材ソース（notebooks / `.py` 移植元 / `*.html` スライド）の移動。

## 検証

1. 全スクリプト構文確認:
   `for f in sac a2c ddpg td3 trpo ppo recurrent_ppo gym_utils; do .venv/bin/python -m py_compile $f.py; done`
2. スモーク（パイプライン疎通と run 生成・解決の確認）:
   - オフポリシー1本: `.venv/bin/python sac.py --timesteps 8000 --mode both`
     （`learning_starts=3000` 超で勾配が走る。`runs/sac/<ts>/` 生成と play 解決を確認）
   - オンポリシー1本: `.venv/bin/python ppo.py --timesteps 2000 --mode both`
     （`runs/ppo/<ts>/` 生成を確認）
3. 移行物のロード確認: `.venv/bin/python a2c.py --mode play`
   （移行した `runs/a2c/<ts>-legacy/` を最新 run として解決し再生できること）。
4. `--run` 指定: 任意アルゴで `--mode play --run <id>` が特定 run を再生できること。
