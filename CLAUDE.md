# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## このリポジトリについて

Stable-Baselines3 + Gymnasium で 7 つの深層強化学習アルゴリズム
（A2C / DDPG / TD3 / SAC / TRPO / PPO / RecurrentPPO）を
`BipedalWalkerHardcore-v3` で「学習 → ベストモデル保存 → 録画 → 再生」する教材用サンプル集。
元は Colab ノートブック 2 本（`10_..._ipynb` = A2C/DDPG/TD3/SAC、`11_..._ipynb` = TRPO/PPO/RecurrentPPO。
それぞれ同名 `.py` も）で、そこから Colab 固有処理を除いてアルゴリズムごとのローカル実行スクリプトに
分割したもの。ノートブックと `.py`（移植元）はリポジトリに残してあるが、現役のコードは
7 スクリプト + `gym_utils.py`。`11_` は素の `BipedalWalker-v3` を使っていたが、既存スクリプトに
合わせて `BipedalWalkerHardcore-v3` に統一した。TRPO / RecurrentPPO は `sb3_contrib` を要する。

## コマンド

```bash
# 初回セットアップ（venv 作成 + swig/依存導入 + 環境生成確認）。2 回目以降は不要。
bash setup.sh

# 実行（venv 有効化は不要。直接 .venv の python を呼ぶのが確実）
.venv/bin/python sac.py                          # 学習(既定2000step) → 録画 → 再生
.venv/bin/python sac.py --timesteps 50000        # しっかり学習
.venv/bin/python sac.py --mode train             # 学習のみ（録画・再生なし）
.venv/bin/python sac.py --mode play              # 保存済みモデルを再生のみ
.venv/bin/python sac.py --n-envs 8               # 並列環境数を変更（既定 4）
.venv/bin/python sac.py --fall-penalty -40       # 転倒ペナルティ(-100)を緩和（既定 -40 / -100 で無効=元の挙動）
# 以下は sac.py 限定のオプション（他3スクリプトには未実装）
.venv/bin/python sac.py --progress-video-every 20000  # best更新時の進捗動画を最低2万step間隔で録画（既定 20000 / 0 で毎回）
.venv/bin/python sac.py --no-progress-video      # 学習中の進捗動画録画を無効化
.venv/bin/python sac.py --mode play --run 20260617-164530  # 特定 run を再生（既定は最新 run）

# 構文チェック（変更後に必ず走らせる軽量な検証。gym_utils も含める）
for f in sac a2c ddpg td3 trpo ppo recurrent_ppo gym_utils; do .venv/bin/python -m py_compile $f.py; done

# 学習過程の可視化（全アルゴリズム共通の tensorboard/ に出力）
.venv/bin/python -m tensorboard.main --logdir tensorboard/
```

このリポジトリにテストスイートは無い。検証は「`py_compile` で構文確認」＋「短い `--timesteps` で
スモーク実行（パイプラインが最後まで通るか）」で行う。SAC は `learning_starts=3000` のため、
学習が実際に走るか確かめるには `--timesteps` を 3000 超（例 8000）にする必要がある。

## アーキテクチャ

### 7 スクリプトは同一スケルトンの「並行コピー」

`a2c.py` / `ddpg.py` / `td3.py` / `sac.py` / `trpo.py` / `ppo.py` / `recurrent_ppo.py` は同じ構造を共有する:

- モジュール冒頭の定数 `ENV_ID` / `LOG_DIR` / `VIDEO_FOLDER` / `FINAL_MODEL`
- `train(timesteps, n_envs, fall_penalty)` — 環境構築 → コールバック → モデル構築 → `learn` → 保存
- `play()` — `best_model` があればそれを、無ければ `FINAL_MODEL` をロードして 1 エピソード録画
- `main()` — argparse（`--timesteps` / `--n-envs` / `--fall-penalty` / `--mode`）→ 経過時間を測って出力

**重要**: 共通の挙動を変えるときは（sac.py を除く）スクリプトすべてを同じように編集する必要がある（共通基底クラスは無い、意図的な重複）。`ddpg.py` / `td3.py` は同型、`a2c.py` / `trpo.py` / `ppo.py` / `recurrent_ppo.py` はオンポリシー版で下記の通り異なる。`recurrent_ppo.py` だけ `play()` が `record_agent_video(..., recurrent=True)` を呼び、再生時に LSTM 隠れ状態を引き継ぐ（他は省略=非再帰）。

**例外（sac.py だけ先行している機能）**: `sac.py` は下記2点で他3スクリプトと異なる。共通化された
部品（`gym_utils.RecordBestVideoCallback` 等）は用意済みなので、必要なら同型で他へ展開できる。
- 出力をタイムスタンプ別 run フォルダに隔離（`train()` が `RUN_ROOT/<ts>/...` を返す、`play(run=None)` は最新 run を解決）。
- 学習中の進捗動画コールバック（`--progress-video-every` / `--no-progress-video`、`--run`）。

### オンポリシー（A2C/TRPO/PPO/RecurrentPPO）とオフポリシー（DDPG/TD3/SAC）の差

スクリプト横断の変更で最も間違えやすい軸。両者で適用すべきパラメータが違う:

- **オフポリシー（ddpg/td3/sac）**: リプレイバッファを持つ。`buffer_size`（現在 1M）、
  `learning_starts`、`train_freq`、`gradient_steps=-1`、（td3 のみ `policy_delay`）が効く。
  `gradient_steps=-1` は「収集ステップ数(=n_envs)ぶん勾配更新」で並列時も更新比 1:1 を維持する。
  DDPG/TD3 は探索用に `NormalActionNoise` を使う。
- **オンポリシー（a2c/trpo/ppo/recurrent_ppo）**: リプレイバッファが無い。`buffer_size` /
  `gradient_steps` / `learning_starts` は存在しない。実効バッチ = `n_steps` × `n_envs` で制御する
  （a2c=5 / trpo=500 / ppo=2048 / recurrent_ppo=128）。ppo/recurrent_ppo は `batch_size` /
  `n_epochs` / `clip_range`、trpo は `gae_lambda` を持つ。**これらに buffer_size 等を足さないこと。**
  TRPO / RecurrentPPO は `sb3_contrib`、PPO は `stable_baselines3` 本体から import する。
  RecurrentPPO は `MlpLstmPolicy`（LSTM 方策）で、再生時に隠れ状態の引き継ぎが必須。

### 環境の並列化（SubprocVecEnv）

学習用環境は `make_vec_env(..., vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv)`。
n_envs=1 では subprocess 起動コストを避けて DummyVecEnv にフォールバックする。
macOS の multiprocessing は `spawn` のため `if __name__ == "__main__"` ガードが必須（実装済み）。

コールバック頻度（`CheckpointCallback.save_freq` / `EvalCallback.eval_freq`）は VecEnv では
vec-step 単位で数えられるため、総タイムステップ基準を保つよう `// n_envs` でスケールしている。
評価環境（`eval_env`）は単一環境のまま（並列化は学習用 env のみ）。

### 報酬整形（転倒ペナルティの緩和）

`gym_utils.FallPenaltyWrapper` が転倒時の報酬 `-100`（`bipedal_walker.py` で転倒時のみ
`reward = -100` に代入される）を `--fall-penalty`（既定 `-40`）に置き換える。各 train() が
`make_vec_env(wrapper_class=FallPenaltyWrapper, wrapper_kwargs=...)` で**学習用 env にのみ**適用し、
eval_env と play() は素の報酬を使う（best_model 選定と `eval/mean_reward` を真の性能に保つため）。
SB3 では `wrapper_class` は Monitor の**外側**に入る（`env_util.py:112-115`）ため、Monitor が記録する
`rollout/ep_rew_mean` は素の報酬（`-100` 込み）のまま＝緩和は勾配にのみ効く。`--fall-penalty -100`
でラッパーを付けず元の挙動に戻せる。SubprocVecEnv(spawn) で pickle されるためモジュールレベルのクラス。

### 録画・再生は gym_utils に集約

`gym_utils.record_agent_video(agent, env_id, video_folder)` が全アルゴリズム共通の再生ループ
（reset → predict → step → close → `display_video`）。Jupyter 上では mp4 を埋め込み表示、
通常実行ではパス表示 + OS 既定プレーヤで自動再生（`GYM_UTILS_NO_OPEN=1` で抑止可）。
再生パイプラインに触れる変更はここ 1 箇所で済む。`record_agent_video` / `display_video` は
後方互換に `name_prefix`（出力名 prefix で履歴を分ける）と `open_player`（プレーヤを開くか）を持つ。
`record_agent_video` はさらに `recurrent`（既定 False）を持ち、`True` のとき predict に
`state` / `episode_start` を渡して LSTM 隠れ状態をステップ間で引き継ぐ（RecurrentPPO 専用。
渡さないと毎ステップ隠れ状態がゼロに戻り再帰方策が正しく動かない）。`recurrent_ppo.py` のみ
`play()` がこれを `True` で呼ぶ。`numpy` は `episode_start` 配列の生成に使う。

### 学習中の進捗動画コールバック（sac.py のみ）

`gym_utils.RecordBestVideoCallback` を `EvalCallback(callback_on_new_best=...)` に渡すと、
**best_model が更新された瞬間**にその時点の `self.model` から 1 エピソードを録画する。SB3 2.4.0 では
`EvalCallback._init_callback` が `callback_on_new_best.init_callback(self.model)` を呼ぶため、
コールバック内で `self.model`（=今ベストになった現行モデル）・`self.num_timesteps` を参照できる。
best_model.zip を再ロードしないので保存との書き込み競合が無い。早期は best が頻繁に更新されるため
`min_interval_steps`（`--progress-video-every`）で最低 step 間隔を空けて間引く。動画は
`best-step<総step数>-episode-0.mp4` と step 入りの名前で**上書きせず**溜まる。録画失敗は try/except で
握りつぶし、長時間学習を絶対に止めない。`--no-progress-video` で無効化できる。

### 成果物とログ

- **sac.py（新構造）**: 1 回の学習を `sac_runs_bipedalwalkerhardcore/<YYYYMMDD-HHMMSS>/` に隔離する。
  配下に `logs/`（チェックポイント・monitor・評価ログ）、`best_model/best_model.zip`、
  `videos/progress/*.mp4`（進捗動画）、`videos/play/*.mp4`（再生）、`final_model.zip`。
  **同じ設定で再学習しても過去 run が消えない**。`play()` は最新 run（または `--run` 指定）を再生し、
  run が無ければ旧構造の固定パスにフォールバックする。TensorBoard の run 名は `SAC_<ts>`。
- **a2c/ddpg/td3/trpo/ppo/recurrent_ppo（旧構造のまま）**: `<algo>_logs_bipedalwalkerhardcore/`
  （チェックポイント・評価ログ・`best_model/best_model.zip`）、`<algo>_bipedalwalkerhardcore.zip`（最終モデル）、
  `<algo>-bipedalwalkerhardcore_videos_practice/*.mp4`。
- `tensorboard/` は全アルゴリズム共有。これらはすべて `.gitignore` 済み（run ルートは
  `*_runs_bipedalwalker*/`）。コミット対象はスクリプト本体・`requirements.txt`・`setup.sh`・
  `README.md`・移植元ノートブックのみ。

## 注意点

- **環境はハードコア版**で通常版より格段に難しく、意味のある方策には数百万ステップ規模が必要。
  既定の `--timesteps 2000` は「最後までエラーなく動くか」の動作確認用。観測 24 次元・行動 4 次元は
  通常版と同じなので、`BipedalWalker-v3` に戻すなら各スクリプト冒頭の `ENV_ID` を変えるだけ。
- 依存はバージョン固定（`gymnasium[box2d]==1.0.0` / `stable-baselines3==2.4.0` /
  `sb3-contrib==2.4.0`）。`sb3_contrib` は TRPO / RecurrentPPO に必須（PPO は sb3 本体）。
  `box2d-py` のビルドに `swig`、動画書き出しに `ffmpeg` が必要（`setup.sh` が前者を導入）。
- `step()` は Gymnasium の 5 値 API（`obs, reward, terminated, truncated, info`）。`import gymnasium as gym`。
