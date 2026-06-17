# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## このリポジトリについて

Stable-Baselines3 + Gymnasium で 4 つの深層強化学習アルゴリズム（A2C / DDPG / TD3 / SAC）を
`BipedalWalkerHardcore-v3` で「学習 → ベストモデル保存 → 録画 → 再生」する教材用サンプル集。
元は単一の Colab ノートブック（`10_DeepReinforcementLearning_..._ipynb` / 同名 `.py`）で、
そこから Colab 固有処理を除いてアルゴリズムごとのローカル実行スクリプトに分割したもの。
ノートブックと `.py`（移植元）はリポジトリに残してあるが、現役のコードは 4 スクリプト + `gym_utils.py`。

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

# 構文チェック（変更後に必ず走らせる軽量な検証）
for f in sac a2c ddpg td3; do .venv/bin/python -m py_compile $f.py; done

# 学習過程の可視化（全アルゴリズム共通の tensorboard/ に出力）
.venv/bin/python -m tensorboard.main --logdir tensorboard/
```

このリポジトリにテストスイートは無い。検証は「`py_compile` で構文確認」＋「短い `--timesteps` で
スモーク実行（パイプラインが最後まで通るか）」で行う。SAC は `learning_starts=3000` のため、
学習が実際に走るか確かめるには `--timesteps` を 3000 超（例 8000）にする必要がある。

## アーキテクチャ

### 4 スクリプトは同一スケルトンの「並行コピー」

`a2c.py` / `ddpg.py` / `td3.py` / `sac.py` は同じ構造を共有する:

- モジュール冒頭の定数 `ENV_ID` / `LOG_DIR` / `VIDEO_FOLDER` / `FINAL_MODEL`
- `train(timesteps, n_envs)` — 環境構築 → コールバック → モデル構築 → `learn` → 保存
- `play()` — `best_model` があればそれを、無ければ `FINAL_MODEL` をロードして 1 エピソード録画
- `main()` — argparse（`--timesteps` / `--n-envs` / `--mode`）→ 経過時間を測って出力

**重要**: 共通の挙動を変えるときは 4 ファイルすべてを同じように編集する必要がある（共通基底クラスは無い、意図的な重複）。`ddpg.py` / `td3.py` / `sac.py` はほぼ同型、`a2c.py` だけ下記の通り異なる。

### オンポリシー（A2C）とオフポリシー（DDPG/TD3/SAC）の差

スクリプト横断の変更で最も間違えやすい軸。両者で適用すべきパラメータが違う:

- **オフポリシー（ddpg/td3/sac）**: リプレイバッファを持つ。`buffer_size`（現在 1M）、
  `learning_starts`、`train_freq`、`gradient_steps=-1`、（td3 のみ `policy_delay`）が効く。
  `gradient_steps=-1` は「収集ステップ数(=n_envs)ぶん勾配更新」で並列時も更新比 1:1 を維持する。
  DDPG/TD3 は探索用に `NormalActionNoise` を使う。
- **オンポリシー（a2c）**: リプレイバッファが無い。`buffer_size` / `gradient_steps` は存在しない。
  実効バッチ = `n_steps`(=5) × `n_envs` で制御する。**A2C には buffer_size 等を足さないこと。**

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
再生パイプラインに触れる変更はここ 1 箇所で済む。

### 成果物とログ

`<algo>_logs_bipedalwalkerhardcore/`（チェックポイント・評価ログ・`best_model/best_model.zip`）、
`<algo>_bipedalwalkerhardcore.zip`（最終モデル）、`<algo>_..._videos_practice/*.mp4`、
`tensorboard/`（全アルゴリズム共有）。これらはすべて `.gitignore` 済みで、コミット対象は
スクリプト本体・`requirements.txt`・`setup.sh`・`README.md`・移植元ノートブックのみ。

## 注意点

- **環境はハードコア版**で通常版より格段に難しく、意味のある方策には数百万ステップ規模が必要。
  既定の `--timesteps 2000` は「最後までエラーなく動くか」の動作確認用。観測 24 次元・行動 4 次元は
  通常版と同じなので、`BipedalWalker-v3` に戻すなら各スクリプト冒頭の `ENV_ID` を変えるだけ。
- 依存はバージョン固定（`gymnasium[box2d]==1.0.0` / `stable-baselines3==2.4.0`）。
  `box2d-py` のビルドに `swig`、動画書き出しに `ffmpeg` が必要（`setup.sh` が前者を導入）。
- `step()` は Gymnasium の 5 値 API（`obs, reward, terminated, truncated, info`）。`import gymnasium as gym`。
