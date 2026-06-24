# 様々な深層強化学習アルゴリズム（ローカル実行版）

StableBaselines3 を用いて、7 つの深層強化学習アルゴリズム
**A2C / DDPG / TD3 / SAC / TRPO / PPO / RecurrentPPO** を Gymnasium の
`BipedalWalkerHardcore-v3` 環境で「学習 → ベストモデル保存 → 録画 → 再生」する
サンプル集です。

元は Google Colab 用ノートブック 2 本
（`10_..._01_fixed_ipynb_のコピー.ipynb` = A2C/DDPG/TD3/SAC、
`11_..._02_fixed_ipynb_のコピー.ipynb` = TRPO/PPO/RecurrentPPO）でしたが、
Colab 固有の処理を取り除き、**ローカル環境で動く Python スクリプト**に移植しています。

> 使用しているのは OpenAI の旧 `gym` ではなく、その後継である
> **Gymnasium**（Farama Foundation, v1.0.0）です。`import gymnasium as gym`、
> `step()` は `obs, reward, terminated, truncated, info` の 5 値 API です。

---

## 前提

- **macOS**（Apple Silicon / Intel）または Linux
- **Python 3.10〜3.12**（動作確認: 3.12）
- **ffmpeg**（動画の書き出しに必要。未導入なら `brew install ffmpeg` / `sudo apt install ffmpeg`）
- **swig**（`box2d-py` のビルドに必要。`setup.sh` が自動導入）

---

## クイックスタート

```bash
bash setup.sh                      # 初回セットアップ（venv作成 + 依存インストール）
source .venv/bin/activate          # 仮想環境を有効化
python a2c.py                      # A2C を学習(2000step) → 録画 → 再生
python sac.py --timesteps 50000    # SAC をしっかり学習させる場合
```

> `setup.sh` は初回のみ実行すればOKです。2回目以降は `source .venv/bin/activate` から始めてください。
> 各コマンドの詳細は下の[セットアップ](#セットアップ)・[使い方](#使い方)を参照してください。

---

## セットアップ

### 推奨: ワンショットスクリプト

```bash
bash setup.sh
```

`setup.sh` は以下を自動で行います。

1. 仮想環境 `.venv` の作成
2. `swig` の確認／インストール（macOS は Homebrew、Linux は apt）
3. `requirements.txt` の依存パッケージのインストール
4. `BipedalWalkerHardcore-v3` が生成できるかの動作確認

### 手動セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# box2d-py のビルドに swig が必要（PyPI 版は環境により壊れるためシステム版を推奨）
brew install swig            # macOS
# sudo apt install swig      # Ubuntu/Debian

pip install -r requirements.txt
```

> **swig について**: `gymnasium[box2d]` は `box2d-py` を C/C++ からビルドするため、
> ビルド時に `swig` コマンドが必要です。PyPI の `swig` パッケージは一部環境
> （macOS arm64 等）でビルド時に壊れることがあるため、Homebrew / apt の
> システム版 swig の利用を推奨しています。

---

## 使い方

仮想環境を有効化してから各スクリプトを実行します。

```bash
source .venv/bin/activate

# オフポリシー（リプレイバッファあり）
python ddpg.py            # DDPG
python td3.py             # TD3
python sac.py             # SAC

# オンポリシー（リプレイバッファなし）
python a2c.py             # A2C  を学習(2000step) → ベストモデルを録画・再生
python trpo.py            # TRPO        （sb3_contrib）
python ppo.py             # PPO
python recurrent_ppo.py   # RecurrentPPO（sb3_contrib、LSTM 方策）
```

> **TRPO / RecurrentPPO は拡張パッケージ `sb3_contrib` を使います**（`requirements.txt` に同梱済み）。
> PPO は `stable_baselines3` 本体に含まれます。RecurrentPPO は LSTM を使うため、再生時に
> ステップ間で隠れ状態を引き継ぎます（`gym_utils.record_agent_video(..., recurrent=True)`）。

### 共通オプション

| オプション | 説明 | 既定値 |
|---|---|---|
| `--timesteps N` | 総学習ステップ数 | `2000`（動作確認用） |
| `--n-envs N` | 並列環境数（`SubprocVecEnv`。`1` で逐次=`DummyVecEnv`） | `4` |
| `--fall-penalty F` | 転倒時の報酬(`-100`)を緩和する置換値。`-100` で緩和なし（元の挙動） | `-40.0` |
| `--mode {train,play,both}` | `train`=学習のみ / `play`=保存済みモデルを再生のみ / `both`=学習して再生 | `both` |

```bash
python a2c.py --timesteps 50000      # しっかり学習させる
python a2c.py --mode train           # 学習だけ行う（録画・再生なし）
python a2c.py --mode play            # 学習済みモデルを再生するだけ
python a2c.py --fall-penalty -40     # 転倒ペナルティを -40 に緩和（既定）
python a2c.py --fall-penalty -100    # 緩和なし（元の挙動に戻す）
```

---

## 生成されるファイル

| パス | 内容 |
|---|---|
| `<algo>_logs_bipedalwalkerhardcore/` | チェックポイント、評価ログ |
| `<algo>_logs_bipedalwalkerhardcore/best_model/best_model.zip` | 評価で最高性能だったモデル（`play` はこれをロード） |
| `<algo>_bipedalwalkerhardcore.zip` | 学習終了時点の最終モデル |
| `<algo>_bipedalwalkerhardcore_videos_practice/rl-video-episode-0.mp4` | 再生時に録画した動画 |
| `tensorboard/` | TensorBoard 用ログ（全アルゴリズム共通） |

`<algo>` は `a2c` / `ddpg` / `td3` / `sac` / `trpo` / `ppo` / `recurrent_ppo`。

- **再生の挙動**: スクリプト実行時、`gym_utils.display_video` が録画した mp4 の
  パスを表示し、OS 既定のプレーヤで自動再生します。
  自動再生を止めたい場合は環境変数 `GYM_UTILS_NO_OPEN=1` を設定してください。
- Jupyter / Colab 上で実行した場合は、mp4 をセル内に HTML5 動画として埋め込み表示します。

---

## TensorBoard で学習過程を見る

別ターミナルで以下を実行し、ブラウザで `http://localhost:6006/` を開きます。

```bash
source .venv/bin/activate
tensorboard --logdir tensorboard/
```

全アルゴリズムが同じ `tensorboard/` に書き込むため、報酬曲線などを横並びで比較できます。

---

## ファイル構成

```
.
├── README.md          # このファイル
├── requirements.txt   # 依存パッケージ（バージョン固定）
├── setup.sh           # venv 作成 + 依存インストール
├── gym_utils.py       # display_video() / record_agent_video()（録画・再生ユーティリティ）
├── a2c.py             # A2C          （オンポリシー）
├── ddpg.py            # DDPG         （オフポリシー）
├── td3.py             # TD3          （オフポリシー）
├── sac.py             # SAC          （オフポリシー）
├── trpo.py            # TRPO         （オンポリシー / sb3_contrib）
├── ppo.py             # PPO          （オンポリシー）
└── recurrent_ppo.py   # RecurrentPPO （オンポリシー / sb3_contrib・LSTM）
```

---

## 注意点・トラブルシュート

- **環境はハードコア版（`BipedalWalkerHardcore-v3`）**: 通常版より格段に難しく
  （穴・段差・障害物があり、転倒で大きな負報酬）、学習には通常版よりはるかに多くの
  ステップ（一般に数百万ステップ規模）を要します。観測・行動の次元は通常版と同じ
  （24 / 4）です。通常版に戻したい場合は、各スクリプト先頭の
  `ENV_ID = "BipedalWalkerHardcore-v3"` を `"BipedalWalker-v3"` に変更してください。

- **SAC の `learning_starts=3000`**: SAC は「最初の 3000 ステップはランダム行動で
  経験を集めてから学習を開始」します。`--timesteps` を 3000 以下にすると勾配更新が
  一度も走らず、パイプラインの動作確認はできても方策はほぼ学習されません。
  実際に学習させる場合は `--timesteps` を大きく（元ノートブックは 50000）してください。

- **学習ステップ数と時間**: 既定の 2000 ステップは「最後までエラーなく動くか」の
  動作確認用です。意味のある方策を得るには数万〜数十万ステップが必要で、
  macOS の CPU/MPS 実行では時間がかかります。

- **転倒ペナルティの緩和（reward shaping）**: 既定で転倒時の報酬 `-100` を
  `--fall-penalty`（既定 `-40`）に置き換えて学習を安定化します（`-100` は転倒時のみ
  発生）。この置換は**学習用環境にのみ**適用され、評価・再生は素の報酬を使います。
  また置換は Monitor の外側で行われるため、`rollout/ep_rew_mean` も `eval/mean_reward` も
  転倒 `-100` 込みの真の報酬で表示されます（緩和は勾配にのみ効きます）。元の挙動に
  戻すには `--fall-penalty -100` を指定してください。

- **`box2d-py` のビルドに失敗する**: `swig` が見つからないことが原因です。
  `brew install swig`（macOS）/ `sudo apt install swig`（Linux）の後に
  `pip install -r requirements.txt` を再実行してください。

- **動画が生成されない / 真っ黒**: `ffmpeg` が未導入の可能性があります。
  `ffmpeg -version` で確認し、無ければ導入してください。

- **ヘッドレスな Linux サーバで動かす**: 描画に仮想ディスプレイが必要な場合があります。
  ```bash
  sudo apt install -y xvfb
  export SDL_VIDEODRIVER=dummy
  xvfb-run -a python a2c.py
  ```
  （macOS では `xvfb` は不要です。）

---

## 元の Colab からの主な変更点

| Colab | ローカル版 |
|---|---|
| `!apt install xvfb` / `python-pygame` | 不要なため削除（macOS はヘッドレス描画可） |
| `!gdown ...`（Drive から `gym_utils.py` 取得） | `gym_utils.py` をローカル用に再作成 |
| `%load_ext tensorboard` / `%tensorboard` | `tensorboard --logdir tensorboard/` を別ターミナルで実行 |
| `!pip install ...` | `requirements.txt` + `setup.sh` に集約（`sb3_contrib` も追加） |
| 1 ノートブックに全アルゴリズム | アルゴリズムごとの Python スクリプトに分割 |
| `BipedalWalker-v3`（11_ ノートブック） | 既存スクリプトに合わせ `BipedalWalkerHardcore-v3` に統一 |
