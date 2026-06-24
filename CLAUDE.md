# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## このリポジトリについて

Stable-Baselines3 + Gymnasium で 9 つの深層強化学習アルゴリズム
（A2C / DDPG / TD3 / SAC / TRPO / PPO / RecurrentPPO / TQC / CrossQ）を
`BipedalWalkerHardcore-v3` で「学習 → ベストモデル保存 → 録画 → 再生」する教材用サンプル集。
元は Colab ノートブック 2 本（`10_..._ipynb` = A2C/DDPG/TD3/SAC、`11_..._ipynb` = TRPO/PPO/RecurrentPPO。
それぞれ同名 `.py` も）で、そこから Colab 固有処理を除いてアルゴリズムごとのローカル実行スクリプトに
分割したもの。ノートブックと `.py`（移植元）はリポジトリに残してあるが、現役のコードは
9 スクリプト + `gym_utils.py`。`11_` は素の `BipedalWalker-v3` を使っていたが、既存スクリプトに
合わせて `BipedalWalkerHardcore-v3` に統一した。TRPO / RecurrentPPO / TQC / CrossQ は `sb3_contrib` を要する。

`tqc.py` / `crossq.py` は「ハードモードで**最速ゴール**を狙う」ために後から追加したもので、
`sac.py` スケルトン（タイムスタンプ別 run・進捗動画）の並行コピー。TQC は本環境の実績トップの
オフポリシー手法、CrossQ は標本効率の高い新手法で、両者を比較できる。両スクリプトは学習用 env に
前進速度ボーナス（`gym_utils.SpeedRewardWrapper`、`--speed-coef`）を加えてゴール所要時間を縮め、
`--max-episodes`（既定 5000）で学習エピソード数を打ち切り、`play()` で `measure_goal_time` により
ゴール所要時間（秒）と成功率を計測する。`learning_starts=10000` のためスモークは `--timesteps 12000` 以上で。

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
.venv/bin/python ppo.py --mode play --run 20260617-164530  # 特定 run を再生（既定は最新 run。全9共通）
# PPO は zoo 準拠でチューン済み（VecNormalize/device=cpu/減衰スケジュール）。既定 --timesteps は 1,000,000 / --n-envs 8。
.venv/bin/python ppo.py --timesteps 3000000 --n-envs 8 --mode train  # PPO 本格学習（数百万〜。CPU+8並列で短時間化）
# 以下の進捗動画オプションは sac.py / tqc.py / crossq.py のみ（他6スクリプトには未実装）
.venv/bin/python sac.py --progress-video-every 20000  # best更新時の進捗動画を最低2万step間隔で録画（既定 20000 / 0 で毎回）
.venv/bin/python sac.py --no-progress-video      # 学習中の進捗動画録画を無効化

# 最速ゴール狙いの追加スクリプト（tqc.py / crossq.py。sac.py と同じオプション + 下記2つ）
.venv/bin/python tqc.py --timesteps 2000000 --max-episodes 5000  # 本番学習（5000 エピソードで打ち切り）
.venv/bin/python tqc.py --speed-coef 0.3         # 前進速度ボーナス係数（既定 0.3 / 0 で速度ボーナス無効）
.venv/bin/python crossq.py --timesteps 12000     # スモーク（learning_starts=10000 超で勾配更新が走る）

# 構文チェック（変更後に必ず走らせる軽量な検証。gym_utils も含める）
for f in sac a2c ddpg td3 trpo ppo recurrent_ppo tqc crossq gym_utils; do .venv/bin/python -m py_compile $f.py; done

# 学習過程の可視化（全アルゴリズム共通の tensorboard/ に出力）
.venv/bin/python -m tensorboard.main --logdir tensorboard/
```

このリポジトリにテストスイートは無い。検証は「`py_compile` で構文確認」＋「短い `--timesteps` で
スモーク実行（パイプラインが最後まで通るか）」で行う。SAC は `learning_starts=3000` のため、
学習が実際に走るか確かめるには `--timesteps` を 3000 超（例 8000）にする必要がある。

## アーキテクチャ

### 9 スクリプトは同一スケルトンの「並行コピー」

`a2c.py` / `ddpg.py` / `td3.py` / `sac.py` / `trpo.py` / `ppo.py` / `recurrent_ppo.py` / `tqc.py` / `crossq.py` は同じ構造を共有する:

- モジュール冒頭の定数 `ENV_ID` / `ALGO`（成果物の出力先 `runs/<ALGO>/<run_id>/` を決める）
- `train(timesteps, n_envs, fall_penalty)` — run ディレクトリ生成 → 環境構築 → コールバック → モデル構築 → `learn` → 保存。**この run ディレクトリのパスを返す**（tqc/crossq は `speed_coef` / `max_episodes` 等を追加で取る）
- `play(run=None)` — run（id/パス。既定は最新 run）を解決し、`best_model` があればそれを、無ければ `final_model` をロードして 1 エピソード録画
- `main()` — argparse（`--timesteps` / `--n-envs` / `--fall-penalty` / `--run` / `--mode`）→ 経過時間を測って出力

**重要**: 共通の挙動を変えるときはスクリプトすべてを同じように編集する必要がある（共通基底クラスは無い、意図的な重複）。run ディレクトリのパス計算は `gym_utils` の共通ヘルパー（`new_run_dir` / `latest_run_dir` / `resolve_run_dir` / `resolve_model_path`）に集約済み。`ddpg.py` / `td3.py` は同型、`a2c.py` / `trpo.py` / `ppo.py` / `recurrent_ppo.py` はオンポリシー版で下記の通り異なる。`recurrent_ppo.py` だけ `play()` が `record_agent_video(..., recurrent=True)` を呼び、再生時に LSTM 隠れ状態を引き継ぐ（他は省略=非再帰）。

**進捗動画コールバックを持つのは sac/tqc/crossq の3つ**: run 単位の出力隔離・`--run`・タイムスタンプ別フォルダは**全9スクリプト共通**（`train()` が `runs/<ALGO>/<run_id>/` を作って返し、`play(run=None)` が最新 run を解決）。`sac.py` / `tqc.py` / `crossq.py` だけが追加で**学習中の進捗動画コールバック**を持つ（`--progress-video-every` / `--no-progress-video`）。共通部品（`gym_utils.RecordBestVideoCallback` 等）は用意済みなので、必要なら同型で他へ展開できる。

**例外（ppo.py だけの強化）**: `ppo.py` のみ rl-baselines3-zoo の BipedalWalkerHardcore-v3 チューニング済み設定に合わせてあり、他8スクリプトと**意図的に異なる**:
- **VecNormalize**（観測+報酬の移動平均正規化）を学習 env に被せる（`norm_obs/norm_reward=True, clip_obs=10, gamma=0.99`）。連続制御 PPO の成否を分ける必須級。学習後 `vec_env.save(run_dir/vecnormalize.pkl)` で統計を保存し、`play()` は `VecNormalize.load` した統計から `normalize_obs` を作って `record_agent_video(..., obs_transform=...)` に渡す（旧 run に pkl が無ければ `obs_transform=None` で素の観測＝後方互換）。eval_env も VecNormalize で包むが `norm_reward=False, training=False`（best 選定と `eval/mean_reward` を真の報酬で行う。統計は EvalCallback が `sync_envs_normalization` で評価前に自動同期）。
- **device="cpu" を明示**。MlpPolicy の PPO は CPU が最速（SB3 公式・実測 CPU>GPU）。Mac の MPS は SB3 で失敗するので使わない。速度は SubprocVecEnv 並列で稼ぐ。
- ハイパラは zoo 準拠: `gae_lambda=0.95` / `ent_coef=0.001` / `learning_rate=linear_schedule(2.5e-4)` / `clip_range=linear_schedule(0.2)`（`lin_` 相当の線形減衰。モジュールの `linear_schedule()` で実装）。`vf_coef`/`max_grad_norm`/`policy_kwargs` は SB3 既定。
- コールバック頻度は本格学習向けに緩め（`save_freq=1_000_000//n_envs` / `eval_freq=100_000//n_envs`, `n_eval_episodes=10`）。
- **既定値の乖離**: `--timesteps` 既定は **1,000,000**（他8は 2000=動作確認用）、`--n-envs` 既定は **8**（他8は 4）。
- **注意**: best_model は途中保存、`vecnormalize.pkl` は学習終了時点の統計なので厳密には時点がずれるが、観測統計は緩やかに収束するため通常問題ない（rl-zoo の enjoy も同方式）。共通挙動を変える編集を全9に展開する際、これらの PPO 専用差分を**素朴にコピーで潰さないこと**。

### オンポリシー（A2C/TRPO/PPO/RecurrentPPO）とオフポリシー（DDPG/TD3/SAC）の差

スクリプト横断の変更で最も間違えやすい軸。両者で適用すべきパラメータが違う:

- **オフポリシー（ddpg/td3/sac/tqc/crossq）**: リプレイバッファを持つ。`buffer_size`（現在 1M）、
  `learning_starts`、`train_freq`、`gradient_steps`、（td3 のみ `policy_delay`）が効く。
  sac は `gradient_steps=-1`（「収集ステップ数(=n_envs)ぶん勾配更新」で並列時も更新比 1:1 を維持）。
  DDPG/TD3 は探索用に `NormalActionNoise` を使う。
  - **tqc.py（TQC, sb3_contrib）**: SAC を分位点クリティックへ拡張し過大評価を抑えた手法。RL-Zoo の
    BipedalWalkerHardcore レシピを採用＝`net_arch=[400,300]` / `learning_rate=linear_schedule(7.3e-4)`
    （線形減衰 `lin_7.3e-4`）/ `tau=0.01` / `gradient_steps=1`（sac の -1 と違い 1 固定）/ `learning_starts=10000`。
  - **crossq.py（CrossQ, sb3_contrib 2.4.0+）**: BatchNorm クリティック＋ターゲットネット撤廃で標本効率が高い。
    SAC とは `tau`/`train_freq`/`gradient_steps`/`ent_coef`/`learning_rate` の意味が異なるため**渡さず既定に委ねる**
    （最小引数で頑健に）。非対称 `net_arch=dict(pi=[256,256], qf=[1024,1024])`、`buffer_size=1M`、`learning_starts=10000`。
  - 両者とも `--speed-coef`（`SpeedRewardWrapper` の前進速度ボーナス）と `--max-episodes`
    （`StopTrainingOnMaxEpisodes`。**per-env × n_envs** が総数なので train() で `// n_envs` して渡す）を持つ。
- **オンポリシー（a2c/trpo/ppo/recurrent_ppo）**: リプレイバッファが無い。`buffer_size` /
  `gradient_steps` / `learning_starts` は存在しない。実効バッチ = `n_steps` × `n_envs` で制御する
  （a2c=5 / trpo=500 / ppo=2048 / recurrent_ppo=128）。ppo/recurrent_ppo は `batch_size` /
  `n_epochs` / `clip_range`、trpo は `gae_lambda` を持つ。**これらに buffer_size 等を足さないこと。**
  TRPO / RecurrentPPO は `sb3_contrib`、PPO は `stable_baselines3` 本体から import する。
  RecurrentPPO は `MlpLstmPolicy`（LSTM 方策）で、再生時に隠れ状態の引き継ぎが必須。
  **PPO だけ**は zoo 準拠でさらに `gae_lambda`/`ent_coef`/`device="cpu"`/VecNormalize/減衰スケジュールを持つ
  （上の「例外（ppo.py だけの強化）」参照）。他のオンポリシー（a2c/trpo/recurrent_ppo）には未展開。

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

`tqc.py` / `crossq.py` は `FallPenaltyWrapper` の上位互換 `gym_utils.SpeedRewardWrapper`
（`fall_penalty` に加え `speed_coef` を取る）を使う。非転倒ステップで前進速度 `obs[2]`
（正規化水平速度、正で前進）に比例した `speed_coef * max(0, obs[2])` を加点し、「同じゴールでも
より速く着く」方策へ誘導する（`--speed-coef`、既定 0.3 / 0 で `FallPenaltyWrapper` と等価）。
これも**学習用 env のみ**。再生時は `gym_utils.measure_goal_time` が**素の env**で複数エピソード走らせ、
ゴール到達ステップ数→秒（`steps/fps`）・成功率を集計表示する（目標 ~17 秒 ≒ 850 step ＠50FPS）。
成功判定は「`terminated` かつ最終報酬 > -100（右端到達）」、`fall`=`terminated` かつ報酬 -100、
`timeout`=`truncated`。

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
`record_agent_video` はもう 1 つ `obs_transform`（既定 None＝変換なし）を持ち、predict 直前に観測を
変換する（env へ返す obs 自体は素のまま）。`ppo.py` のみ `play()` が VecNormalize の `normalize_obs` を
渡し、学習時の統計で観測を正規化して再生する（他8は None）。

### run ディレクトリ管理も gym_utils に集約

成果物の出力先計算は全9スクリプト共通なので `gym_utils` のヘルパーに集約してある:
`new_run_dir(algo)`（`runs/<algo>/<YYYYMMDD-HHMMSS-pid>/` を作成して返す。`logs/` も作る）、
`latest_run_dir(algo)`（辞書順最大の run。`_`/`.` 始まりは除外）、
`resolve_run_dir(algo, run=None)`（run id/パス指定 or 最新 run を解決）、
`resolve_model_path(run_dir)`（run 内で `best_model` → `final_model` の順に解決）。
各 `train()` は `new_run_dir(ALGO)` で run を作り、`play()` は `resolve_run_dir` + `resolve_model_path`
で解決する。run パスの規約を変えるときはここ 1 箇所で済む。

### 学習中の進捗動画コールバック（sac.py / tqc.py / crossq.py）

`gym_utils.RecordBestVideoCallback` を `EvalCallback(callback_on_new_best=...)` に渡すと、
**best_model が更新された瞬間**にその時点の `self.model` から 1 エピソードを録画する。SB3 2.4.0 では
`EvalCallback._init_callback` が `callback_on_new_best.init_callback(self.model)` を呼ぶため、
コールバック内で `self.model`（=今ベストになった現行モデル）・`self.num_timesteps` を参照できる。
best_model.zip を再ロードしないので保存との書き込み競合が無い。早期は best が頻繁に更新されるため
`min_interval_steps`（`--progress-video-every`）で最低 step 間隔を空けて間引く。動画は
`best-step<総step数>-episode-0.mp4` と step 入りの名前で**上書きせず**溜まる。録画失敗は try/except で
握りつぶし、長時間学習を絶対に止めない。`--no-progress-video` で無効化できる。

### 成果物とログ

- **全9スクリプト共通（run 構造）**: 1 回の学習を `runs/<algo>/<YYYYMMDD-HHMMSS-pid>/` に隔離する。
  配下に `logs/`（チェックポイント・monitor・評価ログ）、`best_model/best_model.zip`、
  `videos/play/*.mp4`（再生）、`final_model.zip`。SAC / TQC / CrossQ は `videos/progress/*.mp4`（進捗動画）、
  PPO のみ `vecnormalize.pkl`（VecNormalize 統計。play で観測正規化に使う）も持つ。
  **同じ設定で再学習しても過去 run が消えない**。`play(run=None)` は最新 run（または `--run` 指定）を再生し、
  run 内では `best_model` → `final_model` の順にフォールバックする。TensorBoard の run 名は `<ALGO大文字>_<run_id>`。
- 旧構造（`<algo>_logs_bipedalwalkerhardcore/` 等の固定パスや `sac_runs_bipedalwalkerhardcore/`）は
  `runs/<algo>/<ts>-legacy/` に移行済み。旧構造で学習した成果物も同じ run レイアウトで再生できる。
  移行できなかった素の stdout ログは `runs/_stdout_logs/`（run ではない）に退避してある。
- `tensorboard/` は全アルゴリズム共有。これらはすべて `.gitignore` 済み（成果物ルートは `runs/`）。
  コミット対象はスクリプト本体・`requirements.txt`・`setup.sh`・`README.md`・移植元ノートブックのみ。

## 注意点

- **環境はハードコア版**で通常版より格段に難しく、意味のある方策には数百万ステップ規模が必要。
  既定の `--timesteps 2000`（PPO のみ 1,000,000）は「最後までエラーなく動くか」の動作確認用
  （PPO の 1M は意味のある最小規模）。観測 24 次元・行動 4 次元は通常版と同じなので、
  `BipedalWalker-v3` に戻すなら各スクリプト冒頭の `ENV_ID` を変えるだけ。なお PPO 単体では
  Hardcore を「解く」(平均+300)のは事実上不可（zoo ベンチで 100M でも 122±117）。本格学習でも
  歩き始め〜部分的な前進の観察が現実的なゴール。off-policy の TQC/SAC が本質的にサンプル効率で勝る。
- 依存はバージョン固定（`gymnasium[box2d]==1.0.0` / `stable-baselines3==2.4.0` /
  `sb3-contrib==2.4.0`）。`sb3_contrib` は TRPO / RecurrentPPO / TQC / CrossQ に必須（PPO は sb3 本体）。
  CrossQ は `sb3-contrib 2.4.0` で追加されたため、それ未満では `crossq.py` の import が失敗する。
  `box2d-py` のビルドに `swig`、動画書き出しに `ffmpeg` が必要（`setup.sh` が前者を導入）。
- `step()` は Gymnasium の 5 値 API（`obs, reward, terminated, truncated, info`）。`import gymnasium as gym`。
