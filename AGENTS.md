# Repository Guidelines

## Project Structure & Module Organization

This repository is a local Gymnasium/Stable-Baselines3 teaching project for `BipedalWalkerHardcore-v3`. The active runtime code lives at the repository root:

- `a2c.py`, `ddpg.py`, `td3.py`, `sac.py`, `trpo.py`, `ppo.py`, `recurrent_ppo.py`, `tqc.py`, `crossq.py`: one training/playback script per algorithm. TRPO, RecurrentPPO, TQC, and CrossQ require `sb3_contrib` (CrossQ needs `sb3-contrib>=2.4.0`); PPO ships in `stable_baselines3`. `tqc.py`/`crossq.py` are the "fastest-goal" scripts (forward-speed bonus + goal-time measurement), built on the `sac.py` skeleton.
- `gym_utils.py`: shared video recording/display helpers, the `FallPenaltyWrapper`/`SpeedRewardWrapper` reward shaping, `RecordBestVideoCallback`, `measure_goal_time`, and `linear_schedule`.
- `requirements.txt` and `setup.sh`: pinned Python dependencies and environment setup.
- `README.md`: user-facing setup, usage, and troubleshooting notes.
- `10_...ipynb` (A2C/DDPG/TD3/SAC) and `11_...ipynb` (TRPO/PPO/RecurrentPPO), plus the matching `.py`: original notebook-derived source material.

Generated models, logs, videos, `tensorboard/`, `.venv/`, and `__pycache__/` are ignored and should not be committed.

## Build, Test, and Development Commands

- `bash setup.sh`: create `.venv`, install dependencies, check `swig`, and verify Gymnasium can create the environment.
- `.venv/bin/python a2c.py`: run the default A2C train-and-play pipeline.
- `.venv/bin/python sac.py --timesteps 50000`: run a longer SAC training job.
- `.venv/bin/python sac.py --mode train`: train without opening playback.
- `.venv/bin/python sac.py --n-envs 8`: collect experience with 8 parallel `SubprocVecEnv` workers (`1` falls back to `DummyVecEnv`).
- `.venv/bin/python sac.py --fall-penalty -40`: soften the `-100` fall penalty to `-40` during training (`-100` disables shaping). Applied to the training env only; eval/playback keep the raw reward.
- `.venv/bin/python sac.py --progress-video-every 20000` (sac.py only): record a progress video from the current best model whenever `EvalCallback` saves a new best, throttled to at least 20000 steps apart (`0` = every new best). Disable with `--no-progress-video`.
- `.venv/bin/python sac.py --mode play --run 20260617-164530` (all scripts): replay a specific timestamped run; default replays the latest run.
- `.venv/bin/python tqc.py --timesteps 2000000 --max-episodes 5000` (tqc.py/crossq.py only): train the fastest-goal TQC/CrossQ agent, stopping at 5000 total episodes. Use `--speed-coef` to tune the forward-speed reward bonus (`0` disables it). `learning_starts=10000`, so smoke-test with `--timesteps 12000`.
- `GYM_UTILS_NO_OPEN=1 .venv/bin/python td3.py --mode play`: replay a saved model without launching the OS video player.
- `.venv/bin/python -m tensorboard.main --logdir tensorboard/`: inspect training metrics.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, `snake_case` functions, and uppercase module constants such as `ENV_ID` and `ALGO`. Keep imports grouped as standard library, third-party, then local imports. The nine algorithm scripts intentionally share the same skeleton: constants, `train(...)` (returns the run dir path), `play(run=None)`, and `main()`. The base `train` signature is `train(timesteps, n_envs, fall_penalty)`; `sac.py` adds progress-video args, and `tqc.py`/`crossq.py` extend it further to `train(timesteps, n_envs, fall_penalty, speed_coef, max_episodes, progress_video_every, progress_video)`. When changing CLI options, logging, checkpoint behavior, or playback flow, update all affected scripts consistently. Note the on-policy scripts (`a2c.py`, `trpo.py`, `ppo.py`, `recurrent_ppo.py`) have no replay buffer; `recurrent_ppo.py` additionally calls `record_agent_video(..., recurrent=True)` so its LSTM hidden state carries across playback steps. Put cross-algorithm helpers—video recording, the `FallPenaltyWrapper`/`SpeedRewardWrapper` reward shaping, `RecordBestVideoCallback`, `measure_goal_time`, `linear_schedule`, and the run-directory helpers (`new_run_dir`, `latest_run_dir`, `resolve_run_dir`, `resolve_model_path`)—in `gym_utils.py`. All nine scripts isolate each training run under `runs/<algo>/<YYYYMMDD-HHMMSS-pid>/` (containing `logs/`, `best_model/`, `videos/play/`, and `final_model.zip`); `play(run=None)` resolves the latest run (or `--run <id|path>`). `sac.py`/`tqc.py`/`crossq.py` additionally record progress videos on each new best (`--progress-video-*`); `tqc.py`/`crossq.py` also add the forward-speed bonus (`--speed-coef`), the episode cap (`--max-episodes`, via `StopTrainingOnMaxEpisodes` divided by `n_envs`), and report goal time via `measure_goal_time` in `play()`.

## Testing Guidelines

There is no formal test suite or coverage target. Before submitting code changes, run a lightweight syntax check:

```bash
for f in sac a2c ddpg td3 trpo ppo recurrent_ppo tqc crossq gym_utils; do .venv/bin/python -m py_compile "$f.py"; done
```

For behavior changes, also run a short smoke test with `--timesteps 2000`. Note that SAC uses `learning_starts=3000`, and `tqc.py`/`crossq.py` use `learning_starts=10000`, so use more timesteps than that (e.g. `--timesteps 12000`) when validating that actual learning runs.

## Commit & Pull Request Guidelines

Recent history uses short conventional-style subjects such as `feat: ...` and `docs: ...`; follow that pattern. Pull requests should include a concise summary, validation commands run, any changed algorithm parameters, and screenshots or video paths when playback or slides change. Do not include generated `.zip` models, log directories, TensorBoard data, or mp4 outputs.
