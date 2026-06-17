# Repository Guidelines

## Project Structure & Module Organization

This repository is a local Gymnasium/Stable-Baselines3 teaching project for `BipedalWalkerHardcore-v3`. The active runtime code lives at the repository root:

- `a2c.py`, `ddpg.py`, `td3.py`, `sac.py`: one training/playback script per algorithm.
- `gym_utils.py`: shared video recording and display helpers.
- `requirements.txt` and `setup.sh`: pinned Python dependencies and environment setup.
- `README.md`: user-facing setup, usage, and troubleshooting notes.
- `10_DeepReinforcementLearning...ipynb` and the matching `.py`: original notebook-derived source material.

Generated models, logs, videos, `tensorboard/`, `.venv/`, and `__pycache__/` are ignored and should not be committed.

## Build, Test, and Development Commands

- `bash setup.sh`: create `.venv`, install dependencies, check `swig`, and verify Gymnasium can create the environment.
- `.venv/bin/python a2c.py`: run the default A2C train-and-play pipeline.
- `.venv/bin/python sac.py --timesteps 50000`: run a longer SAC training job.
- `.venv/bin/python sac.py --mode train`: train without opening playback.
- `.venv/bin/python sac.py --n-envs 8`: collect experience with 8 parallel `SubprocVecEnv` workers (`1` falls back to `DummyVecEnv`).
- `.venv/bin/python sac.py --fall-penalty -40`: soften the `-100` fall penalty to `-40` during training (`-100` disables shaping). Applied to the training env only; eval/playback keep the raw reward.
- `GYM_UTILS_NO_OPEN=1 .venv/bin/python td3.py --mode play`: replay a saved model without launching the OS video player.
- `.venv/bin/python -m tensorboard.main --logdir tensorboard/`: inspect training metrics.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, `snake_case` functions, and uppercase module constants such as `ENV_ID`, `LOG_DIR`, and `FINAL_MODEL`. Keep imports grouped as standard library, third-party, then local imports. The four algorithm scripts intentionally share the same skeleton: constants, `train(timesteps, n_envs, fall_penalty)`, `play()`, and `main()`. When changing CLI options, logging, checkpoint behavior, or playback flow, update all affected scripts consistently. Put cross-algorithm helpers—video recording and the `FallPenaltyWrapper` reward shaping—in `gym_utils.py`.

## Testing Guidelines

There is no formal test suite or coverage target. Before submitting code changes, run a lightweight syntax check:

```bash
for f in sac a2c ddpg td3 gym_utils; do .venv/bin/python -m py_compile "$f.py"; done
```

For behavior changes, also run a short smoke test with `--timesteps 2000`. Note that SAC uses `learning_starts=3000`, so use more than 3000 timesteps when validating actual SAC learning.

## Commit & Pull Request Guidelines

Recent history uses short conventional-style subjects such as `feat: ...` and `docs: ...`; follow that pattern. Pull requests should include a concise summary, validation commands run, any changed algorithm parameters, and screenshots or video paths when playback or slides change. Do not include generated `.zip` models, log directories, TensorBoard data, or mp4 outputs.
