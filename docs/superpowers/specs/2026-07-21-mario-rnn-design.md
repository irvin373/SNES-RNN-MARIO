# Super Mario World RNN Player — Design Spec

Date: 2026-07-21

## Goal

Build an agent that plays Super Mario World (SNES) autonomously, learning through self-play/training rather than hand-coded rules. The policy is a recurrent neural network (CNN + LSTM) trained with reinforcement learning against a live SNES emulator.

This spec covers the first sub-project: a working end-to-end pipeline (emulator → observations → RNN policy → reward → PPO training → checkpoint → playback) proven on a single level. Multi-level generalization is an explicit follow-on sub-project, out of scope here.

## Non-goals (this spec)

- Multi-level or full-game generalization
- Distributed/cloud training infrastructure
- Any UI beyond CLI scripts
- Human-vs-agent or multiplayer modes

## Architecture

- **Emulator/env**: `stable-retro` (community-maintained successor to `gym-retro`), which wraps a libretro SNES core and exposes a Gym-style Python environment. Requires the user to supply a legally-owned SMW ROM, imported locally via retro's `import` CLI tool — this project never distributes or fetches ROM data.
- **Reward/done signal**: RAM-based. A custom retro integration data file (`data.json`) declares memory addresses for Mario's x-position, lives/death flag, score, coins, and level-clear flag. These drive reward shaping and episode termination — no pixel/OCR inference needed.
- **Observation**: Raw RGB frame → grayscale → resize to 84×84 → frame-skip of 4 (action repeated across skipped frames, reward accumulated). No frame-stacking: a single frame is passed per step, and the LSTM's hidden state carries motion/velocity information across time instead. This keeps the input tensor small, which matters for CPU-only training.
- **Action space**: A trimmed discrete action set covering d-pad directions plus B (run/jump) and A (jump) button combos relevant to platforming — not the full SNES button combinatorial space, which would be needlessly large for PPO to explore.
- **Policy network**: Small CNN feature extractor feeding an LSTM, via `sb3-contrib`'s `RecurrentPPO` with a `CnnLstmPolicy`. Policy and value heads read off the LSTM hidden state.
- **Training algorithm**: PPO (clipped surrogate objective), recurrent variant, single environment process to start (this Mac is CPU-only/Apple Silicon — no CUDA, and MPS has poor RNN op coverage in PyTorch, so training runs on CPU). Designed so a later switch to `SubprocVecEnv` with multiple parallel retro instances is a config change, not a rewrite.
- **Level**: Yoshi's Island 1 recommended as the first target — flat, simple, no gimmicks — but selectable via config, not hardcoded.

## Components

- `env/`
  - `retro_env.py` — constructs the base retro env (game, state/level, action set)
  - `wrappers.py` — preprocessing (grayscale/resize/frame-skip) and reward-shaping wrapper
  - `data/smw_data.json` — RAM address definitions for reward/done signals
- `training/`
  - `train.py` — builds env + `RecurrentPPO` model, runs training loop, checkpoints
  - `config.py` — hyperparameters (learning rate, gamma, GAE lambda, n_steps, batch size, etc.)
- `eval/`
  - `watch.py` — loads a checkpoint, runs it against the env with rendering on, for visual confirmation
  - `record.py` — same, but writes a video/gif instead of live rendering
- `scripts/`
  - `import_rom.sh` (or documented manual step) — wraps retro's ROM import so the user's legal ROM gets registered

## Data flow

1. `retro_env.step(action)` returns raw RGB frame + access to RAM.
2. Preprocessing wrapper turns the frame into the 84×84 grayscale observation.
3. Reward wrapper reads RAM, computes shaped reward from the deltas described below, and sets `done`/`truncated`.
4. Transitions accumulate in PPO's rollout buffer.
5. Every `n_steps`, the LSTM policy is updated (multiple epochs, minibatches, per standard PPO).
6. Checkpoints save periodically (model weights + optimizer state) with resume support.
7. `watch.py`/`record.py` load a checkpoint and replay it against a fresh env instance for qualitative evaluation.

## Reward function

Per step, after frame-skip:

- `+ k1 * Δx` — reward for net rightward progress since the last step (the primary learning signal)
- `+ level_clear_bonus` (large, e.g. 500) when the level-clear RAM flag flips — episode ends, success
- `- death_penalty` (e.g. 25) when the death/lives-lost RAM flag flips — episode ends, failure
- `+ k2 * Δscore_or_coins` — small-weighted secondary signal so it nudges but doesn't dominate progress
- `- timeout_penalty` if the episode is truncated by a step/time limit without clearing
- Per-step reward is clipped to a bounded range to keep PPO's advantage estimates stable

## Training loop

- `RecurrentPPO` (`sb3-contrib`), `CnnLstmPolicy`
- Single retro process initially; `SubprocVecEnv`-based parallelism is a documented future switch, not built now
- Frame-skip 4
- Hyperparameters starting from standard Atari-scale defaults: γ≈0.99, GAE-λ≈0.95, learning rate ≈2.5e-4 with linear schedule, clip range ≈0.2 — tuned empirically once the pipeline runs
- TensorBoard logging: episode reward, level-clear rate, average x-progress per episode
- Periodic checkpointing to disk with resume-from-checkpoint support

## Success criteria

1. **Pipeline correctness**: env resets/steps without crashing, observation shapes and reward signs are correct (verified by tests below), training runs stably for at least one full checkpoint interval.
2. **Learning signal present**: average episode x-progress rises above a random-policy baseline within a bounded number of training steps.
3. **Level clear**: agent reaches the level-clear flag at least once, then with a rising success rate as training continues.
4. *(Stretch, future sub-project, not this spec)*: generalize to a small curriculum of additional levels.

## Testing

- **Unit tests** for the reward wrapper: feed synthetic sequences of RAM-state dicts (no real emulator) and assert the computed reward and done/truncated flags match expectations for progress, death, level-clear, and timeout cases.
- **Smoke test**: instantiate the real retro env, reset, step with random actions for N steps, assert no crash and observation shape/dtype match spec.
- **Manual verification**: `watch.py` renders the emulator window so improvement across checkpoints can be visually confirmed — this is not automatable and is treated as a manual acceptance step, not a CI test.

## Open items for the user before first training run

- Legally-owned SMW ROM must be imported into stable-retro's game data directory via retro's `import` tool (you confirmed you have this ready).
