# SMW RNN Player Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working end-to-end pipeline — SNES emulator → preprocessed observations → CNN-LSTM policy → RAM-based reward → PPO training → checkpoint → playback — proven on a single Super Mario World level (Yoshi's Island 1).

**Architecture:** `stable-retro` wraps the SNES core and exposes a Gymnasium env for `SuperMarioWorld-Snes-v0`. A stack of wrappers (action discretizer → grayscale/resize → frame-skip → reward-shaping) sits between the raw env and `sb3-contrib`'s `RecurrentPPO`, which trains a `CnnLstmPolicy`. Reward comes from RAM: the bundled integration already exposes `lives`/`coins`/`score` via `info`; Mario's x-position and (optionally) the level-clear flag are read directly off raw WRAM via `env.get_ram()` at addresses the engineer calibrates once, empirically, using a provided RAM-diff discovery script (these values aren't in the bundled integration and can only be confirmed by someone running the real ROM).

**Tech Stack:** Python, `stable-retro` (installed as `stable_retro`), Gymnasium, `stable-baselines3` + `sb3-contrib` (`RecurrentPPO`), PyTorch (CPU), OpenCV (`opencv-python-headless`), pytest.

## Global Constraints

- Game id: `SuperMarioWorld-Snes-v0` (confirmed via installed package; the older bare `SuperMarioWorld-Snes` id does not exist in this version).
- Import as `import stable_retro as retro` (the `retro` package name is a deprecated alias — do not use it in new code).
- Bundled `data.json` for this game only defines `lives` (i1 @ 0x7E0DBE), `coins` (u1 @ 0x7E0DBF), `score` (u4 LE @ 0x7E0F34) — these arrive automatically in the `info` dict from every `step()`/`reset()`. No custom retro integration files are needed; everything else is read via `env.get_ram()`.
- WRAM base address is `0x7E0000` (`rambase` in the SNES core config) — `get_ram()[addr - 0x7E0000]` gives the byte at absolute address `addr`.
- Bundled save states available for this game include `YoshiIsland1` through `YoshiIsland4`, `DonutPlains1-5`, `Forest1-5`, `VanillaDome1-5`, `ChocolateIsland1-3`, `Bridges1-2`, `Start`. This plan uses `YoshiIsland1`.
- `RetroEnv.step()` returns the 5-tuple `(obs, reward, terminated, truncated, info)`; `reset()` returns `(obs, info)` — standard Gymnasium API, not the old 4-tuple.
- Default action space is `MultiBinary(12)` over buttons `["B","Y","SELECT","START","UP","DOWN","LEFT","RIGHT","A","X","L","R"]`. This plan trims it to a small discrete set via the `Discretizer` pattern (adapted from `stable_retro.examples.discretizer`).
- A real, legally-obtained SMW ROM imported into stable-retro (via `python -m stable_retro.import <rom_dir>`) is required to run anything that touches the actual emulator (Tasks 2's discovery script, Task 6's smoke test, Task 7's training, Task 8's playback). Everything else (preprocessing/reward-wrapper unit tests) runs with no ROM.
- **Native macOS is broken for anything that touches the real emulator — always use Docker for those steps.** Confirmed by direct testing: on this machine (macOS/Apple Silicon), `stable-retro==1.0.1`'s compiled SNES core corrupts WRAM to a fixed garbage value after exactly the second `env.step()` call, reproducibly, regardless of Python version (3.12 and 3.14 both affected), sandbox status, action taken, save state, or action-space config. The identical code runs correctly under Linux (verified via a `linux/amd64` container, which runs fine under Docker Desktop's emulation even on Apple Silicon — no native aarch64 wheel exists for this package, so `--platform linux/amd64` is required either way). A `Dockerfile` (installs `requirements.txt` into `python:3.12-slim`) and `scripts/docker-run.sh <command...>` (builds the image if needed, mounts the repo at `/workspace` and the ROM directory at `/roms`, imports the ROM, then runs the given command) already exist at the repo root — use `scripts/docker-run.sh <command>` in place of `./.venv/bin/<command>` for every step in Tasks 2, 3 (manual check), 6, 7, and 8. Tasks 1, 4, and 5 never touch the real emulator and continue to use the local `.venv` directly. The ROM lives at `~/roms/smw/` on this machine (outside the repo, per `.gitignore`); `scripts/docker-run.sh` defaults to that path via `$MARIO_RNN_ROM_DIR`.

---

## File Structure

```
mario_rnn/
  __init__.py
  env/
    __init__.py
    discretizer.py     # SMWDiscretizer action wrapper
    preprocessing.py   # PreprocessFrame, FrameSkip observation/reward-accumulating wrappers
    reward.py           # SMWRewardWrapper (RAM-based reward shaping)
    factory.py          # build_env() — wires the full wrapper stack
  training/
    __init__.py
    config.py           # hyperparameters + calibrated RAM address constants
    train.py             # RecurrentPPO training entrypoint
  eval/
    __init__.py
    play.py              # load a checkpoint, render or record a rollout
scripts/
  find_ram_addresses.py # RAM-diff discovery tool for x-position / level-clear address
tests/
  test_preprocessing.py
  test_reward.py
  test_env_smoke.py     # requires real ROM — see Task 6
requirements.txt
README.md
Dockerfile              # Linux dev image (python:3.12-slim + requirements.txt) — see Global Constraints
scripts/
  docker-run.sh          # runs a command inside the Linux container, ROM mounted + imported
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `mario_rnn/__init__.py`
- Create: `mario_rnn/env/__init__.py`
- Create: `mario_rnn/training/__init__.py`
- Create: `mario_rnn/eval/__init__.py`
- Create: `README.md`

**Interfaces:**
- Produces: an importable `mario_rnn` package (empty `__init__.py` files) and an installed environment later tasks build on.

- [ ] **Step 1: Write requirements.txt**

```
stable-retro>=0.9.6
sb3-contrib>=2.3.0
stable-baselines3>=2.3.0
torch>=2.2.0
gymnasium>=0.29.0
opencv-python-headless>=4.9.0
numpy>=1.26.0
imageio>=2.34.0
pytest>=8.0.0
```

- [ ] **Step 2: Create a virtualenv and install**

Run:
```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```
Expected: installs with no errors. `stable-retro` installs under the importable name `stable_retro`.

- [ ] **Step 3: Verify import**

Run: `./.venv/bin/python -c "import stable_retro as retro; print(retro.__file__)"`
Expected: prints a path ending in `site-packages/stable_retro/__init__.py`, no traceback.

- [ ] **Step 4: Create package skeleton**

```bash
mkdir -p mario_rnn/env mario_rnn/training mario_rnn/eval scripts tests
touch mario_rnn/__init__.py mario_rnn/env/__init__.py mario_rnn/training/__init__.py mario_rnn/eval/__init__.py
```

- [ ] **Step 5: Write README.md**

```markdown
# Mario RNN

RNN (CNN+LSTM) agent trained via PPO to play Super Mario World, self-play/RL only — no hand-coded rules.

## Setup

1. `python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt`
2. Import your own legally-owned SMW ROM:
   `./.venv/bin/python -m stable_retro.import /path/to/your/rom/dir`
   This registers the ROM against the bundled `SuperMarioWorld-Snes-v0` integration.
3. Calibrate RAM addresses (one-time, requires the ROM):
   `./.venv/bin/python scripts/find_ram_addresses.py`
   Follow its instructions, then fill the printed addresses into `mario_rnn/training/config.py`.
4. Train: `./.venv/bin/python -m mario_rnn.training.train`
5. Watch a checkpoint play: `./.venv/bin/python -m mario_rnn.eval.play --checkpoint <path> --render`
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt README.md mario_rnn scripts tests
git commit -m "chore: scaffold mario_rnn package structure"
```

---

### Task 2: RAM address discovery + calibration script

**Files:**
- Create: `scripts/find_ram_addresses.py`

**Interfaces:**
- Produces: a runnable CLI script the engineer uses once (with the real ROM) to find Mario's x-position address and confirm the RAM-indexing formula. Later tasks (`reward.py`, `config.py`) consume the addresses it prints.
- Consumes: `stable_retro` (`retro.make`, `env.get_ram()`, `env.reset()`, `env.step()`) — no other project code.

- [ ] **Step 1: Write the discovery script**

```python
"""
One-time RAM calibration tool for the SMW reward wrapper.

Usage:
    python scripts/find_ram_addresses.py

Requires a real SMW ROM already imported into stable-retro.
"""
import numpy as np
import stable_retro as retro

RAM_BASE = 0x7E0000
GAME = "SuperMarioWorld-Snes-v0"
STATE = "YoshiIsland1"

# Button index for holding right+B (run) — see Task 3 for the full button list.
BUTTONS = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]


def make_action(*names):
    arr = np.zeros(len(BUTTONS), dtype=np.uint8)
    for name in names:
        arr[BUTTONS.index(name)] = 1
    return arr


def sanity_check_known_addresses(env, info):
    """Confirm get_ram() indexing lines up with the info dict's known variables."""
    ram = env.get_ram()
    lives_byte = int(np.int8(ram[0x7E0DBE - RAM_BASE]))
    coins_byte = int(ram[0x7E0DBF - RAM_BASE])
    assert lives_byte == info["lives"], f"lives mismatch: ram={lives_byte} info={info['lives']}"
    assert coins_byte == info["coins"], f"coins mismatch: ram={coins_byte} info={info['coins']}"
    print("RAM indexing formula confirmed: get_ram()[addr - 0x7E0000] matches info dict.")


def find_monotonic_candidates(snapshots):
    """Given a list of full-RAM snapshots taken while holding RIGHT, find addresses
    whose value strictly increased at every step — candidates for x-position."""
    diffs = [snapshots[i + 1].astype(int) - snapshots[i].astype(int) for i in range(len(snapshots) - 1)]
    always_positive = np.all([d > 0 for d in diffs], axis=0)
    return np.where(always_positive)[0]


def main():
    # render_mode="rgb_array" avoids popping an OS window during a headless discovery run.
    env = retro.make(game=GAME, state=STATE, render_mode="rgb_array")
    env.reset()
    # reset() always returns an empty info dict in this stable-retro version — info is
    # only populated by step(), so take one no-op step first to get a real info dict.
    noop_action = make_action()
    obs, reward, terminated, truncated, info = env.step(noop_action)
    sanity_check_known_addresses(env, info)

    right_action = make_action("RIGHT", "B")
    snapshots = [env.get_ram().copy()]
    for _ in range(5):
        for _ in range(15):
            env.step(right_action)
        snapshots.append(env.get_ram().copy())

    candidates = find_monotonic_candidates(snapshots)
    print(f"Found {len(candidates)} address(es) that increased every checkpoint while holding RIGHT.")
    print("Candidate absolute addresses (hex):")
    for idx in candidates:
        print(f"  0x{idx + RAM_BASE:06X}  (value now: {snapshots[-1][idx]})")
    print()
    print("Pick the candidate whose value range makes sense for a screen-width position")
    print("(should grow steadily into the hundreds/thousands over ~75 frames of running right).")
    print("A 16-bit x-position is usually two adjacent addresses (low byte, high byte) —")
    print("look for a pair `addr` and `addr+1` both in the candidate list.")
    print()
    print("Fill the chosen address into X_POSITION_ADDRESS in mario_rnn/training/config.py.")

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it inside the Linux container (see Global Constraints — native macOS corrupts emulator RAM)**

Run: `scripts/docker-run.sh python scripts/find_ram_addresses.py`
Expected: prints "RAM indexing formula confirmed", then a short list of candidate addresses. Several candidates are expected (position-like counters, camera scroll, etc. can all look monotonic over a short window) — note them all down; picking the specific x-position address happens in Task 3 alongside filling in `config.py`.

- [ ] **Step 3: Commit**

```bash
git add scripts/find_ram_addresses.py
git commit -m "feat: add RAM address discovery script for x-position calibration"
```

---

### Task 3: Action discretizer + env factory

**Files:**
- Create: `mario_rnn/env/discretizer.py`
- Create: `mario_rnn/env/factory.py`
- Create: `mario_rnn/training/config.py`

**Interfaces:**
- Produces: `SMWDiscretizer(env)` (a `gym.ActionWrapper`), `make_raw_env(state="YoshiIsland1") -> gym.Env` (retro env + discretizer only, no preprocessing/reward yet — later tasks layer on top of this).
- Consumes: `stable_retro` game id and button list from Global Constraints.

- [ ] **Step 1: Write the discretizer**

```python
import gymnasium as gym
import numpy as np


class SMWDiscretizer(gym.ActionWrapper):
    """Restrict SMW's 12-button MultiBinary action space to a small, useful discrete set."""

    COMBOS = [
        [],
        ["LEFT"],
        ["RIGHT"],
        ["RIGHT", "B"],
        ["RIGHT", "A"],
        ["RIGHT", "B", "A"],
        ["A"],
        ["B"],
        ["DOWN"],
    ]

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decoded = []
        for combo in self.COMBOS:
            arr = np.zeros(len(buttons), dtype=np.uint8)
            for button in combo:
                arr[buttons.index(button)] = 1
            self._decoded.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._decoded))

    def action(self, act):
        return self._decoded[act].copy()
```

- [ ] **Step 2: Write config.py with calibration placeholders**

```python
"""Hyperparameters and RAM addresses calibrated via scripts/find_ram_addresses.py."""

GAME = "SuperMarioWorld-Snes-v0"
STATE = "YoshiIsland1"

# --- Calibrated constants: found via scripts/find_ram_addresses.py, then confirmed by a
# noop-only control run (see plan commit history) — 0x7E0013 was a false positive (a
# free-running frame counter that increases even with no input held); 0x7E007E stayed
# flat under noop and tracked real rightward movement, including plateauing when Mario
# hit an obstacle, so it's the real position address (mirrored identically at 0x7E00D1,
# 0x7E0310, 0x7E0314 — any of the four would work equally well).
X_POSITION_ADDRESS = 0x7E007E  # low byte; high byte assumed at ADDRESS + 1 (see reward.py)

# Address of a byte that goes non-zero when the level-clear sequence starts.
# Optional: leave as None to train on progress/death signal only (level-clear bonus
# is then never awarded, but training still works — see reward.py).
LEVEL_CLEAR_ADDRESS = None

# --- Preprocessing ---
FRAME_SIZE = 84
FRAME_SKIP = 4

# --- Reward shaping ---
PROGRESS_WEIGHT = 1.0
SCORE_WEIGHT = 0.01
DEATH_PENALTY = 25.0
LEVEL_CLEAR_BONUS = 500.0
TIMEOUT_PENALTY = 10.0
REWARD_CLIP = 15.0

# --- PPO ---
TOTAL_TIMESTEPS = 2_000_000
N_STEPS = 256
BATCH_SIZE = 64
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
CHECKPOINT_FREQ = 50_000
MAX_EPISODE_STEPS = 6_000
```

- [ ] **Step 3: Write the raw env factory**

```python
import stable_retro as retro

from mario_rnn.env.discretizer import SMWDiscretizer
from mario_rnn.training import config


def make_raw_env(state=config.STATE, render_mode="rgb_array"):
    # render_mode="rgb_array" is the default so headless runs (tests, training) never pop
    # an OS window; eval/play.py (Task 8) overrides this to "human" when --render is passed.
    env = retro.make(game=config.GAME, state=state, render_mode=render_mode)
    env = SMWDiscretizer(env)
    return env
```

- [ ] **Step 4: Manual check (requires real ROM — run inside the Linux container, see Global Constraints)**

Run:
```bash
scripts/docker-run.sh python -c "
from mario_rnn.env.factory import make_raw_env
env = make_raw_env()
print('action_space', env.action_space)
obs, info = env.reset()
print('obs shape', obs.shape, 'reset info', info)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print('post-step info', info)
env.close()
"
```
Expected: `action_space Discrete(9)`, obs shape like `(224, 256, 3)`. `reset info` is `{}` — this stable-retro version only populates `info` from `step()`, never from `reset()`. `post-step info` contains `lives`, `coins`, `score`.

- [ ] **Step 5: Commit**

```bash
git add mario_rnn/env/discretizer.py mario_rnn/env/factory.py mario_rnn/training/config.py
git commit -m "feat: add action discretizer, raw env factory, and training config"
```

---

### Task 4: Preprocessing wrappers (grayscale/resize/frame-skip)

**Files:**
- Create: `mario_rnn/env/preprocessing.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Produces: `PreprocessFrame(env, size=84)` (`gym.ObservationWrapper`, outputs `Box(0, 255, (size, size, 1), uint8)`), `FrameSkip(env, skip=4)` (`gym.Wrapper`, repeats the action `skip` times, sums reward, returns the final frame).
- Consumes: nothing project-specific — works on any `gym.Env` with an RGB image observation space, so tests use a synthetic dummy env (no ROM needed).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import gymnasium as gym

from mario_rnn.env.preprocessing import PreprocessFrame, FrameSkip


class DummyImageEnv(gym.Env):
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Box(0, 255, (224, 256, 3), dtype=np.uint8)

    def __init__(self):
        self._step_rewards = None
        self._i = 0

    def reset(self, seed=None, options=None):
        self._i = 0
        frame = np.full((224, 256, 3), 10, dtype=np.uint8)
        return frame, {}

    def step(self, action):
        self._i += 1
        frame = np.full((224, 256, 3), 10 + self._i, dtype=np.uint8)
        terminated = self._i >= 100
        return frame, 1.0, terminated, False, {}


def test_preprocess_frame_shape_and_dtype():
    env = PreprocessFrame(DummyImageEnv(), size=84)
    obs, info = env.reset()
    assert obs.shape == (84, 84, 1)
    assert obs.dtype == np.uint8
    assert env.observation_space.shape == (84, 84, 1)


def test_frame_skip_sums_reward_and_repeats_action():
    env = FrameSkip(DummyImageEnv(), skip=4)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 4.0  # 1.0 reward per underlying step, repeated 4 times


def test_frame_skip_stops_early_on_termination():
    env = FrameSkip(DummyImageEnv(), skip=4)
    env.reset()
    terminated = False
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(0)
        if terminated:
            break
    assert terminated is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/pytest tests/test_preprocessing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mario_rnn.env.preprocessing'`

- [ ] **Step 3: Write the implementation**

```python
import cv2
import numpy as np
import gymnasium as gym


class PreprocessFrame(gym.ObservationWrapper):
    """Grayscale + resize an RGB frame to (size, size, 1) uint8."""

    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(size, size, 1), dtype=np.uint8
        )

    def observation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return resized[:, :, None]


class FrameSkip(gym.Wrapper):
    """Repeat the given action `skip` times, summing reward, stopping early on episode end."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        obs, info, terminated, truncated = None, {}, False, False
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/pytest tests/test_preprocessing.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add mario_rnn/env/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add frame preprocessing and frame-skip wrappers"
```

---

### Task 5: Reward wrapper (RAM-based shaping)

**Files:**
- Create: `mario_rnn/env/reward.py`
- Test: `tests/test_reward.py`

**Interfaces:**
- Produces: `SMWRewardWrapper(env, x_address, level_clear_address=None, ram_base=0x7E0000, progress_weight=1.0, score_weight=0.01, death_penalty=25.0, level_clear_bonus=500.0, timeout_penalty=10.0, reward_clip=15.0)` — a `gym.Wrapper` that overrides `step()`'s reward and `terminated` using RAM/`info`.
- Consumes: `env.unwrapped.get_ram()` and the `info` dict's `lives`/`score` keys (both already provided by the bundled SMW integration — see Global Constraints). Uses a fake env in tests, so no ROM is required.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import gymnasium as gym

from mario_rnn.env.reward import SMWRewardWrapper

X_ADDR = 10  # arbitrary small address for tests; ram_base=0 so this is the raw index
LEVEL_CLEAR_ADDR = 20


class FakeSMWEnv(gym.Env):
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Box(0, 255, (4, 4, 1), dtype=np.uint8)

    def __init__(self):
        self.ram = np.zeros(64, dtype=np.uint8)
        self.info = {"lives": 4, "coins": 0, "score": 0}
        self.script = []
        self._i = 0

    def set_script(self, script):
        """script: list of (ram_updates: dict[addr, val], info_updates: dict, truncated: bool)"""
        self.script = script
        self._i = 0

    def get_ram(self):
        return self.ram

    def reset(self, seed=None, options=None):
        self._i = 0
        return np.zeros((4, 4, 1), dtype=np.uint8), dict(self.info)

    def step(self, action):
        ram_updates, info_updates, truncated = self.script[self._i]
        self._i += 1
        for addr, val in ram_updates.items():
            self.ram[addr] = val
        self.info.update(info_updates)
        obs = np.zeros((4, 4, 1), dtype=np.uint8)
        terminated = False
        return obs, 0.0, terminated, truncated, dict(self.info)


def make_wrapper(**overrides):
    env = FakeSMWEnv()
    kwargs = dict(
        x_address=X_ADDR,
        level_clear_address=LEVEL_CLEAR_ADDR,
        ram_base=0,
        progress_weight=1.0,
        score_weight=0.01,
        death_penalty=25.0,
        level_clear_bonus=500.0,
        timeout_penalty=10.0,
        reward_clip=1000.0,  # high in most tests so we can see the unclipped value
    )
    kwargs.update(overrides)
    return SMWRewardWrapper(env, **kwargs), env


def test_progress_reward():
    wrapper, env = make_wrapper()
    wrapper.reset()  # x starts at 0
    env.set_script([({X_ADDR: 50}, {}, False)])
    obs, reward, terminated, truncated, info = wrapper.step(0)
    assert reward == 50.0
    assert terminated is False


def test_death_penalty_and_terminates():
    wrapper, env = make_wrapper()
    wrapper.reset()
    env.set_script([({}, {"lives": -1}, False)])
    obs, reward, terminated, truncated, info = wrapper.step(0)
    assert reward == -25.0
    assert terminated is True


def test_level_clear_bonus_and_terminates():
    wrapper, env = make_wrapper()
    wrapper.reset()
    env.set_script([({LEVEL_CLEAR_ADDR: 1}, {}, False)])
    obs, reward, terminated, truncated, info = wrapper.step(0)
    assert reward == 500.0
    assert terminated is True


def test_timeout_penalty_on_truncation():
    wrapper, env = make_wrapper()
    wrapper.reset()
    env.set_script([({}, {}, True)])
    obs, reward, terminated, truncated, info = wrapper.step(0)
    assert reward == -10.0
    assert truncated is True


def test_reward_is_clipped():
    wrapper, env = make_wrapper(reward_clip=15.0)
    wrapper.reset()
    env.set_script([({X_ADDR: 200}, {}, False)])
    obs, reward, terminated, truncated, info = wrapper.step(0)
    assert reward == 15.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/pytest tests/test_reward.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mario_rnn.env.reward'`

- [ ] **Step 3: Write the implementation**

```python
import numpy as np
import gymnasium as gym


class SMWRewardWrapper(gym.Wrapper):
    """RAM-based reward shaping: rightward progress, level-clear bonus, death penalty,
    small score nudge, timeout penalty. Terminates the episode on death or level clear."""

    def __init__(
        self,
        env,
        x_address,
        level_clear_address=None,
        ram_base=0x7E0000,
        progress_weight=1.0,
        score_weight=0.01,
        death_penalty=25.0,
        level_clear_bonus=500.0,
        timeout_penalty=10.0,
        reward_clip=15.0,
    ):
        super().__init__(env)
        self.x_address = x_address
        self.level_clear_address = level_clear_address
        self.ram_base = ram_base
        self.progress_weight = progress_weight
        self.score_weight = score_weight
        self.death_penalty = death_penalty
        self.level_clear_bonus = level_clear_bonus
        self.timeout_penalty = timeout_penalty
        self.reward_clip = reward_clip
        self._prev_x = None
        self._prev_score = 0

    def _read_x(self):
        ram = self.env.unwrapped.get_ram()
        idx = self.x_address - self.ram_base
        lo = int(ram[idx])
        hi = int(ram[idx + 1])
        return lo + hi * 256

    def _is_level_clear(self):
        if self.level_clear_address is None:
            return False
        ram = self.env.unwrapped.get_ram()
        return bool(ram[self.level_clear_address - self.ram_base] != 0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_x = self._read_x()
        self._prev_score = info.get("score", 0)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        x = self._read_x()
        score = info.get("score", 0)

        reward = self.progress_weight * (x - self._prev_x)
        reward += self.score_weight * (score - self._prev_score)

        if self._is_level_clear():
            reward += self.level_clear_bonus
            terminated = True
        elif info.get("lives", 0) < 0:
            reward -= self.death_penalty
            terminated = True
        elif truncated:
            reward -= self.timeout_penalty

        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        self._prev_x = x
        self._prev_score = score
        return obs, reward, terminated, truncated, info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/pytest tests/test_reward.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add mario_rnn/env/reward.py tests/test_reward.py
git commit -m "feat: add RAM-based reward shaping wrapper"
```

---

### Task 6: Full env integration + smoke test

**Files:**
- Modify: `mario_rnn/env/factory.py`
- Test: `tests/test_env_smoke.py`

**Interfaces:**
- Produces: `build_env(state=config.STATE) -> gym.Env` — the complete wrapper stack (raw env → discretizer → preprocessing → frame-skip → reward shaping → `TimeLimit`), used by both training and eval.
- Consumes: `make_raw_env` (Task 3), `PreprocessFrame`/`FrameSkip` (Task 4), `SMWRewardWrapper` (Task 5), `config` constants (Task 3) — specifically requires `config.X_POSITION_ADDRESS` to be filled in via Task 2's script.

- [ ] **Step 1: Extend factory.py with build_env**

```python
import gymnasium as gym
import stable_retro as retro

from mario_rnn.env.discretizer import SMWDiscretizer
from mario_rnn.env.preprocessing import FrameSkip, PreprocessFrame
from mario_rnn.env.reward import SMWRewardWrapper
from mario_rnn.training import config


def make_raw_env(state=config.STATE, render_mode="rgb_array"):
    env = retro.make(game=config.GAME, state=state, render_mode=render_mode)
    env = SMWDiscretizer(env)
    return env


def build_env(state=config.STATE):
    assert config.X_POSITION_ADDRESS is not None, (
        "config.X_POSITION_ADDRESS is not set — run scripts/find_ram_addresses.py "
        "and fill in the discovered address first."
    )
    env = make_raw_env(state=state)
    env = PreprocessFrame(env, size=config.FRAME_SIZE)
    env = FrameSkip(env, skip=config.FRAME_SKIP)
    env = SMWRewardWrapper(
        env,
        x_address=config.X_POSITION_ADDRESS,
        level_clear_address=config.LEVEL_CLEAR_ADDRESS,
        progress_weight=config.PROGRESS_WEIGHT,
        score_weight=config.SCORE_WEIGHT,
        death_penalty=config.DEATH_PENALTY,
        level_clear_bonus=config.LEVEL_CLEAR_BONUS,
        timeout_penalty=config.TIMEOUT_PENALTY,
        reward_clip=config.REWARD_CLIP,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.MAX_EPISODE_STEPS)
    return env
```

- [ ] **Step 2: Write the smoke test (requires real ROM + calibrated config.X_POSITION_ADDRESS)**

```python
import numpy as np

from mario_rnn.env.factory import build_env


def test_env_resets_and_steps_without_crashing():
    env = build_env()
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.uint8

    rng = np.random.default_rng(0)
    for _ in range(50):
        action = rng.integers(env.action_space.n)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
```

- [ ] **Step 3: Run it inside the Linux container (only passes with ROM imported and config.X_POSITION_ADDRESS set — see Global Constraints)**

Run: `scripts/docker-run.sh pytest tests/test_env_smoke.py -v`
Expected: 1 passed. If it fails with the `X_POSITION_ADDRESS is not set` assertion, go back to Task 2's script output and fill in `mario_rnn/training/config.py`.

- [ ] **Step 4: Commit**

```bash
git add mario_rnn/env/factory.py tests/test_env_smoke.py
git commit -m "feat: wire full env wrapper stack and add smoke test"
```

---

### Task 7: PPO+LSTM training entrypoint

**Files:**
- Create: `mario_rnn/training/train.py`

**Interfaces:**
- Produces: a runnable module `python -m mario_rnn.training.train` that trains `RecurrentPPO` on `build_env()` and writes checkpoints + TensorBoard logs.
- Consumes: `build_env` (Task 6), all hyperparameters from `config.py` (Task 3).

- [ ] **Step 1: Write train.py**

```python
import os

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from mario_rnn.env.factory import build_env
from mario_rnn.training import config

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"


def make_env():
    return build_env()


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = DummyVecEnv([make_env])

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config.CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="smw_ppo_lstm",
    )

    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=checkpoint_callback)
    model.save(os.path.join(CHECKPOINT_DIR, "smw_ppo_lstm_final"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run training for a tiny number of steps (requires real ROM + calibrated config — run inside the Linux container, see Global Constraints)**

Temporarily run with a small override to confirm the wiring works end-to-end before committing to a multi-million-step run:

Run:
```bash
scripts/docker-run.sh python -c "
from mario_rnn.training import config
config.TOTAL_TIMESTEPS = 200
config.N_STEPS = 32
config.BATCH_SIZE = 16
config.CHECKPOINT_FREQ = 100
from mario_rnn.training.train import main
main()
"
```
Expected: runs to completion with no exceptions, prints SB3's training log table at least once, and leaves files under `checkpoints/`.

- [ ] **Step 3: Commit**

```bash
git add mario_rnn/training/train.py
git commit -m "feat: add RecurrentPPO training entrypoint"
```

---

### Task 8: Checkpoint playback / evaluation script

**Files:**
- Create: `mario_rnn/eval/play.py`

**Interfaces:**
- Produces: `python -m mario_rnn.eval.play --checkpoint <path> [--render] [--record out.gif] [--episodes N]` — loads a saved `RecurrentPPO` model and runs it against `build_env()`, printing per-episode reward/x-progress, optionally rendering live and/or writing a gif.
- Consumes: `build_env` (Task 6), `RecurrentPPO.load` (sb3-contrib), `imageio` for gif export.

- [ ] **Step 1: Write play.py**

```python
import argparse

import imageio
import numpy as np
from sb3_contrib import RecurrentPPO

from mario_rnn.env.factory import build_env


def run_episode(model, env, render, collect_frames):
    obs, info = env.reset()
    lstm_states = None
    episode_start = np.array([True])
    total_reward = 0.0
    terminated = truncated = False
    frames = []

    if collect_frames:
        frames.append(env.unwrapped.get_screen().copy())

    while not (terminated or truncated):
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_start, deterministic=True
        )
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_start = np.array([False])
        if render:
            env.render()
        if collect_frames:
            frames.append(env.unwrapped.get_screen().copy())

    return total_reward, info, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record", help="path to write a .gif of the first episode")
    args = parser.parse_args()

    env = build_env()
    if args.render:
        env.unwrapped.render_mode = "human"
    model = RecurrentPPO.load(args.checkpoint)

    for ep in range(args.episodes):
        collect_frames = args.record is not None and ep == 0
        total_reward, info, frames = run_episode(model, env, args.render, collect_frames)
        print(f"episode {ep}: reward={total_reward:.1f} info={info}")
        if collect_frames:
            imageio.mimsave(args.record, frames, fps=30)
            print(f"wrote {len(frames)} frames to {args.record}")

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run against a checkpoint from Task 7's smoke run (inside the Linux container — see Global Constraints)**

Run: `scripts/docker-run.sh python -m mario_rnn.eval.play --checkpoint checkpoints/smw_ppo_lstm_final --episodes 1`
Expected: prints one `episode 0: reward=... info={...}` line with no exceptions. (`--render` is not meaningful inside the headless container — use `--record` to inspect a rollout visually instead.)

- [ ] **Step 3: Run with --record to confirm gif export**

Run: `scripts/docker-run.sh python -m mario_rnn.eval.play --checkpoint checkpoints/smw_ppo_lstm_final --episodes 1 --record /tmp/smw_rollout.gif`
Expected: prints the episode line, then `wrote N frames to /tmp/smw_rollout.gif`, and the file exists (`/tmp` inside the container isn't on the host — pass a path under `/workspace`, e.g. `/workspace/smw_rollout.gif`, so it lands back in the repo directory on the host).

- [ ] **Step 4: Commit**

```bash
git add mario_rnn/eval/play.py
git commit -m "feat: add checkpoint playback/evaluation script with gif export"
```

---

## Notes for the executing engineer

- Tasks 2, 3 (manual check), 6, 7, and 8 require the real ROM imported and, for Tasks 6-8, `config.X_POSITION_ADDRESS` filled in from Task 2's output. Tasks 1, 4, and 5 have no such dependency and their tests should pass on any machine.
- If Task 2's discovery script finds zero or an implausible number of monotonic candidates, widen the hold-right window (increase the inner loop from 15 to ~30 frames) — Mario may be blocked by scenery for the first few frames of `YoshiIsland1`.
- `LEVEL_CLEAR_ADDRESS` is optional. Training and playback both work with it left as `None` (the level-clear bonus term is simply never triggered, and episodes end only via death or the `TimeLimit` truncation) — treat finding that address as a nice-to-have follow-up, not a blocker for the rest of this plan.
- Confirmed on this machine (macOS, Apple Silicon): `retro.make(..., render_mode="human")` followed later by `env.close()` raises `AttributeError: 'CocoaAlternateEventLoop' object has no attribute 'platform_event_loop'` from pyglet's window teardown — a pyglet/macOS bug, not something in this project's code. It fires during `close()`, after rendering has already happened, so it doesn't block Task 8's `--render` path from working; if it comes up during review, it's a known upstream issue, not a defect to fix here.
