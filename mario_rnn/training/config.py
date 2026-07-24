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
FPS = 60

# --- Anti-freeze watchdog ---
# If mario makes no rightward progress for this many seconds, force a fixed
# action (running jump) every step until he moves again or the episode ends.
STUCK_PATIENCE_SECONDS = 1.0
STUCK_FORCED_ACTION = 3  # index into SMWDiscretizer.COMBOS: RIGHT + B + A

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
N_EPOCHS = 4  # SB3 default is 10; lowered to cut per-iteration compute time
# Entropy bonus: without this, ent_coef defaults to 0.0 and the policy can collapse to a
# single near-deterministic action within the first few hundred thousand steps and never
# explore again (observed directly: train/entropy_loss crashed to ~0 by step 400k in an
# earlier 2M-step run, with clip_fraction and value_loss also flatlining at 0 — a policy
# that stopped changing). 0.01 is the standard PPO/Atari-scale default.
ENT_COEF = 0.01
CHECKPOINT_FREQ = 50_000
MAX_EPISODE_STEPS = 6_000
