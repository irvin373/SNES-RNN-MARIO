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
