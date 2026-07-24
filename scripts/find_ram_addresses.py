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
