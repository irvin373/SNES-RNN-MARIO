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
