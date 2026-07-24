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
