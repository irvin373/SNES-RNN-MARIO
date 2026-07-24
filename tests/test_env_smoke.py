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
