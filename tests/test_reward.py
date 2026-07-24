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
    env.set_script([({}, {"lives": 3}, False)])
    obs, reward, terminated, truncated, info = wrapper.step(0)
    assert reward == -25.0
    assert terminated is True


def test_no_death_penalty_when_lives_unchanged():
    wrapper, env = make_wrapper()
    wrapper.reset()
    env.set_script([({}, {"lives": 4}, False)])
    obs, reward, terminated, truncated, info = wrapper.step(0)
    assert reward == 0.0
    assert terminated is False


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
