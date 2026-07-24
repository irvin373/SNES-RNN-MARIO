import os

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mario_rnn.env.factory import build_env
from mario_rnn.training import config

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"


def make_env():
    return Monitor(build_env())


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
        n_epochs=config.N_EPOCHS,
        ent_coef=config.ENT_COEF,
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
