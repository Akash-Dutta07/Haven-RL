"""Practice: the agent plays 2020-2021 many times and learns. After every episode it sits a
practice exam on 2022, and only the brain with the best practice-exam score is saved.
Run: uv run python -m scripts.train"""
import random
from pathlib import Path

import numpy as np
import torch

from havenrl.agent import DQNAgent
from havenrl.data import prepare_data
from havenrl.env import HavenEnv, play

EPISODES = 100
SEED = 42
MODEL_PATH = Path("artifacts/model.pth")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data = prepare_data()
    env = HavenEnv(*data["train"])
    val_env = HavenEnv(*data["val"])
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    _, baseline_values, _ = play(val_env, lambda obs: 0)
    print(f"Practice-exam (2022) baseline, always basket 0: Rs {baseline_values[-1]:,.0f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    best_val, best_episode = -np.inf, None
    for episode in range(1, EPISODES + 1):
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, done)
            agent.learn()
            obs = next_obs

        agent.decay_epsilon()  # explore a bit less next episode
        agent.update_target()  # refresh the frozen copy

        _, val_values, _ = play(val_env, lambda obs: agent.act(obs, greedy=True))
        mark = ""
        if val_values[-1] > best_val:
            best_val, best_episode = val_values[-1], episode
            agent.save(MODEL_PATH)
            mark = "  <- best so far, saved"
        print(f"Episode {episode:3d}/{EPISODES}  train Rs {info['value']:,.0f}  "
              f"practice exam Rs {val_values[-1]:,.0f}  epsilon {agent.epsilon:.3f}{mark}")

    print(f"Best brain: episode {best_episode} (practice exam Rs {best_val:,.0f}), saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
