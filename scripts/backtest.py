"""The exam: the trained agent plays 2023-2024 once, against the always-basket-0 baseline.
Run: uv run python -m scripts.backtest"""
import json
from pathlib import Path

from havenrl import metrics
from havenrl.agent import DQNAgent
from havenrl.data import TEST_END, TEST_START, prepare_data
from havenrl.env import BASKETS, HavenEnv, play

MODEL_PATH = Path("artifacts/model.pth")
RESULTS_PATH = Path("artifacts/results.json")
BASELINE_BASKET = 0  # always mostly Gold: the score to beat


def count_switches(actions):
    """Days the basket changed. (The old notebook called every day a 'trade'.)"""
    return sum(1 for a, b in zip(actions, actions[1:]) if a != b)


def main():
    env = HavenEnv(*prepare_data()["test"])

    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    agent.load(MODEL_PATH)

    dates, agent_values, agent_actions = play(env, lambda obs: agent.act(obs, greedy=True))
    _, baseline_values, _ = play(env, lambda obs: BASELINE_BASKET)

    results = {
        "test_period": {"start": TEST_START, "end": TEST_END},
        "baskets": BASKETS.tolist(),
        "agent": {**metrics.summary(agent_values), "basket_switches": count_switches(agent_actions)},
        "baseline": {**metrics.summary(baseline_values), "basket": BASELINE_BASKET},
        "daily": [
            {"date": d, "agent_value": round(a, 2), "baseline_value": round(b, 2)}
            for d, a, b in zip(dates, agent_values, baseline_values)
        ],
        "agent_actions": agent_actions,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    for name in ("agent", "baseline"):
        r = results[name]
        print(f"{name:8s}  Rs {r['final_value']:>11,.0f}  return {r['total_return_pct']:6.2f}%  "
              f"drawdown {r['max_drawdown_pct']:5.2f}%  Sharpe {r['sharpe_ratio']:5.2f}")
    winner = "Agent" if results["agent"]["final_value"] > results["baseline"]["final_value"] else "Baseline"
    print(f"{winner} wins. Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
