"""
Generate cross-validation fixtures for a random POMDP using pymdp.

Run:
    uv run python generate_random_pomdp_fixtures.py

Output: random_pomdp_fixtures.json — a deterministically-seeded random
POMDP (num_obs=4, num_states=5, num_actions=3), a fixed observation
sequence, and per-step pymdp posteriors / EFE / actions.

Why a separate scenario from T-maze: random POMDPs exercise the
likelihood / transition tensors in their full generality (no zero
columns, no absorbing arms, dense correlations between factors). Bugs
that the structured T-maze masks should surface here.
"""

import json
import numpy as np

NUM_OBS = 4
NUM_STATES = 5
NUM_ACTIONS = 3
SEED = 0xCAFE


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def build_random_pomdp(rng):
    A = np.zeros((NUM_OBS, NUM_STATES))
    for s in range(NUM_STATES):
        A[:, s] = softmax(rng.standard_normal(NUM_OBS))

    B = np.zeros((NUM_STATES, NUM_STATES, NUM_ACTIONS))
    for a in range(NUM_ACTIONS):
        for s in range(NUM_STATES):
            B[:, s, a] = softmax(rng.standard_normal(NUM_STATES))

    C = rng.standard_normal(NUM_OBS)
    D = softmax(rng.standard_normal(NUM_STATES))

    return A, B, C, D


def main():
    try:
        from pymdp.legacy.agent import Agent as PyMDPAgent
        from pymdp.legacy.utils import obj_array
    except ImportError as e:
        raise SystemExit(
            "pymdp.legacy not available. Install: uv pip install -r requirements.txt"
        ) from e

    rng = np.random.default_rng(SEED)
    A_np, B_np, C_np, D_np = build_random_pomdp(rng)

    A = obj_array(1); A[0] = A_np
    B = obj_array(1); B[0] = B_np
    C = obj_array(1); C[0] = C_np
    D = obj_array(1); D[0] = D_np

    agent = PyMDPAgent(
        A=A, B=B, C=C, D=D,
        policy_len=2,
        gamma=8.0, alpha=8.0,
        action_selection="deterministic",
    )

    # Deterministic observation sequence (same seed for reproducibility)
    obs_rng = np.random.default_rng(SEED + 1)
    observations = [int(obs_rng.integers(0, NUM_OBS)) for _ in range(5)]

    posteriors_per_step = []
    actions_per_step = []
    efe_per_step = []
    for o in observations:
        qs = agent.infer_states([o])
        q_pi, G = agent.infer_policies()
        action = agent.sample_action()

        posteriors_per_step.append(np.asarray(qs[0]).flatten().tolist())
        actions_per_step.append(int(np.asarray(action).flatten()[0]))
        efe_per_step.append(np.asarray(G).flatten().tolist())

    fixtures = {
        "seed":          SEED,
        "num_obs":       NUM_OBS,
        "num_states":    NUM_STATES,
        "num_actions":   NUM_ACTIONS,
        "A":             A_np.tolist(),
        "B":             B_np.tolist(),
        "C":             C_np.tolist(),
        "D":             D_np.tolist(),
        "observations":  observations,
        "posteriors":    posteriors_per_step,
        "actions":       actions_per_step,
        "efe_per_step":  efe_per_step,
    }

    with open("random_pomdp_fixtures.json", "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"Wrote random_pomdp_fixtures.json ({len(observations)} steps).")
    print(f"  Observations: {observations}")
    print(f"  Actions: {actions_per_step}")


if __name__ == "__main__":
    main()
