"""
Generate cross-validation fixtures for the canonical T-maze using pymdp.

Run:
    uv run python generate_tmaze_fixtures.py

Output: tmaze_fixtures.json — A, B, C, D, observation sequence, per-step
posterior, EFE per policy, action distribution.

The Julia test at test/golden/test_pymdp_tmaze.jl reads this file and
asserts numerical agreement with Aifc.jl outputs.

API
---
We use `pymdp.legacy` (the numpy-based API), since it's API-stable and
maps cleanly onto the obj_array per-modality / per-factor convention.
The newer JAX-based API in `pymdp.agent` is more powerful but its
broadcasting / batching conventions are still in flux.

Conventions
-----------
The 8-state single-factor T-maze (matching Aifc's
`tmaze_model`). Single-factor removes any factorization-convention
ambiguity for cross-validation; Aifc's multi-factor
representation is independently cross-validated against single-factor
in `test/unit/integration/test_tmaze_multi_factor.jl`.
"""

import json
import numpy as np

# ---- Build the T-maze A/B/C/D in 8-state single-factor form ---------

CUE_RELIABILITY = 0.95
PREFERENCE = 4.0

NUM_LOC, NUM_CTX = 4, 2
NUM_STATES = NUM_LOC * NUM_CTX  # = 8
NUM_OBS = 5
NUM_ACTIONS = 4


def state_idx(loc, ctx):
    return loc * 2 + ctx


def build_A():
    A = np.zeros((NUM_OBS, NUM_STATES))
    for loc in range(NUM_LOC):
        for ctx in range(NUM_CTX):
            s = state_idx(loc, ctx)
            if loc == 0:
                A[0, s] = 1.0
            elif loc == 1:
                if ctx == 0:
                    A[1, s] = CUE_RELIABILITY
                    A[2, s] = 1 - CUE_RELIABILITY
                else:
                    A[2, s] = CUE_RELIABILITY
                    A[1, s] = 1 - CUE_RELIABILITY
            elif loc == 2:
                A[3 if ctx == 0 else 4, s] = 1.0
            else:
                A[3 if ctx == 1 else 4, s] = 1.0
    return A


def build_B():
    B = np.zeros((NUM_STATES, NUM_STATES, NUM_ACTIONS))
    for a in range(NUM_ACTIONS):
        for loc in range(NUM_LOC):
            for ctx in range(NUM_CTX):
                s = state_idx(loc, ctx)
                on_arm = loc in (2, 3)
                if on_arm:
                    next_loc = loc
                elif a == 0:
                    next_loc = loc
                else:
                    next_loc = a
                sn = state_idx(next_loc, ctx)
                B[sn, s, a] = 1.0
    return B


def build_C():
    C = np.zeros(NUM_OBS)
    C[3] = PREFERENCE
    C[4] = -PREFERENCE
    return C


def build_D():
    D = np.zeros(NUM_STATES)
    D[state_idx(0, 0)] = 0.5
    D[state_idx(0, 1)] = 0.5
    return D


def main():
    A_np = build_A()
    B_np = build_B()
    C_np = build_C()
    D_np = build_D()

    try:
        from pymdp.legacy.agent import Agent as PyMDPAgent
        from pymdp.legacy.utils import obj_array
    except ImportError as e:
        raise SystemExit(
            "pymdp.legacy not available. Install: uv pip install -r requirements.txt"
        ) from e

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

    # Fixed observation sequence: neutral, cue->L, reward
    observations = [0, 1, 3]

    posteriors_per_step = []
    actions_per_step = []
    efe_per_step = []

    for t, o in enumerate(observations):
        qs = agent.infer_states([o])
        q_pi, G = agent.infer_policies()
        action = agent.sample_action()

        posteriors_per_step.append(np.asarray(qs[0]).flatten().tolist())
        actions_per_step.append(int(np.asarray(action).flatten()[0]))
        efe_per_step.append(np.asarray(G).flatten().tolist())

    fixtures = {
        "cue_reliability": CUE_RELIABILITY,
        "preference":      PREFERENCE,
        "A":               A_np.tolist(),
        "B":               B_np.tolist(),
        "C":               C_np.tolist(),
        "D":               D_np.tolist(),
        "observations":    observations,
        "posteriors":      posteriors_per_step,
        "actions":         actions_per_step,
        "efe_per_step":    efe_per_step,
    }

    with open("tmaze_fixtures.json", "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"Wrote tmaze_fixtures.json with {len(observations)} step fixtures.")
    for t, (o, a, qs) in enumerate(zip(observations, actions_per_step,
                                          posteriors_per_step)):
        loc_marg = [sum(qs[s] for s in range(NUM_STATES) if s // 2 == loc)
                    for loc in range(NUM_LOC)]
        ctx_marg = [sum(qs[s] for s in range(NUM_STATES) if s % 2 == ctx)
                    for ctx in range(NUM_CTX)]
        print(f"  step {t}: obs={o} action={a} "
              f"loc={[round(v, 3) for v in loc_marg]} "
              f"ctx={[round(v, 3) for v in ctx_marg]}")


if __name__ == "__main__":
    main()
