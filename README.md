# Aifc.jl

[![CI](https://github.com/jaco4873/Aifc.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jaco4873/Aifc.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/jaco4873/Aifc.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jaco4873/Aifc.jl)

A Julia package for [active inference](https://en.wikipedia.org/wiki/Free_energy_principle),
written as an alternative to
[ActiveInference.jl](https://github.com/samuelnehrer02/ActiveInference.jl)
(Laursen & Nehrer, 2024) — the first Julia package for active inference, which
remains the more pymdp-faithful choice. The two share no code; this one explores
a different design point built around abstract-type interfaces, parametric
autodiff types, and a separated math/inference/planning/action layer cake.

Four principles:

1. **Generative-model-agnostic.** The classic A/B/C/D POMDP is one default
   among several; users can plug in custom models that implement the
   `GenerativeModel` interface.
2. **Composable.** An agent is `(model, state_inference, policy_inference,
   action_selection, parameter_learning)`. Each piece is independently
   swappable.
3. **Mathematically trustworthy.** Every interface ships with property
   tests; numerical agreement with both reference implementations
   ([pymdp](https://github.com/infer-actively/pymdp) targets) and an
   independently-validated TypeScript primer is checked in CI.
4. **Backend-agnostic differentiation.** No autodiff backend is named in
   the core; downstream users will compose with
   [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)
   to choose ForwardDiff / ReverseDiff / Zygote / Enzyme / Mooncake at call
   time.

## Status

**v0.1 development.** Core layers are landed and tested; specific
extensions (ActionModels.jl integration, multi-factor states, generalized
filtering, hierarchical models, MCTS) are explicitly out of scope for the
initial release and will land in follow-up PRs.

3,010 tests passing on Julia 1.10 + 1.12 (Linux + macOS).

## What's shipping

| Layer | Components |
|---|---|
| **Math primitives** | `softmax` (with tempered variant), `logsoftmax`, `entropy`, `kl_divergence`, `cross_entropy`, `bayesian_surprise`, `variational_free_energy` with `accuracy`/`complexity` decomposition |
| **Beliefs** | `Distributions.jl`-based; `belief_style` trait for invariant testing; `categorical_belief` / `product_belief` constructors |
| **Models** | `DiscretePOMDP` (single-factor A/B/C/D/E with optional Dirichlet hyperparameters and goal-state prior), `FunctionalModel` (closure-based) |
| **State inference** | `FixedPointIteration` (single-factor closed form q(s) ∝ A[o,s]·prior[s]) |
| **Policy inference** | `EnumerativeEFE`, `SophisticatedInference` (Bellman recursion), `InductiveEFE` (goal-state-prior weighting; wraps any base planner) |
| **Action selection** | `Stochastic(α)`, `Deterministic` |
| **Parameter learning** | `DirichletConjugate` (online conjugate update, optional digamma-corrected effective likelihood) |
| **Environments** | `tmaze_model` |
| **Testing** | `Aifc.Testing` submodule with conformance suites for every interface |
| **Integrations** | `ActionModels.jl` package extension (auto-loaded when both packages are present) — exposes our agents as `ActionModels.ActionModel` for simulation via `init_agent` / `simulate!` and parameter introspection via `get_parameters` / `set_parameters!` |

## Quick start

```julia
using Aifc
using Distributions: probs

# Build the canonical T-maze
m = tmaze_model(cue_reliability=0.95, preference=4.0)

# Compose an agent with sophisticated planning at depth 2
agent = Agent(
    m,
    FixedPointIteration(),                       # state inference
    SophisticatedInference(γ=16.0, horizon=2),    # planning
    Deterministic(),                              # action selection
)

# Step 1: agent at C, observes "neutral"
a1 = step!(agent, 1)
# > 2  (= go to cue: SophisticatedInference picks the information-seeking action)

# Step 2: agent at Q, observes cue→L
a2 = step!(agent, 2)
# > 3  (= go-left: agent has resolved context and commits to the rewarded arm)

# Inspect the trajectory
agent.history.steps[end].free_energy
agent.history.steps[end].policy_posterior
```

## ActionModels.jl integration

Active-inference agents compose with ActionModels.jl (the Aarhus/ilabcode
behavioral-modeling toolkit) via a package extension. With both packages
loaded:

```julia
using Aifc, ActionModels

m = tmaze_model(cue_reliability=0.95, preference=4.0)
am = active_inference_actionmodel(
    model            = m,
    state_inference  = FixedPointIteration(),
    policy_inference = SophisticatedInference(γ=8.0, horizon=2),
    action_selection = Stochastic(α=8.0),
)

agent  = init_agent(am; save_history=true)
actions = simulate!(agent, [1, 2, 4])

# Parameter introspection (Turing-sampleable parameters):
get_parameters(agent)                                # NamedTuple of values
set_parameters!(agent, :policy_precision, 16.0)      # update γ
```

v0.1 limitations: single observation modality + single action factor;
`Stochastic` action selection only for fitting; online Dirichlet learning
not supported through this entry point; Turing-NUTS fitting works for
forward simulation but full propagation of `ForwardDiff.Dual` numbers
through inference is a follow-up.

## Designing your own model

Subtype `GenerativeModel` and implement five methods:

```julia
struct MyModel <: GenerativeModel
    # ...
end

state_prior(m::MyModel)                          = ...   # Distribution
observation_distribution(m::MyModel, s)          = ...   # Distribution
transition_distribution(m::MyModel, s, a)        = ...   # Distribution
log_preferences(m::MyModel, o)                   = ...   # Real
action_space(m::MyModel)                         = ...   # iterable
predict_states(m::MyModel, prior, action)        = ...   # Distribution
```

Verify it satisfies the contract:

```julia
using Aifc.Testing
using Test
@testset begin
    test_generative_model(MyModel())
    # Bring your own state-inference, policy-inference, and action selectors;
    # the testing submodule verifies they conform too.
end
```

## Mathematical correctness

The package's credibility lives in the test suite. We verify:

- **Probabilistic invariants** — beliefs sum to 1, transition columns
  normalized, KL ≥ 0, entropy ≥ 0
- **Free energy invariants** — Gibbs' inequality `F[q] ≥ -log P(o)` with
  equality at the exact posterior; F = -accuracy + complexity decomposition
- **EFE decomposition** — `G(π) = -(pragmatic_value(π) + epistemic_value(π))`
  for every conforming planner; conformance suite verifies on every implementation
- **Information-theoretic identities** — Bayesian surprise = expected KL;
  H(p, q) = entropy(p) + KL(p || q); softmax shift-invariance
- **Closed-form sanity checks** — single-factor FPI closed form;
  uniform-A → posterior = prior; perfectly-informative-A → I = H[prior]
- **Gradient correctness** — every analytic gradient verified against
  `FiniteDifferences.jl` central-5 stencil to atol=1e-7
- **Cross-implementation verification** — T-maze's best-vanilla-2-step =
  `2·log(2)` and sophisticated depth-2 EFE = 4.293 match the
  TypeScript primer's machine-precision-validated reference values
- **Numerical stability sweeps** — softmax with inputs in `[-1e10, +1e10]`
  produces no NaNs / Infs / underflow

3,010 assertions in CI on every push.

## What's coming (roughly in priority order)

- ActionModels.jl integration (package extension; lets the agent fit to
  behavioral data via Turing.jl)
- Multi-factor state spaces and multi-modality observations
- Continuous-time state-space models + generalized filtering (Buckley/Kim/
  McGregor/Seth 2017; Friston 2008)
- Hierarchical models with timescale separation
- MCTS-based planning for large policy spaces
- Amortized inference (neural q(s|o))
- Simulation-based calibration (SBC) infrastructure

## Installation

Not yet registered. From the Julia REPL:

```julia
import Pkg
Pkg.develop(path="path/to/Aifc.jl")
```

## Running the tests

```bash
julia --project=. -e 'import Pkg; Pkg.test()'
```

## Running coverage locally

```bash
julia --project=. scripts/coverage.jl
```

Runs the test suite under `--code-coverage=user` tracking, writes an
LCOV report to `.coverage/lcov.info`, and prints a per-file summary
sorted by coverage percentage. The CI workflow does the same and uploads
to Codecov on the canonical matrix entry (Linux + Julia stable).

## License

MIT.

## References

The package implements algorithms and conventions from:

- Friston (2010) "The free-energy principle: a unified brain theory?"
  *Nature Reviews Neuroscience* 11: 127–138.
- Friston (2017) "Active inference: a process theory." *Neural Computation*
  29: 1.
- Friston, Da Costa, Hafner, Hesp, Parr (2021) "Sophisticated inference."
  *Neural Computation* 33(3): 713–763.
- Friston, Da Costa, Sajid et al. (2023) "Active inference and intentional
  behaviour." *arXiv*:2312.07547.
- Buckley, Kim, McGregor, Seth (2017) "The free energy principle for action
  and perception: A mathematical review." *J. Mathematical Psychology* 81:
  55–79.
- Heins, Mirza, Parr, Friston, Kagan, Pio-Lopez (2022) "pymdp: A Python
  library for active inference in discrete state spaces." *JOSS* 7(73): 4098.
- Laursen, Nehrer (2024) ActiveInference.jl — first Julia package for
  active inference. <https://github.com/samuelnehrer02/ActiveInference.jl>.
- Da Costa, Parr, Sajid, Veselic, Neacsu, Friston (2020) "Active inference
  on discrete state-spaces: A synthesis." *J. Mathematical Psychology* 99:
  102447.
- Smith, Friston, Whyte (2022) "A step-by-step tutorial on active inference
  and its application to empirical data." *J. Mathematical Psychology*
  107: 102632.
