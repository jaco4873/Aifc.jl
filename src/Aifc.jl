"""
    Aifc

A ground-up reimplementation of active inference for Julia, designed around
four principles:

1. Generative-model-agnostic (POMDP, continuous-time state space, hierarchical,
   functional, inductive — all via the `GenerativeModel` interface)
2. Backend-agnostic differentiation via `DifferentiationInterface.jl`
3. Mathematically trustworthy — every interface ships with property tests
4. Composable — agents are `(model, inferences, policy_inference, action_selection)`

# Module organization

- **`math/`** — pure functions: softmax, KL divergence, entropy, free energy
- **`core/`** — abstract types: `GenerativeModel`, `Inference`, `PolicyInference`,
  `ActionSelector`, `Agent`, `AgentHistory`
- **`models/`** — default model implementations
- **`inference/`** — default state-inference algorithms
- **`planning/`** — default policy-inference algorithms
- **`action/`** — default action selectors
- **`learning/`** — default parameter-learning algorithms
- **`environments/`** — example environments (T-maze, etc.)
- **`testing/`** — conformance test suites for downstream implementations
"""
module Aifc

using LinearAlgebra: dot

# --- Layer 1: math primitives ---
include("math/softmax.jl")
include("math/information.jl")
include("math/free_energy.jl")

# Math primitives. NOTE: `entropy` is also exported by Distributions.jl;
# downstream users should qualify if both are loaded
# (e.g. `Aifc.entropy(::AbstractVector)` vs
#  `Distributions.entropy(::Distribution)`).
export softmax, logsoftmax
export entropy, kl_divergence, cross_entropy, bayesian_surprise
export variational_free_energy, accuracy, complexity

# --- Layer 2: core interfaces ---
include("core/beliefs.jl")
export BeliefStyle, CategoricalStyle, GaussianStyle, ProductStyle, ParticleStyle, UnknownStyle
export belief_style, isnormalized, marginal, nfactors, factor_probs
export categorical_belief, product_belief

include("core/generative_model.jl")
export GenerativeModel
export state_prior, observation_distribution, transition_distribution
export log_preferences, action_space, log_policy_prior, goal_state_prior
export state_factors, observation_modalities, policy_horizon
export nstates, nobservations, predict_states

include("core/inference.jl")
export Inference
export infer_states, free_energy

include("core/parameter_learning.jl")
export ParameterLearning
export infer_parameters

include("core/policy_inference.jl")
export PolicyInference
export posterior_policies, expected_free_energy
export pragmatic_value, epistemic_value, enumerate_policies

include("core/action_selection.jl")
export ActionSelector
export action_distribution, sample_action, log_action_probabilities

include("core/history.jl")
export AgentStep, AgentHistory
export current_belief, last_action, observation_history, belief_history
export action_history, free_energy_history, push_step!

include("core/agent.jl")
export Agent, step!, reset!

# --- Layer 3: default implementations ---

include("models/discrete_pomdp.jl")
export AbstractDiscretePOMDP, DiscretePOMDP, random_pomdp, nactions

include("models/multi_factor_pomdp.jl")
export MultiFactorDiscretePOMDP

include("models/functional.jl")
export FunctionalModel

include("inference/fixed_point.jl")
export FixedPointIteration

include("learning/dirichlet_conjugate.jl")
export DirichletConjugate

include("planning/enumerative.jl")
export EnumerativeEFE

include("planning/sophisticated.jl")
export SophisticatedInference

include("planning/inductive.jl")
export InductiveEFE

include("action/stochastic.jl")
export Stochastic

include("action/deterministic.jl")
export Deterministic

# --- Layer 4: integrations (stubs; implementations in ext/) ---

include("integrations/actionmodels.jl")
export active_inference_actionmodel

# --- Layer 5: example environments ---

include("environments/tmaze.jl")
export tmaze_model, tmaze_model_multi_factor
export tmaze_state_index, TMAZE_ACTIONS, TMAZE_MF_ACTIONS

# --- Layer 6: testing utilities ---
include("testing/conformance.jl")

end # module
