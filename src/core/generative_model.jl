# `GenerativeModel` interface — the central abstraction.
#
# A generative model defines a joint distribution `P(o, s, a, π)` over
# observations, hidden states, actions, and policies. Concretely, the
# interface requires the agent to be able to query:
#
#   - the prior over initial states `P(s₀)`
#   - the observation likelihood `P(o | s)`
#   - the transition dynamics `P(s' | s, a)`
#   - the agent's preferences `log P̃(o)`
#   - the action / policy space
#
# Optional methods cover habits, goal-state priors (for inductive inference),
# and structural metadata (state factorization, observation modalities).

"""
    abstract type GenerativeModel end

A generative model implements the joint `P(o, s, a, π)` via the methods listed
under "Required" below. Every active-inference algorithm operates exclusively
through this interface — model implementations are free to be discrete,
continuous, hierarchical, neural, etc., as long as the methods are defined.

# Required methods

- `state_prior(m)` → `Distribution` over initial states `P(s₀)`
- `observation_distribution(m, s)` → `Distribution` over observations `P(o | s)`
- `transition_distribution(m, s, a)` → `Distribution` over next states `P(s' | s, a)`
- `log_preferences(m, o)` → `Real`, the log-preference `log P̃(o)`
- `action_space(m)` → iterable of valid actions

# Optional methods (with defaults)

- `log_policy_prior(m, π)` → `Real` (default: 0, i.e. uniform habit)
- `goal_state_prior(m)` → `Distribution` or `Nothing` (default: nothing —
  the model has no inductive goal-state prior)
- `state_factors(m)` → `Tuple{Vararg{Int}}` of factor sizes (default: `(1,)`)
- `observation_modalities(m)` → `Tuple{Vararg{Int}}` of modality sizes
  (default: `(1,)`)
- `policy_horizon(m)` → `Int` planning horizon hint (default: 1)
"""
abstract type GenerativeModel end

# --- Required methods (throw informative error if not implemented) ---

function state_prior(m::GenerativeModel)
    error("state_prior not implemented for $(typeof(m))")
end

function observation_distribution(m::GenerativeModel, s)
    error("observation_distribution not implemented for $(typeof(m))")
end

function transition_distribution(m::GenerativeModel, s, a)
    error("transition_distribution not implemented for $(typeof(m))")
end

function log_preferences(m::GenerativeModel, o)
    error("log_preferences not implemented for $(typeof(m))")
end

function action_space(m::GenerativeModel)
    error("action_space not implemented for $(typeof(m))")
end

# --- Optional methods with sensible defaults ---

"""
    log_policy_prior(m::GenerativeModel, π)

Log-prior over policies (Friston's "habit" vector E in the discrete POMDP
formulation). Default: `0.0` (uniform habit).
"""
log_policy_prior(::GenerativeModel, π) = 0.0

"""
    goal_state_prior(m::GenerativeModel) -> Union{Distribution, Nothing}

The inductive prior over *goal states* (Friston, Da Costa, Sajid 2023). When
non-`nothing`, planners that support inductive inference (e.g. `InductiveEFE`)
weight policies by their probability of reaching states under this prior.
Default: `nothing` (model has no inductive goal-state prior).
"""
goal_state_prior(::GenerativeModel) = nothing

"""
    state_factors(m::GenerativeModel) -> Tuple{Vararg{Int}}

Sizes of the model's state factors, as a tuple. For a single-factor 25-state
POMDP, `(25,)`. For a 25 × 2 factorized model, `(25, 2)`. Default: `(1,)`.
"""
state_factors(::GenerativeModel) = (1,)

"""
    observation_modalities(m::GenerativeModel) -> Tuple{Vararg{Int}}

Sizes of the model's observation modalities. Defaults to `(1,)`.
"""
observation_modalities(::GenerativeModel) = (1,)

"""
    policy_horizon(m::GenerativeModel) -> Int

Default planning horizon for this model. Planners may override this. Default: 1.
"""
policy_horizon(::GenerativeModel) = 1

"""
    nstates(m::GenerativeModel) -> Int

Total state-space size (product of factor sizes).
"""
nstates(m::GenerativeModel) = prod(state_factors(m))

"""
    nobservations(m::GenerativeModel) -> Int

Total observation-space size (product of modality sizes).
"""
nobservations(m::GenerativeModel) = prod(observation_modalities(m))

"""
    predict_states(m::GenerativeModel, prior, action) -> Distribution

Predict next-state belief: marginalize `transition_distribution(m, ·, action)`
over `s ~ prior`. Default falls back to constructing a per-state mixture; concrete
model types can override for closed-form efficiency (e.g. `B[:,:,a] · prior`
for `DiscretePOMDP`).
"""
function predict_states(m::GenerativeModel, prior, action)
    error("predict_states not implemented for $(typeof(m))")
end
