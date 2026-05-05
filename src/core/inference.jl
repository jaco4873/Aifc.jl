# `Inference` interface — variational inference over latent variables.
#
# Friston's view: perception, learning, and (some forms of) precision
# adaptation are all the same operation — gradient descent on free energy —
# at different timescales. This interface unifies them under one abstract
# type. A concrete algorithm declares which targets it supports by which
# methods it implements:
#
#   - `infer_states(alg, m, prior, observation)`  —  per-step state inference
#   - `infer_parameters(alg, m, history)`         —  parameter learning
#   - `infer_jointly(alg, m, history, observation)` — joint state+param+hyperparam
#                                                     inference (e.g. generalized
#                                                     filtering)
#
# Every algorithm that does state inference must also implement
#   - `free_energy(alg, m, prior, observation, q)` — VFE at the posterior
# so the conformance suite can verify monotone-decrease invariants.

"""
    abstract type Inference end

Variational inference algorithm. Concrete subtypes specify which targets
they support by overriding methods.

# Methods

- `infer_states(alg, m, prior, observation)` → posterior `Distribution`

  Single-step filtering: given the predicted prior and a new observation,
  return the posterior over hidden states by minimizing F[q].

- `free_energy(alg, m, prior, observation, q)` → `Real`

  Variational free energy at the given posterior. Required if `infer_states`
  is implemented. Conformance tests verify F[q] is non-increasing across
  iterations of any iterative algorithm.

- `infer_parameters(alg, m, history)` → updated `GenerativeModel`

  Multi-step parameter learning: return a model with updated parameter
  posteriors (e.g. Dirichlet hyperparameters) given a full trajectory.

- `infer_jointly(alg, m, history, observation)` → `NamedTuple`

  Joint inference over states + parameters + hyperparameters. Used by
  generalized filtering and similar algorithms that don't separate timescales.
  Returns `(states::Distribution, model::GenerativeModel, free_energy::Real)`.

Use [`supports_states`](@ref), [`supports_parameters`](@ref),
[`supports_joint`](@ref) to query at runtime.
"""
abstract type Inference end

# --- LatentSubset type tags (reserved for future joint-inference dispatch) ---

abstract type LatentSubset end
struct States <: LatentSubset end
struct Parameters <: LatentSubset end
struct Hyperparameters <: LatentSubset end
struct Joint{T<:Tuple} <: LatentSubset end

# --- Default fallbacks ---

function infer_states(alg::Inference, m::GenerativeModel, prior, observation)
    error("$(typeof(alg)) does not implement infer_states. Available targets: $(supported_targets(alg))")
end

function infer_parameters(alg::Inference, m::GenerativeModel, history)
    error("$(typeof(alg)) does not implement infer_parameters. Available targets: $(supported_targets(alg))")
end

function infer_jointly(alg::Inference, m::GenerativeModel, history, observation)
    error("$(typeof(alg)) does not implement infer_jointly. Available targets: $(supported_targets(alg))")
end

function free_energy(alg::Inference, m::GenerativeModel, prior, observation, q)
    error("free_energy not implemented for $(typeof(alg))")
end

# --- Capability queries ---
#
# Each algorithm declares its capabilities by overriding these. Defaults are
# `false`; concrete algorithms override the appropriate ones.
# (Using an explicit trait method rather than `hasmethod` because the latter
# requires exactly-matching argument types — too brittle for a public API.)

supports_states(::Inference)     = false
supports_parameters(::Inference) = false
supports_joint(::Inference)      = false

function supported_targets(alg::Inference)
    targets = Symbol[]
    supports_states(alg) && push!(targets, :states)
    supports_parameters(alg) && push!(targets, :parameters)
    supports_joint(alg) && push!(targets, :joint)
    return targets
end
