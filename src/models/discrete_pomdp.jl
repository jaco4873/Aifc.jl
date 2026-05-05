# `DiscretePOMDP` — the classic A/B/C/D/E generative model.
#
# Represents
#
#   A : P(o | s)             [num_obs × num_states]      column-stochastic
#   B : P(s' | s, a)         [num_states × num_states × num_actions]
#                            B[s', s, a] = P(s' | s, a); column-stochastic per a
#   C : log P̃(o)            [num_obs]                    log preferences (nats)
#   D : P(s_0)               [num_states]                 prior over initial state
#   E : (optional) habit     [num_policies]               policy prior; passed
#                                                         to the planner per-policy
#
# Optional Dirichlet hyperparameters `pA`, `pB`, `pD` of the same shapes as
# `A`, `B`, `D` enable parameter learning (see `DirichletConjugate`).
#
# A `goal` field carrying a probability vector over states enables inductive
# inference (see `InductiveEFE`). When set, `goal_state_prior(m)` returns
# `Categorical(goal)`.
#
# Single-factor / single-modality / single-action-factor for v0.1. Multi-factor
# state spaces and multi-modality observations are deferred — they require a
# more complex storage layout but no fundamentally new design.

using Distributions: Categorical
using LinearAlgebra: I

"""
    DiscretePOMDP(A, B, C, D; E=nothing, pA=nothing, pB=nothing, pD=nothing,
                              goal=nothing, check=true)

Construct a discrete POMDP from likelihood matrix `A`, transition tensor `B`,
log-preference vector `C`, and initial-state prior `D`. Optional Dirichlet
hyperparameter tensors `pA`, `pB`, `pD`; optional habit vector `E`; optional
goal-state prior `goal` (for inductive inference).

When `check=true` (default), validates that `A` and `B[:,:,a]` are
column-stochastic and that `D` and `goal` (if set) sum to 1.

# Shape conventions

- `A`: `[num_obs × num_states]`,    `A[o, s] = P(o | s)`
- `B`: `[num_states × num_states × num_actions]`,    `B[s', s, a] = P(s' | s, a)`
- `C`: `[num_obs]`, log preferences (nats)
- `D`: `[num_states]`, sums to 1
- `goal`: `[num_states]`, sums to 1, or `nothing`
"""
struct DiscretePOMDP{T<:Real} <: GenerativeModel
    A::Matrix{T}
    B::Array{T, 3}
    C::Vector{T}
    D::Vector{T}
    E::Union{Nothing, Vector{T}}
    pA::Union{Nothing, Matrix{T}}
    pB::Union{Nothing, Array{T, 3}}
    pD::Union{Nothing, Vector{T}}
    goal::Union{Nothing, Vector{T}}
end

function DiscretePOMDP(A::AbstractMatrix, B::AbstractArray{<:Any, 3},
                       C::AbstractVector, D::AbstractVector;
                       E::Union{Nothing, AbstractVector} = nothing,
                       pA::Union{Nothing, AbstractMatrix} = nothing,
                       pB::Union{Nothing, AbstractArray{<:Any, 3}} = nothing,
                       pD::Union{Nothing, AbstractVector} = nothing,
                       goal::Union{Nothing, AbstractVector} = nothing,
                       check::Bool = true,
                       atol::Real = 1e-8)
    num_obs = size(A, 1)
    num_states = size(A, 2)
    num_actions = size(B, 3)

    size(B, 1) == num_states || throw(DimensionMismatch(
        "DiscretePOMDP: B has $(size(B,1)) next-states, expected $num_states (matching A)"))
    size(B, 2) == num_states || throw(DimensionMismatch(
        "DiscretePOMDP: B has $(size(B,2)) source-states, expected $num_states"))
    length(C) == num_obs || throw(DimensionMismatch(
        "DiscretePOMDP: C has $(length(C)) entries, expected $num_obs"))
    length(D) == num_states || throw(DimensionMismatch(
        "DiscretePOMDP: D has $(length(D)) entries, expected $num_states"))

    if check
        for s in axes(A, 2)
            colsum = sum(@view A[:, s])
            isapprox(colsum, 1; atol=atol) ||
                throw(ArgumentError("DiscretePOMDP: A column $s sums to $colsum, expected 1 (atol=$atol)"))
        end
        for a in axes(B, 3), s in axes(B, 2)
            colsum = sum(@view B[:, s, a])
            isapprox(colsum, 1; atol=atol) ||
                throw(ArgumentError("DiscretePOMDP: B[:, $s, $a] sums to $colsum, expected 1 (atol=$atol)"))
        end
        Dsum = sum(D)
        isapprox(Dsum, 1; atol=atol) ||
            throw(ArgumentError("DiscretePOMDP: D sums to $Dsum, expected 1"))
        if goal !== nothing
            gsum = sum(goal)
            isapprox(gsum, 1; atol=atol) ||
                throw(ArgumentError("DiscretePOMDP: goal sums to $gsum, expected 1"))
            length(goal) == num_states || throw(DimensionMismatch(
                "DiscretePOMDP: goal has $(length(goal)) entries, expected $num_states"))
        end
    end

    T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D),
                     E === nothing  ? Float64 : eltype(E),
                     pA === nothing ? Float64 : eltype(pA),
                     pB === nothing ? Float64 : eltype(pB),
                     pD === nothing ? Float64 : eltype(pD),
                     goal === nothing ? Float64 : eltype(goal))
    T = float(T)

    return DiscretePOMDP{T}(
        Matrix{T}(A),
        Array{T, 3}(B),
        Vector{T}(C),
        Vector{T}(D),
        E === nothing ? nothing : Vector{T}(E),
        pA === nothing ? nothing : Matrix{T}(pA),
        pB === nothing ? nothing : Array{T, 3}(pB),
        pD === nothing ? nothing : Vector{T}(pD),
        goal === nothing ? nothing : Vector{T}(goal),
    )
end

# --- Required GenerativeModel interface ---

state_prior(m::DiscretePOMDP) = Categorical(m.D)

function observation_distribution(m::DiscretePOMDP, s::Integer)
    1 <= s <= size(m.A, 2) ||
        throw(BoundsError("observation_distribution: state $s out of range [1, $(size(m.A, 2))]"))
    return Categorical(m.A[:, s])
end

function transition_distribution(m::DiscretePOMDP, s::Integer, a::Integer)
    1 <= s <= size(m.B, 2) ||
        throw(BoundsError("transition_distribution: state $s out of range"))
    1 <= a <= size(m.B, 3) ||
        throw(BoundsError("transition_distribution: action $a out of range"))
    return Categorical(m.B[:, s, a])
end

log_preferences(m::DiscretePOMDP, o::Integer) = m.C[o]

action_space(m::DiscretePOMDP) = 1:size(m.B, 3)

# --- Optional GenerativeModel interface ---

goal_state_prior(m::DiscretePOMDP) = m.goal === nothing ? nothing : Categorical(m.goal)

state_factors(m::DiscretePOMDP) = (size(m.A, 2),)
observation_modalities(m::DiscretePOMDP) = (size(m.A, 1),)
nstates(m::DiscretePOMDP) = size(m.A, 2)
nobservations(m::DiscretePOMDP) = size(m.A, 1)
nactions(m::DiscretePOMDP) = size(m.B, 3)

# --- Predict-states (closed-form: B[:,:,a] · q) ---

function predict_states(m::DiscretePOMDP, prior::Categorical, a::Integer)
    p_next = m.B[:, :, a] * probs(prior)
    return Categorical(p_next ./ sum(p_next))   # renormalize for numerical safety
end

# Allow predict_states to also accept a raw vector; converts to Categorical
function predict_states(m::DiscretePOMDP, prior::AbstractVector{<:Real}, a::Integer)
    p_next = m.B[:, :, a] * prior
    return Categorical(p_next ./ sum(p_next))
end

# --- Convenience constructors ---

"""
    random_pomdp(num_obs, num_states, num_actions; rng=Xoshiro(0))

Construct a `DiscretePOMDP` with random column-stochastic A and B (each
column drawn as `softmax(randn)`), zero log-preferences, uniform initial
state prior. Useful for testing.
"""
function random_pomdp(num_obs::Integer, num_states::Integer, num_actions::Integer;
                       rng = Xoshiro(0))
    A = zeros(Float64, num_obs, num_states)
    for s in 1:num_states
        A[:, s] = softmax(randn(rng, num_obs))
    end
    B = zeros(Float64, num_states, num_states, num_actions)
    for a in 1:num_actions, s in 1:num_states
        B[:, s, a] = softmax(randn(rng, num_states))
    end
    C = zeros(Float64, num_obs)
    D = fill(1 / num_states, num_states)
    return DiscretePOMDP(A, B, C, D; check=false)   # construction is exact, skip checks
end

# Need to import Xoshiro for the keyword default to work at call time.
using Random: Xoshiro
