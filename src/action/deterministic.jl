# `Deterministic` — argmax of the action marginal. Multi-factor: per-factor argmax.

using Random: AbstractRNG
using Distributions: Categorical

"""
    Deterministic()

Pick the action with maximum policy-marginal probability. The induced
"distribution" is a one-hot Categorical at the argmax — *degenerate* for
fitting purposes (any non-argmax observed action has zero likelihood).
Use `Stochastic` for fitting; `Deterministic` is fine for simulation.
"""
struct Deterministic <: ActionSelector end


# --- Single-factor ---

function action_distribution(::Deterministic,
                              q_pi::AbstractVector{<:Real},
                              policies::AbstractVector{<:AbstractVector{<:Integer}})
    n_actions = maximum(first(π) for π in policies)
    marg = zeros(eltype(q_pi), n_actions)
    @inbounds for (i, π) in enumerate(policies)
        marg[first(π)] += q_pi[i]
    end
    a_star = argmax(marg)
    p = zeros(eltype(q_pi), n_actions)
    p[a_star] = 1.0
    return Categorical(p)
end

function sample_action(::Deterministic,
                       q_pi::AbstractVector{<:Real},
                       policies::AbstractVector{<:AbstractVector{<:Integer}},
                       ::AbstractRNG)
    n_actions = maximum(first(π) for π in policies)
    marg = zeros(eltype(q_pi), n_actions)
    @inbounds for (i, π) in enumerate(policies)
        marg[first(π)] += q_pi[i]
    end
    return argmax(marg)
end

function log_action_probabilities(::Deterministic,
                                    q_pi::AbstractVector{<:Real},
                                    policies::AbstractVector{<:AbstractVector{<:Integer}})
    n_actions = maximum(first(π) for π in policies)
    marg = zeros(eltype(q_pi), n_actions)
    @inbounds for (i, π) in enumerate(policies)
        marg[first(π)] += q_pi[i]
    end
    a_star = argmax(marg)
    out = fill(typemin(Float64), n_actions)
    out[a_star] = 0.0
    return out
end


# --- Multi-factor: per-factor argmax of the action marginal ---

function action_distribution(::Deterministic,
                              q_pi::AbstractVector{<:Real},
                              policies::AbstractVector{<:AbstractVector{<:AbstractVector{<:Integer}}})
    F = length(first(first(policies)))
    marginals = [_action_marginal_factor(q_pi, policies, f) for f in 1:F]
    return product_belief([
        let p = zeros(eltype(q_pi), length(m_f)); p[argmax(m_f)] = 1.0; Categorical(p) end
        for m_f in marginals
    ])
end

function sample_action(::Deterministic,
                       q_pi::AbstractVector{<:Real},
                       policies::AbstractVector{<:AbstractVector{<:AbstractVector{<:Integer}}},
                       ::AbstractRNG)
    F = length(first(first(policies)))
    return [argmax(_action_marginal_factor(q_pi, policies, f)) for f in 1:F]
end
