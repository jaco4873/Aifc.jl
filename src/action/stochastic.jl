# `Stochastic(α)` — sample an action from `softmax(α · log q(a))`.
#
# `q(a) = Σ_π q(π) · 𝟙[π_1 = a]` is the policy-marginal over the first
# action. The selector then sharpens it with precision `α`:
#
#     p(a) = softmax(α · log q(a))
#
# At α=1, p(a) = q(a) (since q(a) sums to 1 and softmax(log q) = q for
# normalized q). As α→∞, p(a) → δ_argmax(q). As α→0, p(a) → uniform.
#
# Parametric over `T<:Real` so that the field `α` can hold ForwardDiff
# `Dual` numbers or ReverseDiff `TrackedReal`s during Turing fitting —
# without this, Float64 conversion strips gradient information at
# construction time. (See ReverseDiff.jl issue #264 for the related
# `Vector{Real}` workaround the original ActiveInference.jl needed.)
#
# Multi-factor support: when `policies` is `Vector{Vector{Vector{Int}}}`
# (sequence of action vectors of length F), the action distribution is
# a `Distributions.Product` of per-factor `Categorical`s, each with its
# own α-tempered softmax of the per-factor marginal.

using Random: AbstractRNG
using Distributions: Categorical

"""
    Stochastic(; α=16.0)

Sample an action from `softmax(α · log q_marginal(a))` where `q_marginal(a)`
is the policy-marginal over the first action. Higher `α` makes selection
more deterministic. Parametric over `α`'s type so that gradients flow
through during Turing fitting.
"""
struct Stochastic{T<:Real} <: ActionSelector
    α::T
end
Stochastic(; α::Real = 16.0) = Stochastic(float(α))

# --- Single-factor: policies are Vector{Vector{Int}}, actions are Int ---

function action_distribution(sel::Stochastic,
                              q_pi::AbstractVector{<:Real},
                              policies::AbstractVector{<:AbstractVector{<:Integer}})
    return Categorical(_action_probabilities(sel, q_pi, policies))
end

function log_action_probabilities(sel::Stochastic,
                                    q_pi::AbstractVector{<:Real},
                                    policies::AbstractVector{<:AbstractVector{<:Integer}})
    p = _action_probabilities(sel, q_pi, policies)
    return log.(p .+ eps(eltype(p)))
end

function _action_marginal(q_pi::AbstractVector{<:Real},
                            policies::AbstractVector{<:AbstractVector{<:Integer}})
    n_actions = maximum(first(π) for π in policies)
    out = zeros(promote_type(eltype(q_pi), Float64), n_actions)
    @inbounds for (i, π) in enumerate(policies)
        out[first(π)] += q_pi[i]
    end
    return out
end

function _action_probabilities(sel::Stochastic, q_pi, policies::AbstractVector{<:AbstractVector{<:Integer}})
    marg = _action_marginal(q_pi, policies)
    log_marg = log.(marg .+ eps(eltype(marg)))
    return softmax(log_marg, sel.α)
end


# --- Multi-factor: policies are Vector{Vector{Vector{Int}}}; per-step
#     action is a Vector{Int} of length F. Action distribution is a
#     Product of per-factor Categoricals. ---

function action_distribution(sel::Stochastic,
                              q_pi::AbstractVector{<:Real},
                              policies::AbstractVector{<:AbstractVector{<:AbstractVector{<:Integer}}})
    F = length(first(first(policies)))
    return product_belief([
        Categorical(_action_probabilities_factor(sel, q_pi, policies, f)) for f in 1:F
    ])
end

function log_action_probabilities(sel::Stochastic,
                                    q_pi::AbstractVector{<:Real},
                                    policies::AbstractVector{<:AbstractVector{<:AbstractVector{<:Integer}}})
    F = length(first(first(policies)))
    return [log.(_action_probabilities_factor(sel, q_pi, policies, f) .+
                  eps(eltype(_action_probabilities_factor(sel, q_pi, policies, f))))
            for f in 1:F]
end

# Per-factor action marginal: q(a_f) = Σ_{π : π[1][f] = a_f} q(π)
function _action_marginal_factor(q_pi::AbstractVector{<:Real},
                                   policies::AbstractVector{<:AbstractVector{<:AbstractVector{<:Integer}}},
                                   f::Integer)
    K_f = maximum(π[1][f] for π in policies)
    out = zeros(promote_type(eltype(q_pi), Float64), K_f)
    @inbounds for (i, π) in enumerate(policies)
        out[π[1][f]] += q_pi[i]
    end
    return out
end

function _action_probabilities_factor(sel::Stochastic, q_pi, policies, f::Integer)
    marg = _action_marginal_factor(q_pi, policies, f)
    log_marg = log.(marg .+ eps(eltype(marg)))
    return softmax(log_marg, sel.α)
end
