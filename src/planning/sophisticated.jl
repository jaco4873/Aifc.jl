# `SophisticatedInference` — recursive Bellman expectation over the policy tree.
#
# Friston, Da Costa, Hafner, Hesp, Parr (2021) "Sophisticated Inference"
# (Neural Comp 33(3): 713-763). Rather than committing to a fixed action
# sequence, the agent considers what its FUTURE BELIEFS will look like under
# each candidate observation, and at each branch picks the action that is
# locally optimal. The Bellman recursion (in the maximize convention,
# `total = -G`) is:
#
#     total*(qs, d) = max_a [
#         total_immediate(a, qs)
#         +  E_{q(o|a)} [total*(qs[o←Bayes], d-1)]
#     ]
#
# We compute (pragmatic, epistemic) components separately along the
# *optimal* continuation, so the decomposition identity
#   `G = -(pragmatic + epistemic)`
# still holds for the conformance suite.
#
# Each "policy" exposed via `enumerate_policies` is just a single first
# action — the rest of the path is implicitly the optimal continuation.

using Distributions: Categorical, probs
using LinearAlgebra: dot

"""
    SophisticatedInference(; γ=16.0, horizon=2, use_pragmatic=true, use_epistemic=true)

Bellman-recursive policy inference. At each branch the agent picks the
action that minimizes expected free energy in expectation over future
observations.

`horizon` is the maximum recursion depth. With `horizon=1` this reduces to
single-step `EnumerativeEFE`. With `horizon=2` you get the canonical
Bayesian-decision case. Branching factor is
`(nactions(m) · nobservations(m))^horizon`, so depth must stay modest.
"""
struct SophisticatedInference{T<:Real} <: PolicyInference
    γ::T
    horizon::Int
    use_pragmatic::Bool
    use_epistemic::Bool
end

SophisticatedInference(; γ::Real = 16.0,
                          horizon::Integer = 2,
                          use_pragmatic::Bool = true,
                          use_epistemic::Bool = true) =
    SophisticatedInference(float(γ), Int(horizon), use_pragmatic, use_epistemic)

function pragmatic_value(planner::SophisticatedInference,
                          m::DiscretePOMDP,
                          qs::Categorical,
                          π::AbstractVector{<:Integer})
    _, prag, _ = _sophisticated_eval(planner, m, qs, first(π), planner.horizon)
    return prag
end

function epistemic_value(planner::SophisticatedInference,
                          m::DiscretePOMDP,
                          qs::Categorical,
                          π::AbstractVector{<:Integer})
    _, _, epi = _sophisticated_eval(planner, m, qs, first(π), planner.horizon)
    return epi
end

function expected_free_energy(planner::SophisticatedInference,
                                m::DiscretePOMDP,
                                qs::Categorical,
                                π::AbstractVector{<:Integer})
    G, _, _ = _sophisticated_eval(planner, m, qs, first(π), planner.horizon)
    return G
end

function posterior_policies(planner::SophisticatedInference,
                              m::DiscretePOMDP,
                              qs::Categorical,
                              policies::AbstractVector)
    G = [expected_free_energy(planner, m, qs, π) for π in policies]
    q_pi = softmax(.-G, planner.γ)
    return q_pi, G
end

function enumerate_policies(planner::SophisticatedInference,
                              m::DiscretePOMDP,
                              ::Categorical)
    # The "policies" sophisticated inference reasons about are just the
    # next-action choices; the future is implicit in the optimal continuation.
    return [[a] for a in 1:nactions(m)]
end

# Internal: evaluate (G, pragmatic_total, epistemic_total) along the
# OPTIMAL continuation starting with action `a` at depth `depth`.
function _sophisticated_eval(planner::SophisticatedInference,
                              m::DiscretePOMDP,
                              qs::Categorical,
                              a::Integer,
                              depth::Integer)
    cur_qs_p = m.B[:, :, a] * collect(probs(qs))
    cur_qs_p ./= sum(cur_qs_p)        # numerical-safety renormalization
    qo        = m.A * cur_qs_p

    prag_imm = planner.use_pragmatic ? dot(qo, m.C) : 0.0
    epi_imm  = planner.use_epistemic ? bayesian_surprise(m.A, cur_qs_p) : 0.0

    if depth <= 1
        G = -(prag_imm + epi_imm)
        return (G, prag_imm, epi_imm)
    end

    # Recurse: at each predicted next observation, the agent picks the
    # locally-optimal action and we average over q(o|a).
    future_prag = 0.0
    future_epi  = 0.0
    @inbounds for o in eachindex(qo)
        qo[o] > 1e-12 || continue
        # Posterior given we observe o: q(s | o, π) ∝ q(s | π) · A[o, s]
        post = cur_qs_p .* (@view m.A[o, :])
        post ./= sum(post)
        post_belief = Categorical(post)

        best_G = Inf
        best_prag_branch = 0.0
        best_epi_branch  = 0.0
        for a_next in 1:nactions(m)
            G_branch, prag_branch, epi_branch =
                _sophisticated_eval(planner, m, post_belief, a_next, depth - 1)
            if G_branch < best_G
                best_G = G_branch
                best_prag_branch = prag_branch
                best_epi_branch  = epi_branch
            end
        end
        future_prag += qo[o] * best_prag_branch
        future_epi  += qo[o] * best_epi_branch
    end

    total_prag = prag_imm + future_prag
    total_epi  = epi_imm  + future_epi
    return (-(total_prag + total_epi), total_prag, total_epi)
end
