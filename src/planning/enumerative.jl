# `EnumerativeEFE` — enumerate policies, compute G(π) for each, form q(π).
#
# For a `DiscretePOMDP` with finite action space and finite horizon, this
# enumerates all action sequences `π ∈ A^H` and evaluates the expected free
# energy of each:
#
#     G(π) = - Σ_τ E_q(o_τ|π)[log P̃(o)]                 (pragmatic)
#            - Σ_τ I[s_τ; o_τ | π]                       (epistemic)
#
# where the predicted state and observation distributions are propagated
# forward through the model: q(s_τ|π) = B[a_τ-1]·q(s_{τ-1}|π) and
# q(o_τ|π) = A · q(s_τ|π). The policy posterior is
#
#     q(π) = softmax(γ · -G(π))
#
# (no policy prior in v0.1; habit / E vector support is a follow-up).
#
# Sign convention: this implementation returns `G` in the convention where
# G is the quantity to MINIMIZE, so q(π) ∝ exp(-γG). The interface contract
# (`expected_free_energy = -(pragmatic + epistemic)`) is satisfied.

using Distributions: Categorical, probs
using LinearAlgebra: dot

"""
    EnumerativeEFE(; γ=16.0, horizon=1, use_pragmatic=true, use_epistemic=true)

Brute-force planner that enumerates all action sequences of length `horizon`
and computes EFE for each. Practical for `nactions(m)^horizon` ≤ a few
thousand; for larger spaces use `SophisticatedInference` (which prunes via
recursive Bellman) or a sampled / MCTS planner (later PRs).

Toggles:
- `use_pragmatic=false` disables the preference term — the agent becomes
  purely curiosity-driven (epistemic-only).
- `use_epistemic=false` disables the information-gain term — the agent
  becomes purely reward-seeking (no exploration).
"""
struct EnumerativeEFE{T<:Real} <: PolicyInference
    γ::T
    horizon::Int
    use_pragmatic::Bool
    use_epistemic::Bool
end

EnumerativeEFE(; γ::Real = 16.0,
                  horizon::Integer = 1,
                  use_pragmatic::Bool = true,
                  use_epistemic::Bool = true) =
    EnumerativeEFE(float(γ), Int(horizon), use_pragmatic, use_epistemic)

# --- The required PolicyInference interface ---

function pragmatic_value(planner::EnumerativeEFE,
                          m::DiscretePOMDP,
                          qs::Categorical,
                          π::AbstractVector{<:Integer})
    planner.use_pragmatic || return 0.0
    cur_qs = collect(probs(qs))
    total = 0.0
    @inbounds for a in π
        cur_qs = m.B[:, :, a] * cur_qs
        qo = m.A * cur_qs
        total += dot(qo, m.C)
    end
    return total
end

function epistemic_value(planner::EnumerativeEFE,
                          m::DiscretePOMDP,
                          qs::Categorical,
                          π::AbstractVector{<:Integer})
    planner.use_epistemic || return 0.0
    cur_qs = collect(probs(qs))
    total = 0.0
    @inbounds for a in π
        cur_qs = m.B[:, :, a] * cur_qs
        total += bayesian_surprise(m.A, cur_qs)
    end
    return total
end

function expected_free_energy(planner::EnumerativeEFE,
                                m::DiscretePOMDP,
                                qs::Categorical,
                                π::AbstractVector{<:Integer})
    return -(pragmatic_value(planner, m, qs, π) + epistemic_value(planner, m, qs, π))
end

function posterior_policies(planner::EnumerativeEFE,
                              m::DiscretePOMDP,
                              qs::Categorical,
                              policies::AbstractVector)
    G = [expected_free_energy(planner, m, qs, π) for π in policies]
    # q(π) ∝ exp(γ · -G(π))
    q_pi = softmax(.-G, planner.γ)
    return q_pi, G
end

"""
    enumerate_policies(planner::EnumerativeEFE, m::DiscretePOMDP, qs)

All action sequences of length `planner.horizon`. Returns a vector of
`Vector{Int}`s, each of length `horizon`.
"""
function enumerate_policies(planner::EnumerativeEFE, m::DiscretePOMDP, ::Categorical)
    n_actions = nactions(m)
    H = planner.horizon
    n_policies = n_actions ^ H
    policies = Vector{Vector{Int}}(undef, n_policies)
    for i in 1:n_policies
        π = Vector{Int}(undef, H)
        idx = i - 1
        for t in 1:H
            π[t] = (idx % n_actions) + 1
            idx ÷= n_actions
        end
        policies[i] = π
    end
    return policies
end


# ============================================================================
# Multi-factor EnumerativeEFE
# ============================================================================
#
# Each "policy" is a sequence of action vectors `Vector{Vector{Int}}`,
# where the inner `Vector{Int}` has length F (one entry per state factor).
# State predictions advance per-factor under the corresponding `B[f][:,:,a_f]`.
# Observation predictions and Bayesian surprise are computed against the
# joint `q(s) = ∏_f q_f(s_f)` at each step.

function pragmatic_value(planner::EnumerativeEFE,
                          m::MultiFactorDiscretePOMDP,
                          qs::Union{MultivariateDistribution, Categorical},
                          π::AbstractVector{<:AbstractVector{<:Integer}})
    planner.use_pragmatic || return 0.0
    F = length(m.D)
    factor_sizes = ntuple(f -> length(m.D[f]), F)
    cur_factors = _factor_probs_for_planning(qs, F)

    total = 0.0
    @inbounds for action_vec in π
        # Advance each factor
        for f in 1:F
            cur_factors[f] = m.B[f][:, :, action_vec[f]] * cur_factors[f]
        end
        # Predict observations per modality, accumulate dot(qo, C)
        for m_idx in eachindex(m.A)
            qo = _mf_predict_obs_modality(m.A[m_idx], cur_factors, factor_sizes)
            total += dot(qo, m.C[m_idx])
        end
    end
    return total
end

function epistemic_value(planner::EnumerativeEFE,
                          m::MultiFactorDiscretePOMDP,
                          qs::Union{MultivariateDistribution, Categorical},
                          π::AbstractVector{<:AbstractVector{<:Integer}})
    planner.use_epistemic || return 0.0
    F = length(m.D)
    factor_sizes = ntuple(f -> length(m.D[f]), F)
    cur_factors = _factor_probs_for_planning(qs, F)

    total = 0.0
    @inbounds for action_vec in π
        for f in 1:F
            cur_factors[f] = m.B[f][:, :, action_vec[f]] * cur_factors[f]
        end
        # I[s; o_m] per modality, summed
        for m_idx in eachindex(m.A)
            total += _mf_bayesian_surprise(m.A[m_idx], cur_factors, factor_sizes)
        end
    end
    return total
end

function expected_free_energy(planner::EnumerativeEFE,
                                m::MultiFactorDiscretePOMDP,
                                qs::Union{MultivariateDistribution, Categorical},
                                π::AbstractVector{<:AbstractVector{<:Integer}})
    return -(pragmatic_value(planner, m, qs, π) + epistemic_value(planner, m, qs, π))
end

function posterior_policies(planner::EnumerativeEFE,
                              m::MultiFactorDiscretePOMDP,
                              qs::Union{MultivariateDistribution, Categorical},
                              policies::AbstractVector)
    G = [expected_free_energy(planner, m, qs, π) for π in policies]
    q_pi = softmax(.-G, planner.γ)
    return q_pi, G
end

function enumerate_policies(planner::EnumerativeEFE,
                              m::MultiFactorDiscretePOMDP,
                              ::Union{MultivariateDistribution, Categorical})
    F = length(m.B)
    K_per_factor = ntuple(f -> size(m.B[f], 3), F)
    H = planner.horizon

    # Enumerate per-step action vectors
    action_vecs = [collect(t) for t in Iterators.product(ntuple(f -> 1:K_per_factor[f], F)...)]
    n_action_vecs = length(action_vecs)

    n_policies = n_action_vecs ^ H
    policies = Vector{Vector{Vector{Int}}}(undef, n_policies)
    for i in 1:n_policies
        π = Vector{Vector{Int}}(undef, H)
        idx = i - 1
        for t in 1:H
            π[t] = action_vecs[(idx % n_action_vecs) + 1]
            idx ÷= n_action_vecs
        end
        policies[i] = π
    end
    return policies
end


# --- Multi-factor helpers ---

# Extract per-factor probability vectors from a Distribution
function _factor_probs_for_planning(qs::Categorical, F::Int)
    F == 1 || throw(ArgumentError(
        "_factor_probs_for_planning: Categorical requires F=1 (got F=$F)"))
    return [collect(probs(qs))]
end
_factor_probs_for_planning(qs::MultivariateDistribution, F::Int) =
    [collect(probs(marginal(qs, f))) for f in 1:F]

# qo[o_m] = Σ_s A_m[o_m, s_1, ..., s_F] * ∏_f q_f(s_f)
function _mf_predict_obs_modality(A_m::AbstractArray,
                                    q_factors,
                                    factor_sizes::NTuple{F,Int}) where F
    n_obs = size(A_m, 1)
    qo = zeros(eltype(A_m), n_obs)
    @inbounds for ci in CartesianIndices(factor_sizes)
        weight = 1.0
        for f in 1:F
            weight *= q_factors[f][ci[f]]
        end
        for o in 1:n_obs
            qo[o] += weight * A_m[o, ci]
        end
    end
    return qo
end

# I[s; o_m] = H[q(o_m)] - E_{q(s)}[H[A_m(:|s)]]
function _mf_bayesian_surprise(A_m::AbstractArray,
                                  q_factors,
                                  factor_sizes::NTuple{F,Int}) where F
    qo = _mf_predict_obs_modality(A_m, q_factors, factor_sizes)
    H_qo = entropy(qo)

    H_qs_A = 0.0
    n_obs = size(A_m, 1)
    @inbounds for ci in CartesianIndices(factor_sizes)
        weight = 1.0
        for f in 1:F
            weight *= q_factors[f][ci[f]]
        end
        col_entropy = 0.0
        for o in 1:n_obs
            a = A_m[o, ci]
            if a > 0
                col_entropy -= a * log(a)
            end
        end
        H_qs_A += weight * col_entropy
    end
    return H_qo - H_qs_A
end
