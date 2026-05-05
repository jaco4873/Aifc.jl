# `FixedPointIteration` — variational state inference for discrete models.
#
# For a single-factor `DiscretePOMDP`, the variational free energy at q has
# a closed-form minimizer:
#
#   q*(s) ∝ P(o | s) · P(s) = exp(log A[o, s] + log prior[s])
#
# This converges in a single fixed-point iteration. The algorithm stops as
# soon as the change in F drops below `dF_tol`, with `num_iter` as a safety
# cap. For more complex factorizations (mean-field over multiple state
# factors) the same algorithm requires multiple iterations as factors
# alternately update against each other's expectations — that path is
# implemented when multi-factor support lands.
#
# References:
#   Friston (2017) "Active inference: a process theory" Neural Comp 29: 1.
#   Heins et al. (2022) pymdp toolbox, JOSS 7(73): 4098, §3.

using Distributions: Categorical, probs

"""
    FixedPointIteration(; num_iter=10, dF_tol=1e-3)

Mean-field variational state inference for `DiscretePOMDP`. Iterates the
fixed-point equation `q(s) ∝ exp(log A[o, s] + log prior[s])` until the
change in free energy is below `dF_tol`, with at most `num_iter` iterations.

For single-factor POMDPs (the case currently supported) the closed form is
exact in one iteration; the iteration count and tolerance only matter when
multi-factor mean-field is added.
"""
struct FixedPointIteration <: Inference
    num_iter::Int
    dF_tol::Float64
end

FixedPointIteration(; num_iter::Int=10, dF_tol::Real=1e-3) =
    FixedPointIteration(num_iter, Float64(dF_tol))

supports_states(::FixedPointIteration) = true

function infer_states(alg::FixedPointIteration,
                      m::DiscretePOMDP,
                      prior::Categorical,
                      observation::Integer)
    1 <= observation <= size(m.A, 1) ||
        throw(BoundsError("infer_states: observation $observation out of range [1, $(size(m.A, 1))]"))

    log_lik = _capped_log.(@view m.A[observation, :])
    log_prior = _capped_log.(probs(prior))

    # Single-factor closed form: q(s) ∝ exp(log A[o, s] + log prior[s])
    q_logp = log_lik .+ log_prior
    q = softmax(q_logp)
    return Categorical(q)
end

# Free energy F[q] = -⟨log P(o|s)⟩_q + KL[q ‖ prior]
function free_energy(alg::FixedPointIteration,
                     m::DiscretePOMDP,
                     prior::Categorical,
                     observation::Integer,
                     q::Categorical)
    log_lik = _capped_log.(@view m.A[observation, :])
    log_prior = _capped_log.(probs(prior))
    return variational_free_energy(probs(q), log_lik, log_prior)
end

# Capped log: avoids -Inf when probabilities are exactly zero. Returns the
# log of `max(x, eps(typeof(x)))`. Conventionally used in active-inference
# implementations to keep gradients well-defined at the simplex boundary.
_capped_log(x::T) where {T<:Real} = log(max(float(x), eps(float(T))))


# ============================================================================
# Multi-factor mean-field FPI
# ============================================================================
#
# For a multi-factor model with state factors of sizes (N₁, …, N_F) and
# observation modalities of sizes (O₁, …, O_M), the mean-field posterior
# factorizes as q(s) = ∏_f q_f(s_f). Each q_f is updated by:
#
#   q_f(s_f) ∝ exp(E_{q(s_-f)}[Σ_m log A[m][o_m, s_1, …, s_F]] + log D_f(s_f))
#
# We precompute the joint-state log-likelihood tensor
#   LL[s_1, …, s_F] = Σ_m log A[m][o_m, s_1, …, s_F]
# then iterate the per-factor updates until F decreases by less than `dF_tol`
# or `num_iter` is reached.

function infer_states(alg::FixedPointIteration,
                       m::MultiFactorDiscretePOMDP,
                       prior,
                       observation::AbstractVector{<:Integer})
    F = length(m.D)
    M = length(m.A)
    length(observation) == M || throw(DimensionMismatch(
        "infer_states: |observation|=$(length(observation)), expected $M (number of modalities)"))

    factor_sizes = ntuple(f -> length(m.D[f]), F)

    # Per-factor priors (in log domain)
    log_priors = _factor_log_priors(prior, F)

    # Initial q = prior (factorized)
    q_factors = [exp.(log_priors[f]) for f in 1:F]
    # Renormalize each (since _capped_log clips small entries)
    for f in 1:F
        s = sum(q_factors[f])
        q_factors[f] ./= s
    end

    # Joint log-likelihood tensor at the observed outcomes
    LL = _build_joint_loglik(m, observation, factor_sizes)

    # Mean-field iterations
    if F == 1
        # Closed form
        q_factors[1] = softmax(vec(LL) .+ log_priors[1])
    else
        prev_F = Inf
        for _ in 1:alg.num_iter
            for f in 1:F
                E_log_lik = _expected_logp_marginalized(LL, q_factors, f)
                q_factors[f] = softmax(E_log_lik .+ log_priors[f])
            end
            curr_F = _vfe_multi_factor(q_factors, log_priors, LL)
            abs(prev_F - curr_F) < alg.dF_tol && break
            prev_F = curr_F
        end
    end

    if F == 1
        return Categorical(q_factors[1])
    else
        return product_belief([Categorical(q_f) for q_f in q_factors])
    end
end

function free_energy(alg::FixedPointIteration,
                     m::MultiFactorDiscretePOMDP,
                     prior,
                     observation::AbstractVector{<:Integer},
                     q)
    F = length(m.D)
    M = length(m.A)
    factor_sizes = ntuple(f -> length(m.D[f]), F)

    LL = _build_joint_loglik(m, observation, factor_sizes)
    log_priors = _factor_log_priors(prior, F)
    q_factors = _factor_probs(q, F)

    return _vfe_multi_factor(q_factors, log_priors, LL)
end

# --- Multi-factor helpers ---

# Extract per-factor log priors from a Distribution (Categorical or multivariate)
function _factor_log_priors(prior, F::Int)
    if prior isa Categorical
        F == 1 || throw(ArgumentError(
            "_factor_log_priors: Categorical prior requires F=1 (got F=$F)"))
        return [_capped_log.(probs(prior))]
    else
        return [_capped_log.(probs(marginal(prior, f))) for f in 1:F]
    end
end

# Extract per-factor probability vectors
function _factor_probs(q, F::Int)
    if q isa Categorical
        F == 1 || throw(ArgumentError(
            "_factor_probs: Categorical q requires F=1 (got F=$F)"))
        return [collect(probs(q))]
    else
        return [collect(probs(marginal(q, f))) for f in 1:F]
    end
end

# Build the joint-state log-likelihood tensor LL[s_1, …, s_F] = Σ_m log A[m][o_m, s_1, …, s_F]
function _build_joint_loglik(m::MultiFactorDiscretePOMDP,
                              observation::AbstractVector{<:Integer},
                              factor_sizes::NTuple{F,Int}) where F
    LL = zeros(Float64, factor_sizes...)
    for m_idx in eachindex(m.A)
        A_m = m.A[m_idx]
        o_m = observation[m_idx]
        # Iterate over all factor states
        for ci in CartesianIndices(factor_sizes)
            LL[ci] += _capped_log(A_m[o_m, ci])
        end
    end
    return LL
end

# E_{q(s_-f)}[LL[s_1, …, s_F]] for factor f.
# Returns a vector of length factor_sizes[f].
function _expected_logp_marginalized(LL::AbstractArray{T,N},
                                       q_factors::AbstractVector,
                                       f::Int) where {T<:Real, N}
    factor_sizes = size(LL)
    result = zeros(T, factor_sizes[f])
    @inbounds for ci in CartesianIndices(LL)
        s_f = ci[f]
        weight = one(T)
        for g in 1:N
            g == f && continue
            weight *= q_factors[g][ci[g]]
        end
        result[s_f] += weight * LL[ci]
    end
    return result
end

# Mean-field variational free energy:
#   F = -E_q[LL] + Σ_f KL[q_f ‖ prior_f]
function _vfe_multi_factor(q_factors::AbstractVector,
                            log_priors::AbstractVector,
                            LL::AbstractArray)
    F = length(q_factors)
    # Accuracy term: E_q[LL] = Σ_full ∏_f q_f * LL
    accuracy = 0.0
    @inbounds for ci in CartesianIndices(LL)
        weight = 1.0
        for f in 1:F
            weight *= q_factors[f][ci[f]]
        end
        accuracy += weight * LL[ci]
    end
    # Complexity term: Σ_f KL[q_f ‖ prior_f]
    complexity = 0.0
    @inbounds for f in 1:F
        for s_f in eachindex(q_factors[f])
            qfs = q_factors[f][s_f]
            if qfs > zero(qfs)
                complexity += qfs * (log(qfs) - log_priors[f][s_f])
            end
        end
    end
    return -accuracy + complexity
end
