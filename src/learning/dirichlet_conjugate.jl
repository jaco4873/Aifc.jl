# `DirichletConjugate` — parameter learning for `DiscretePOMDP` via the
# Dirichlet conjugacy of the Categorical likelihood.
#
# The agent maintains Dirichlet hyperparameters `pA`, `pB`, `pD` over the
# generative model's parameters. After observing a trajectory of states and
# actions, the posterior is updated by adding outer-product counts:
#
#   pA[o, s]            ←  pA[o, s]    + α · q(s) · 𝟙[o_t = o]              (per τ)
#   pB[s', s, a]        ←  pB[s', s, a] + α · q(s' from prev) · q(s) · 𝟙[a_t = a]
#   pD[s]               ←  pD[s]       + α · q(s_1)
#
# where `α` is the learning rate. This is exactly the conjugate update for
# Dirichlet over Categorical likelihoods (Heins et al. 2022 pymdp §3.5; Da
# Costa et al. 2020 Eq. 21). Forgetting is implemented by multiplying the
# accumulated counts by `fr` (fr = 1.0 means no forgetting).
#
# A subsequent agent step uses the updated `pA/pB/pD` to derive the
# digamma-corrected effective likelihoods (Friston, Da Costa et al. 2020):
#
#   log Ã[o, s]  =  ψ(pA[o, s])  -  ψ(Σ_o pA[o, s])     (column-wise digamma)
#   Ã[o, s]      =  exp(log Ã[o, s])  / Σ_o exp(log Ã[o, s])     (per-column normalize)
#
# This is what makes parameter uncertainty propagate into belief updates:
# under-sampled columns of A are FLATTER than naive `pA / Σ pA` would give,
# coupling parameter uncertainty to the agent's epistemic value.

using Distributions: Categorical, probs

"""
    DirichletConjugate(; lr_pA=1.0, fr_pA=1.0, lr_pB=1.0, fr_pB=1.0,
                          lr_pD=1.0, fr_pD=1.0,
                          learn_pA=true, learn_pB=true, learn_pD=true,
                          use_effective_A=true)

Online conjugate update for the Dirichlet hyperparameters of a
`DiscretePOMDP`. Per-channel learning rate `lr_pX` and forgetting rate
`fr_pX` (`fr=1` = no forgetting). Booleans `learn_pX` toggle each channel.

When `use_effective_A=true` (default), the returned model has its `A` matrix
replaced by the digamma-corrected effective likelihood (`expectedLogA` →
column-normalized exponential). This is what couples parameter uncertainty
to belief inference. Set to `false` to use the simple `pA / Σ pA` mean.
"""
struct DirichletConjugate{T<:Real} <: Inference
    lr_pA::T
    fr_pA::T
    lr_pB::T
    fr_pB::T
    lr_pD::T
    fr_pD::T
    learn_pA::Bool
    learn_pB::Bool
    learn_pD::Bool
    use_effective_A::Bool
end

function DirichletConjugate(; lr_pA::Real = 1.0, fr_pA::Real = 1.0,
                              lr_pB::Real = 1.0, fr_pB::Real = 1.0,
                              lr_pD::Real = 1.0, fr_pD::Real = 1.0,
                              learn_pA::Bool = true,
                              learn_pB::Bool = true,
                              learn_pD::Bool = true,
                              use_effective_A::Bool = true)
    T = float(promote_type(typeof(lr_pA), typeof(fr_pA),
                            typeof(lr_pB), typeof(fr_pB),
                            typeof(lr_pD), typeof(fr_pD)))
    return DirichletConjugate{T}(T(lr_pA), T(fr_pA),
                                  T(lr_pB), T(fr_pB),
                                  T(lr_pD), T(fr_pD),
                                  learn_pA, learn_pB, learn_pD, use_effective_A)
end

supports_parameters(::DirichletConjugate) = true

function infer_parameters(alg::DirichletConjugate,
                          m::DiscretePOMDP,
                          history::AgentHistory)
    isempty(history) && return m

    # The Dirichlet conjugate update is a per-step recurrence:
    #   pA[t+1] = fr · pA[t] + lr · q(s_t) ⊗ onehot(o_t)
    # `Agent.step!` calls this once per step, so we apply ONLY the most
    # recent step's contribution to the *current* pA. (Replaying the full
    # history each call would multi-count earlier steps and produce
    # garbage — see `test_dirichlet_conjugate_incremental.jl`.)

    new_pA = m.pA
    new_pB = m.pB
    new_pD = m.pD
    new_A  = m.A

    last_idx  = length(history)
    cur_step  = history.steps[last_idx]
    prev_step = last_idx > 1 ? history.steps[last_idx - 1] : nothing

    # --- pA: pA[t+1] = fr · pA[t] + lr · q(s_t) ⊗ onehot(o_t) ---
    if alg.learn_pA && m.pA !== nothing
        new_pA = alg.fr_pA .* m.pA
        o  = cur_step.observation
        qs = probs(cur_step.belief)
        for s in eachindex(qs)
            new_pA[o, s] += alg.lr_pA * qs[s]
        end
        new_A = alg.use_effective_A ? _effective_A(new_pA) : _dirichlet_mean_A(new_pA)
    end

    # --- pB: pB[t+1] = fr · pB[t] + lr · q(s_t) ⊗ q(s_{t-1}) at action a_{t-1}.
    #         No transition data on the very first step — apply forgetting only. ---
    if alg.learn_pB && m.pB !== nothing
        new_pB = alg.fr_pB .* m.pB
        if prev_step !== nothing
            prev_qs = probs(prev_step.belief)
            cur_qs  = probs(cur_step.belief)
            a       = prev_step.action
            for s in eachindex(prev_qs), s′ in eachindex(cur_qs)
                new_pB[s′, s, a] += alg.lr_pB * cur_qs[s′] * prev_qs[s]
            end
        end
    end

    # --- pD: prior over the *initial* state. Updated only on step 1; later
    #         calls just apply forgetting. ---
    if alg.learn_pD && m.pD !== nothing
        new_pD = alg.fr_pD .* m.pD
        if last_idx == 1
            qs1 = probs(cur_step.belief)
            @. new_pD += alg.lr_pD * qs1
        end
    end

    # B is NOT regenerated from pB by default (B and pB diverge as learning
    # progresses; the agent uses the *current* B for prediction, while pB
    # accumulates parameter uncertainty separately).
    return DiscretePOMDP{eltype(new_A)}(
        new_A, m.B, m.C, m.D, m.E, new_pA, new_pB, new_pD, m.goal)
end

# Mean of a Dirichlet over A's columns: pA / Σ_o pA (column-wise)
function _dirichlet_mean_A(pA::AbstractMatrix{T}) where T<:Real
    out = similar(pA, float(T))
    @inbounds for s in axes(pA, 2)
        col_sum = zero(float(T))
        for o in axes(pA, 1)
            col_sum += pA[o, s]
        end
        z = col_sum > zero(col_sum) ? col_sum : one(col_sum)
        for o in axes(pA, 1)
            out[o, s] = pA[o, s] / z
        end
    end
    return out
end

# Digamma-corrected effective likelihood:
#   log Ã[o, s]  =  ψ(pA[o, s])  -  ψ(Σ_o pA[o, s])
#   Ã[o, s]      =  exp(log Ã[o, s]) / Σ_o exp(log Ã[o, s])
#
# Yields a flatter likelihood under high parameter uncertainty (small pA),
# coupling parameter uncertainty to belief inference (Friston/Da Costa 2020,
# Heins et al. 2022 pymdp).
function _effective_A(pA::AbstractMatrix{T}) where T<:Real
    out = similar(pA, float(T))
    @inbounds for s in axes(pA, 2)
        col_sum = zero(float(T))
        for o in axes(pA, 1)
            col_sum += pA[o, s]
        end
        ψ_sum = SpecialFunctions_digamma(max(col_sum, eps(float(T))))

        # Compute log Ã[:, s], then column-wise softmax for stability
        log_col = similar(out[:, s])
        for o in axes(pA, 1)
            log_col[o] = SpecialFunctions_digamma(max(pA[o, s], eps(float(T)))) - ψ_sum
        end
        m = maximum(log_col)
        z = zero(float(T))
        for o in axes(pA, 1)
            out[o, s] = exp(log_col[o] - m)
            z += out[o, s]
        end
        if z > zero(z)
            for o in axes(pA, 1)
                out[o, s] /= z
            end
        end
    end
    return out
end

# Implementation of digamma using Stirling-series asymptotic + recurrence
# down to the asymptotic regime. Standalone (no SpecialFunctions dep) for
# v0.1 to keep the dependency graph small. Same algorithm used by the TS
# primer in `discrete.ts` (validated to ~10 dp at typical magnitudes).
#
# Mathematically: ψ(x) ≈ log(x) - 1/(2x) - Σ B_{2k} / (2k x^{2k}) ; we use
# the first few Bernoulli numbers and recur the input up to x ≥ 6 first.
function SpecialFunctions_digamma(x::Real)
    result = zero(float(typeof(x)))
    while x < 6
        result -= 1 / x
        x += 1
    end
    result += log(x) - 1 / (2 * x)
    x2 = 1 / (x * x)
    result -= x2 * (1 / 12 - x2 * (1 / 120 - x2 / 252))
    return result
end


# ============================================================================
# Multi-factor DirichletConjugate
# ============================================================================
#
# Per-factor / per-modality conjugate updates:
#   pA[m][o, s_1, …, s_F]  +=  lr · 𝟙[o = obs_τ[m]] · ∏_f q_f(s_f, τ)
#   pB[f][s', s, a_f]      +=  lr · q_f(s'_τ) · q_f(s_{τ-1})
#   pD[f][s_f]             +=  lr · q_f(s_1)
#
# `use_effective_A` flag: only meaningful for single-factor models in the
# current implementation; for multi-factor models we just write the
# Dirichlet mean (column-normalized) into A. The full digamma-corrected
# effective likelihood for multi-factor A involves contractions and is a
# follow-up.

function infer_parameters(alg::DirichletConjugate,
                          m::MultiFactorDiscretePOMDP,
                          history::AgentHistory)
    isempty(history) && return m

    # See the single-factor version above for the per-step recurrence
    # rationale: this consumes only the most recent step (and the
    # second-to-last for pB transitions).

    new_pA = m.pA
    new_pB = m.pB
    new_pD = m.pD
    new_A  = m.A

    F = length(m.D)
    M = length(m.A)
    factor_sizes = ntuple(f -> length(m.D[f]), F)

    last_idx  = length(history)
    cur_step  = history.steps[last_idx]
    prev_step = last_idx > 1 ? history.steps[last_idx - 1] : nothing

    # --- pA update ---
    if alg.learn_pA && m.pA !== nothing
        new_pA = [alg.fr_pA .* m.pA[m_idx] for m_idx in 1:M]
        obs = cur_step.observation
        qs  = _factor_probs_for_planning(cur_step.belief, F)
        for m_idx in 1:M
            o_m = obs[m_idx]
            # Outer product of factor marginals, scaled by lr; add at slice [o_m, :, :, …]
            for ci in CartesianIndices(factor_sizes)
                weight = 1.0
                for f in 1:F
                    weight *= qs[f][ci[f]]
                end
                new_pA[m_idx][o_m, ci] += alg.lr_pA * weight
            end
        end
        new_A = [_mf_dirichlet_mean_A(new_pA[m_idx]) for m_idx in 1:M]
    end

    # --- pB update ---
    if alg.learn_pB && m.pB !== nothing
        new_pB = [alg.fr_pB .* m.pB[f] for f in 1:F]
        if prev_step !== nothing
            prev_qs = _factor_probs_for_planning(prev_step.belief, F)
            cur_qs  = _factor_probs_for_planning(cur_step.belief,   F)
            a_vec   = prev_step.action
            for f in 1:F
                a_f = a_vec[f]
                for s in eachindex(prev_qs[f]), s′ in eachindex(cur_qs[f])
                    new_pB[f][s′, s, a_f] += alg.lr_pB * cur_qs[f][s′] * prev_qs[f][s]
                end
            end
        end
    end

    # --- pD update: only on step 1 ---
    if alg.learn_pD && m.pD !== nothing
        new_pD = [alg.fr_pD .* m.pD[f] for f in 1:F]
        if last_idx == 1
            qs1 = _factor_probs_for_planning(cur_step.belief, F)
            for f in 1:F
                @. new_pD[f] += alg.lr_pD * qs1[f]
            end
        end
    end

    return MultiFactorDiscretePOMDP{eltype(new_A[1])}(
        new_A, m.B, m.C, m.D, m.E, new_pA, new_pB, new_pD, m.goal)
end

# Column-normalized Dirichlet mean: pA / Σ_o pA, per (s_1, …, s_F) column.
function _mf_dirichlet_mean_A(pA_m::AbstractArray{T}) where {T<:Real}
    out = similar(pA_m, float(T))
    factor_dims = size(pA_m)[2:end]
    @inbounds for ci in CartesianIndices(factor_dims)
        col_sum = zero(float(T))
        for o in axes(pA_m, 1)
            col_sum += pA_m[o, ci]
        end
        z = col_sum > zero(col_sum) ? col_sum : one(col_sum)
        for o in axes(pA_m, 1)
            out[o, ci] = pA_m[o, ci] / z
        end
    end
    return out
end

# Re-use _factor_probs_for_planning from src/planning/enumerative.jl.
# It's already exported into the Aifc module namespace.
