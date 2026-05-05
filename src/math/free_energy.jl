# Variational free energy F[q] for a single-step categorical model.
#
# The variational free energy of a posterior `q` over hidden states `s`,
# given an observation `o`, is
#
#   F[q] = E_q[log q(s) - log P(o, s)]
#
# Two equivalent decompositions are exposed here:
#
#   (A)   F = (-accuracy) + complexity
#           = -E_q[log P(o|s)] + D_KL[q(s) ‖ P(s)]                  (computed)
#
#   (B)   F = -log P(o) + D_KL[q(s) ‖ P(s|o)]                       (Gibbs form)
#
# (A) is what we evaluate. (B) proves that F is an upper bound on the
# negative log-evidence (-log P(o), the agent's "surprise"), with equality
# iff q matches the exact posterior P(s|o). Perception minimizes F.
#
# References:
#   Friston (2010) "The free-energy principle: a unified brain theory?"
#     Nature Reviews Neuroscience 11: 127–138, Eq. 2.
#   Buckley, Kim, McGregor, Seth (2017) "The free energy principle for
#     action and perception: A mathematical review."
#     J Math Psychol 81: 55–79, §2.

"""
    variational_free_energy(q, log_likelihood, log_prior)

Variational free energy `F[q]` of a categorical posterior `q` under a
single-step model specified by per-state log-likelihood and log-prior:

    F[q] = -⟨log P(o|s)⟩_q + D_KL[q(s) ‖ P(s)]
         = Σ_s q[s] · (log q[s] - log P(s) - log P(o|s))

`log_likelihood[s]` is `log P(o|s)` for the *observed* outcome `o`.
`log_prior[s]` is `log P(s)`.

By Gibbs' inequality, `F[q] ≥ -log P(o)`, with equality iff
`q[s] = P(s|o) ∝ P(o|s)·P(s)`. Perception minimizes `F` over `q`.

See also [`accuracy`](@ref), [`complexity`](@ref).
"""
function variational_free_energy(q::AbstractVector{Tq},
                                 log_likelihood::AbstractVector{Tℓ},
                                 log_prior::AbstractVector{Tπ}) where {Tq<:Real,Tℓ<:Real,Tπ<:Real}
    length(q) == length(log_likelihood) == length(log_prior) ||
        throw(DimensionMismatch("variational_free_energy: q, log_likelihood, log_prior must have matching length (got $(length(q)), $(length(log_likelihood)), $(length(log_prior)))"))
    R = float(promote_type(Tq, Tℓ, Tπ))
    F = zero(R)
    @inbounds for i in eachindex(q)
        qᵢ = q[i]
        if qᵢ > zero(qᵢ)
            F += qᵢ * (log(qᵢ) - log_prior[i] - log_likelihood[i])
        end
    end
    return F
end

"""
    accuracy(q, log_likelihood)

The accuracy term of the free-energy decomposition:

    accuracy[q] = ⟨log P(o|s)⟩_q = Σ_s q[s] · log P(o|s)

Higher accuracy = the posterior `q` concentrates on states that explain the
observation well. The accuracy enters `F` with a *minus* sign:
`F = -accuracy + complexity`.
"""
function accuracy(q::AbstractVector{Tq},
                  log_likelihood::AbstractVector{Tℓ}) where {Tq<:Real,Tℓ<:Real}
    length(q) == length(log_likelihood) ||
        throw(DimensionMismatch("accuracy: q has length $(length(q)), log_likelihood has length $(length(log_likelihood))"))
    R = float(promote_type(Tq, Tℓ))
    a = zero(R)
    @inbounds for i in eachindex(q)
        qᵢ = q[i]
        if qᵢ > zero(qᵢ)
            a += qᵢ * log_likelihood[i]
        end
    end
    return a
end

"""
    complexity(q, log_prior)

The complexity term of the free-energy decomposition:

    complexity[q] = D_KL[q(s) ‖ P(s)] = Σ_s q[s] · (log q[s] - log P(s))

Higher complexity = the posterior departs further from the prior. Penalizes
overconfident or off-prior posteriors. Mathematically equal to
`kl_divergence(q, exp.(log_prior))` but computed directly from `log_prior`
for numerical stability when prior probabilities are tiny.
"""
function complexity(q::AbstractVector{Tq},
                    log_prior::AbstractVector{Tπ}) where {Tq<:Real,Tπ<:Real}
    length(q) == length(log_prior) ||
        throw(DimensionMismatch("complexity: q has length $(length(q)), log_prior has length $(length(log_prior))"))
    R = float(promote_type(Tq, Tπ))
    c = zero(R)
    @inbounds for i in eachindex(q)
        qᵢ = q[i]
        if qᵢ > zero(qᵢ)
            c += qᵢ * (log(qᵢ) - log_prior[i])
        end
    end
    return c
end
