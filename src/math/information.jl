# Information-theoretic primitives for categorical distributions.
#
# Conventions:
#   - All quantities use natural log (nats), matching Friston's papers.
#   - Inputs are non-negative vectors (probability vectors when the input is
#     a single distribution; weights / masses for expectations).
#   - The `0·log(0) = 0` convention is taken everywhere: zero entries
#     contribute zero, even though `log(0) = -∞`.
#   - Where a divergence is genuinely infinite (e.g. KL with `q[i] = 0` and
#     `p[i] > 0`), `Inf` is returned rather than thrown — keeps these as
#     well-defined functions on the closed simplex.

"""
    entropy(p)

Shannon entropy of a categorical distribution `p`, in nats:

    H[p] = -Σᵢ pᵢ log pᵢ

Treats zero entries via the convention `0·log(0) = 0`. Does not check that
`p` sums to 1: accepts unnormalized non-negative weights, returning the
"un-normalized entropy" in that case (occasionally useful for diagnostics
but usually a sign of a bug if you didn't intend it).

For a uniform distribution over `n` outcomes, `entropy(p) = log(n)`.
For a delta distribution, `entropy(p) = 0`.

# Examples
```jldoctest
julia> entropy([0.5, 0.5]) ≈ log(2)
true

julia> entropy([1.0, 0.0])
0.0

julia> entropy(fill(1/4, 4)) ≈ log(4)
true
```
"""
function entropy(p::AbstractVector{T}) where {T<:Real}
    R = float(T)
    H = zero(R)
    @inbounds for pᵢ in p
        if pᵢ > zero(pᵢ)
            H -= pᵢ * log(pᵢ)
        end
    end
    return H
end

"""
    kl_divergence(p, q)

Kullback–Leibler divergence `D_KL[p ‖ q]` for categorical distributions, in nats:

    D_KL[p ‖ q] = Σᵢ pᵢ · (log pᵢ - log qᵢ)

Returns `Inf` if there exists `i` with `pᵢ > 0` and `qᵢ = 0` (the divergence
diverges). Returns 0 when `pᵢ = 0` regardless of `qᵢ`, by the convention
`0·log(0/qᵢ) = 0`.

By Gibbs' inequality, `kl_divergence(p, q) ≥ 0`, with equality iff `p == q`
(on the support of `p`). KL divergence is asymmetric:
`kl_divergence(p, q) ≠ kl_divergence(q, p)` in general.

# Examples
```jldoctest
julia> kl_divergence([1.0, 0.0], [0.5, 0.5]) ≈ log(2)
true

julia> kl_divergence([0.5, 0.5], [0.5, 0.5])
0.0

julia> isinf(kl_divergence([1.0, 0.0], [0.0, 1.0]))
true
```
"""
function kl_divergence(p::AbstractVector{T},
                       q::AbstractVector{S}) where {T<:Real,S<:Real}
    length(p) == length(q) ||
        throw(DimensionMismatch("kl_divergence: p has length $(length(p)), q has length $(length(q))"))
    R = float(promote_type(T, S))
    d = zero(R)
    @inbounds for i in eachindex(p)
        pᵢ, qᵢ = p[i], q[i]
        if pᵢ > zero(pᵢ)
            qᵢ <= zero(qᵢ) && return R(Inf)
            d += pᵢ * (log(pᵢ) - log(qᵢ))
        end
    end
    return d
end

"""
    cross_entropy(p, q)

Cross-entropy `H(p, q) = -Σᵢ pᵢ log qᵢ`, in nats.

Satisfies the identity:

    cross_entropy(p, q) = entropy(p) + kl_divergence(p, q)

(verified as a property test). Returns `Inf` if there exists `i` with
`pᵢ > 0` and `qᵢ = 0`.

# Examples
```jldoctest
julia> cross_entropy([0.5, 0.5], [0.5, 0.5]) ≈ log(2)
true

julia> p, q = [0.7, 0.3], [0.4, 0.6];

julia> cross_entropy(p, q) ≈ entropy(p) + kl_divergence(p, q)
true
```
"""
function cross_entropy(p::AbstractVector{T},
                       q::AbstractVector{S}) where {T<:Real,S<:Real}
    length(p) == length(q) ||
        throw(DimensionMismatch("cross_entropy: p has length $(length(p)), q has length $(length(q))"))
    R = float(promote_type(T, S))
    h = zero(R)
    @inbounds for i in eachindex(p)
        pᵢ, qᵢ = p[i], q[i]
        if pᵢ > zero(pᵢ)
            qᵢ <= zero(qᵢ) && return R(Inf)
            h -= pᵢ * log(qᵢ)
        end
    end
    return h
end

"""
    bayesian_surprise(A, qs)

Bayesian surprise (mutual information between hidden state and outcome) under
likelihood matrix `A` and state belief `qs`:

    I[s; o] = H[q(o)] - E_qs[H[A(·|s)]]

where `A[o, s] = P(o | s)` is column-stochastic, and
`q(o) = Σ_s A[:, s] · qs[s]` is the marginal observation distribution.

Equivalently, this is the *expected information gain* about the hidden state
that a single observation provides:

    I[s; o] = E_q(o)[D_KL[q(s | o) ‖ qs]]

In active inference, the epistemic component of expected free energy under
a candidate policy is exactly this quantity, evaluated at the predicted
state distribution at the future timestep.

`A` is `[numObs × numStates]`. Returns a non-negative scalar.

# Examples
```jldoctest
julia> A = [1.0 0.0; 0.0 1.0];   # observation perfectly identifies state

julia> bayesian_surprise(A, [0.5, 0.5]) ≈ log(2)
true

julia> A = [0.5 0.5; 0.5 0.5];   # observation independent of state

julia> abs(bayesian_surprise(A, [0.5, 0.5])) < 1e-12
true
```
"""
function bayesian_surprise(A::AbstractMatrix{T},
                           qs::AbstractVector{S}) where {T<:Real,S<:Real}
    size(A, 2) == length(qs) ||
        throw(DimensionMismatch("bayesian_surprise: A has $(size(A,2)) columns, qs has length $(length(qs))"))
    R = float(promote_type(T, S))

    # Marginal observation distribution: q(o) = Σ_s A[o,s] · qs[s]
    qo = A * qs
    H_qo = entropy(qo)

    # Expected entropy of the conditional likelihood: E_qs[H[A(·|s)]]
    H_qsA = zero(R)
    @inbounds for s in eachindex(qs)
        qs_s = qs[s]
        qs_s > zero(qs_s) || continue
        col_entropy = zero(R)
        for o in axes(A, 1)
            a = A[o, s]
            if a > zero(a)
                col_entropy -= a * log(a)
            end
        end
        H_qsA += qs_s * col_entropy
    end

    return H_qo - H_qsA
end
