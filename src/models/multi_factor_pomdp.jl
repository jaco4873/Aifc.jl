# `MultiFactorDiscretePOMDP` — multi-factor / multi-modality A/B/C/D/E.
#
# State space factorizes into `F` factors of sizes `(N₁, …, N_F)`. Observation
# space factorizes into `M` modalities of sizes `(O₁, …, O_M)`. Each state
# factor has its own (potentially trivial) action set with `K_f` actions.
#
# Tensor layout:
#
#   A : Vector{Array{T}}, length M
#       A[m]  has shape  (O_m, N₁, N₂, …, N_F)
#       A[m][o, s₁, …, s_F] = P(o_m | s₁, …, s_F)
#       Each "column" (varying o for fixed s₁, …, s_F) sums to 1.
#
#   B : Vector{Array{T,3}}, length F
#       B[f]  has shape  (N_f, N_f, K_f)
#       B[f][s', s, a] = P(s'_f | s_f, a_f)
#       Each "column" (varying s' for fixed s, a) sums to 1. Factor f's
#       transition is independent of the other factors' transitions.
#
#   C : Vector{Vector{T}}, length M;  C[m] is the per-modality log preference.
#   D : Vector{Vector{T}}, length F;  D[f] is the prior over state factor f.
#
# `DiscretePOMDP` (single-factor) and `MultiFactorDiscretePOMDP` are separate
# types: the algorithms (FPI, EnumerativeEFE, DirichletConjugate) dispatch on
# the type, picking single-factor fast paths or full mean-field iteration as
# appropriate. The cleaner alternative — one unified type with `F=1` as a
# special case — would have required cascade-updating dozens of tests; the
# two-type design isolates the multi-factor work behind a clean type
# boundary. Long-term these may be unified.

using Distributions: Categorical, Distribution, MultivariateDistribution
using Random: Xoshiro

"""
    MultiFactorDiscretePOMDP(A, B, C, D; E=nothing, pA=nothing, pB=nothing, pD=nothing,
                                          goal=nothing, check=true, atol=1e-8)

Multi-factor / multi-modality discrete POMDP. See file header for the
tensor-layout convention.

# Arguments

- `A::Vector{Array{T}}` — one tensor per modality, shape
  `(num_obs[m], factor_sizes...)`
- `B::Vector{Array{T,3}}` — one tensor per state factor, shape
  `(num_states[f], num_states[f], num_actions[f])`
- `C::Vector{Vector{T}}` — log preferences per modality
- `D::Vector{Vector{T}}` — prior per state factor
- `E::Vector` — optional habit / policy prior
- `pA, pB, pD` — optional Dirichlet hyperparameter tensors of the same
  shapes as `A, B, D`
- `goal::Vector{Vector{T}}` — optional per-factor goal-state prior for
  inductive inference
- `check::Bool` — validate stochasticity + shape match (default `true`)
- `atol::Real` — tolerance for stochasticity check (default `1e-8`)
"""
struct MultiFactorDiscretePOMDP{T<:Real} <: GenerativeModel
    A::Vector{Array{T}}
    B::Vector{Array{T,3}}
    C::Vector{Vector{T}}
    D::Vector{Vector{T}}
    E::Union{Nothing, Vector{T}}
    pA::Union{Nothing, Vector{Array{T}}}
    pB::Union{Nothing, Vector{Array{T,3}}}
    pD::Union{Nothing, Vector{Vector{T}}}
    goal::Union{Nothing, Vector{Vector{T}}}
end

function MultiFactorDiscretePOMDP(A::AbstractVector{<:AbstractArray},
                                    B::AbstractVector{<:AbstractArray},
                                    C::AbstractVector{<:AbstractVector},
                                    D::AbstractVector{<:AbstractVector};
                                    E::Union{Nothing, AbstractVector} = nothing,
                                    pA::Union{Nothing, AbstractVector{<:AbstractArray}} = nothing,
                                    pB::Union{Nothing, AbstractVector{<:AbstractArray}} = nothing,
                                    pD::Union{Nothing, AbstractVector{<:AbstractVector}} = nothing,
                                    goal::Union{Nothing, AbstractVector{<:AbstractVector}} = nothing,
                                    check::Bool = true,
                                    atol::Real = 1e-8)
    F = length(D)
    M = length(C)
    F == length(B) || throw(DimensionMismatch(
        "MultiFactorDiscretePOMDP: |B|=$(length(B)) ≠ |D|=$F (number of state factors)"))
    M == length(A) || throw(DimensionMismatch(
        "MultiFactorDiscretePOMDP: |A|=$(length(A)) ≠ |C|=$M (number of modalities)"))

    factor_sizes = ntuple(f -> length(D[f]), F)
    obs_sizes    = ntuple(m -> length(C[m]), M)

    _mf_validate_A(A, factor_sizes, obs_sizes; check=check, atol=atol)
    _mf_validate_B(B, factor_sizes; check=check, atol=atol)
    _mf_validate_D(D; check=check, atol=atol)
    goal === nothing || _mf_validate_goal(goal, factor_sizes; check=check, atol=atol)
    pA   === nothing || _mf_validate_shape_match(pA, A, "pA")
    pB   === nothing || _mf_validate_shape_match(pB, B, "pB")
    pD   === nothing || _mf_validate_shape_match(pD, D, "pD")

    T = _mf_promote_eltype(A, B, C, D, E, pA, pB, pD, goal)

    return MultiFactorDiscretePOMDP{T}(
        Array{T}[Array{T}(a) for a in A],
        Array{T,3}[Array{T,3}(b) for b in B],
        Vector{T}[Vector{T}(c) for c in C],
        Vector{T}[Vector{T}(d) for d in D],
        E    === nothing ? nothing : Vector{T}(E),
        pA   === nothing ? nothing : Array{T}[Array{T}(a)   for a in pA],
        pB   === nothing ? nothing : Array{T,3}[Array{T,3}(b) for b in pB],
        pD   === nothing ? nothing : Vector{T}[Vector{T}(d) for d in pD],
        goal === nothing ? nothing : Vector{T}[Vector{T}(g) for g in goal],
    )
end

# --- Validation helpers ---

function _mf_validate_A(A::AbstractVector{<:AbstractArray},
                          factor_sizes::NTuple{F,Int},
                          obs_sizes::NTuple{M,Int};
                          check::Bool=true, atol::Real=1e-8) where {F, M}
    for (m, A_m) in enumerate(A)
        ndims(A_m) == 1 + F || throw(DimensionMismatch(
            "MultiFactorDiscretePOMDP: A[$m] has rank $(ndims(A_m)), expected $(1 + F) (1 + number of state factors)"))
        size(A_m, 1) == obs_sizes[m] || throw(DimensionMismatch(
            "MultiFactorDiscretePOMDP: A[$m] first dim $(size(A_m,1)) ≠ |C[$m]|=$(obs_sizes[m])"))
        for f in 1:F
            size(A_m, 1+f) == factor_sizes[f] || throw(DimensionMismatch(
                "MultiFactorDiscretePOMDP: A[$m] dim $(1+f) is $(size(A_m,1+f)), expected $(factor_sizes[f]) (= |D[$f]|)"))
        end
        if check
            inner_shape = size(A_m)[2:end]
            for idx in CartesianIndices(inner_shape)
                colsum = sum(@view A_m[:, idx])
                isapprox(colsum, 1; atol=atol) ||
                    throw(ArgumentError("MultiFactorDiscretePOMDP: A[$m] column at $(Tuple(idx)) sums to $colsum, expected 1 (atol=$atol)"))
            end
        end
    end
    return nothing
end

function _mf_validate_B(B::AbstractVector{<:AbstractArray},
                          factor_sizes::NTuple{F,Int};
                          check::Bool=true, atol::Real=1e-8) where F
    for (f, B_f) in enumerate(B)
        ndims(B_f) == 3 || throw(DimensionMismatch(
            "MultiFactorDiscretePOMDP: B[$f] has rank $(ndims(B_f)), expected 3"))
        size(B_f, 1) == factor_sizes[f] || throw(DimensionMismatch(
            "MultiFactorDiscretePOMDP: B[$f] first dim $(size(B_f,1)) ≠ |D[$f]|=$(factor_sizes[f])"))
        size(B_f, 2) == factor_sizes[f] || throw(DimensionMismatch(
            "MultiFactorDiscretePOMDP: B[$f] second dim $(size(B_f,2)) ≠ |D[$f]|=$(factor_sizes[f])"))
        if check
            for a in axes(B_f, 3), s in axes(B_f, 2)
                colsum = sum(@view B_f[:, s, a])
                isapprox(colsum, 1; atol=atol) ||
                    throw(ArgumentError("MultiFactorDiscretePOMDP: B[$f][:, $s, $a] sums to $colsum, expected 1 (atol=$atol)"))
            end
        end
    end
    return nothing
end

function _mf_validate_D(D::AbstractVector{<:AbstractVector}; check::Bool=true, atol::Real=1e-8)
    if check
        for (f, D_f) in enumerate(D)
            s = sum(D_f)
            isapprox(s, 1; atol=atol) ||
                throw(ArgumentError("MultiFactorDiscretePOMDP: D[$f] sums to $s, expected 1 (atol=$atol)"))
        end
    end
    return nothing
end

function _mf_validate_goal(goal::AbstractVector{<:AbstractVector},
                            factor_sizes::NTuple{F,Int};
                            check::Bool=true, atol::Real=1e-8) where F
    length(goal) == F || throw(DimensionMismatch(
        "MultiFactorDiscretePOMDP: |goal|=$(length(goal)), expected $F state factors"))
    for (f, g) in enumerate(goal)
        length(g) == factor_sizes[f] || throw(DimensionMismatch(
            "MultiFactorDiscretePOMDP: goal[$f] has length $(length(g)), expected $(factor_sizes[f])"))
        if check
            s = sum(g)
            isapprox(s, 1; atol=atol) ||
                throw(ArgumentError("MultiFactorDiscretePOMDP: goal[$f] sums to $s, expected 1 (atol=$atol)"))
        end
    end
    return nothing
end

function _mf_validate_shape_match(p::AbstractVector, ref::AbstractVector, name::String)
    length(p) == length(ref) || throw(DimensionMismatch(
        "MultiFactorDiscretePOMDP: |$name|=$(length(p)) ≠ |reference|=$(length(ref))"))
    for (i, (pi_, ref_i)) in enumerate(zip(p, ref))
        size(pi_) == size(ref_i) || throw(DimensionMismatch(
            "MultiFactorDiscretePOMDP: $name[$i] shape $(size(pi_)) ≠ reference shape $(size(ref_i))"))
    end
    return nothing
end

function _mf_promote_eltype(A, B, C, D, E, pA, pB, pD, goal)
    types = Type[]
    for v in A; push!(types, eltype(v)); end
    for v in B; push!(types, eltype(v)); end
    for v in C; push!(types, eltype(v)); end
    for v in D; push!(types, eltype(v)); end
    E    === nothing || push!(types, eltype(E))
    pA   === nothing || foreach(v -> push!(types, eltype(v)), pA)
    pB   === nothing || foreach(v -> push!(types, eltype(v)), pB)
    pD   === nothing || foreach(v -> push!(types, eltype(v)), pD)
    goal === nothing || foreach(v -> push!(types, eltype(v)), goal)
    return float(reduce(promote_type, types; init=Float64))
end


# =============================================================================
# GenerativeModel interface
# =============================================================================

function state_prior(m::MultiFactorDiscretePOMDP)
    if length(m.D) == 1
        return Categorical(m.D[1])
    else
        return product_belief([Categorical(D_f) for D_f in m.D])
    end
end

# observation_distribution(m, s::Vector{Int}): general F
function observation_distribution(m::MultiFactorDiscretePOMDP,
                                    s::AbstractVector{<:Integer})
    length(s) == length(m.D) || throw(DimensionMismatch(
        "observation_distribution: |state|=$(length(s)), expected $(length(m.D)) (number of state factors)"))
    M = length(m.A)
    if M == 1
        return Categorical(_mf_extract_obs_column(m.A[1], s))
    else
        return product_belief([Categorical(_mf_extract_obs_column(m.A[m_], s)) for m_ in 1:M])
    end
end

# Convenience: F=1 model accepts a scalar state
function observation_distribution(m::MultiFactorDiscretePOMDP, s::Integer)
    length(m.D) == 1 || throw(ArgumentError(
        "observation_distribution: state must be Vector{Int} for F>1 (got F=$(length(m.D)))"))
    return observation_distribution(m, [Int(s)])
end

# A[m][:, s_1, s_2, ...] for a multi-factor state s
_mf_extract_obs_column(A_m::AbstractArray, s::AbstractVector{<:Integer}) =
    A_m[:, CartesianIndex(Tuple(s))]

# transition_distribution: per-factor product
function transition_distribution(m::MultiFactorDiscretePOMDP,
                                   s::AbstractVector{<:Integer},
                                   a::AbstractVector{<:Integer})
    F = length(m.D)
    length(s) == F || throw(DimensionMismatch(
        "transition_distribution: |state|=$(length(s)), expected $F"))
    length(a) == F || throw(DimensionMismatch(
        "transition_distribution: |action|=$(length(a)), expected $F"))
    if F == 1
        return Categorical(m.B[1][:, s[1], a[1]])
    else
        return product_belief([Categorical(m.B[f][:, s[f], a[f]]) for f in 1:F])
    end
end

# Convenience: F=1 model accepts scalar
function transition_distribution(m::MultiFactorDiscretePOMDP, s::Integer, a::Integer)
    length(m.D) == 1 || throw(ArgumentError(
        "transition_distribution: scalar arguments require F=1 (got F=$(length(m.D)))"))
    return transition_distribution(m, [Int(s)], [Int(a)])
end

# log_preferences: sum across modalities
function log_preferences(m::MultiFactorDiscretePOMDP, o::AbstractVector{<:Integer})
    M = length(m.C)
    length(o) == M || throw(DimensionMismatch(
        "log_preferences: |o|=$(length(o)), expected $M (number of modalities)"))
    total = zero(eltype(m.C[1]))
    @inbounds for m_ in 1:M
        total += m.C[m_][o[m_]]
    end
    return total
end

# Convenience: M=1 model accepts a scalar
function log_preferences(m::MultiFactorDiscretePOMDP, o::Integer)
    length(m.C) == 1 || throw(ArgumentError(
        "log_preferences: scalar argument requires M=1 (got M=$(length(m.C)))"))
    return m.C[1][o]
end

# action_space: Cartesian product of per-factor action ranges
function action_space(m::MultiFactorDiscretePOMDP)
    F = length(m.B)
    if F == 1
        return [[a] for a in 1:size(m.B[1], 3)]
    else
        return collect(Iterators.product(ntuple(f -> 1:size(m.B[f], 3), F)...))
    end
end


# =============================================================================
# Optional GenerativeModel interface
# =============================================================================

state_factors(m::MultiFactorDiscretePOMDP)          = ntuple(f -> length(m.D[f]), length(m.D))
observation_modalities(m::MultiFactorDiscretePOMDP) = ntuple(o -> length(m.C[o]), length(m.C))
nstates(m::MultiFactorDiscretePOMDP)        = prod(state_factors(m))
nobservations(m::MultiFactorDiscretePOMDP)  = prod(observation_modalities(m))

function nactions(m::MultiFactorDiscretePOMDP)
    F = length(m.B)
    return F == 1 ? size(m.B[1], 3) : ntuple(f -> size(m.B[f], 3), F)
end

function goal_state_prior(m::MultiFactorDiscretePOMDP)
    m.goal === nothing && return nothing
    if length(m.goal) == 1
        return Categorical(m.goal[1])
    else
        return product_belief([Categorical(g) for g in m.goal])
    end
end


# =============================================================================
# predict_states: each factor advances independently under its own B[f]
# =============================================================================

# General multi-factor: prior is multivariate, action is a vector
function predict_states(m::MultiFactorDiscretePOMDP,
                          prior::MultivariateDistribution,
                          a::AbstractVector{<:Integer})
    F = length(m.D)
    length(a) == F || throw(DimensionMismatch(
        "predict_states: |action|=$(length(a)), expected $F"))
    next_marginals = [
        Categorical(let p = m.B[f][:, :, a[f]] * probs(marginal(prior, f))
                        p ./ sum(p)
                    end)
        for f in 1:F
    ]
    return product_belief(next_marginals)
end

# F=1 with Categorical prior
function predict_states(m::MultiFactorDiscretePOMDP,
                          prior::Categorical,
                          a::AbstractVector{<:Integer})
    length(m.D) == 1 || throw(ArgumentError(
        "predict_states: Categorical prior requires F=1 (got F=$(length(m.D)))"))
    length(a) == 1 || throw(DimensionMismatch(
        "predict_states: |action|=$(length(a)), expected 1"))
    p_next = m.B[1][:, :, a[1]] * probs(prior)
    return Categorical(p_next ./ sum(p_next))
end

function predict_states(m::MultiFactorDiscretePOMDP,
                          prior::Categorical,
                          a::Integer)
    length(m.D) == 1 || throw(ArgumentError(
        "predict_states: scalar action requires F=1 (got F=$(length(m.D)))"))
    return predict_states(m, prior, [Int(a)])
end


# =============================================================================
# Conversion: single-factor MultiFactorDiscretePOMDP <-> DiscretePOMDP
# =============================================================================

"""
    DiscretePOMDP(m::MultiFactorDiscretePOMDP)

Convert a single-factor / single-modality `MultiFactorDiscretePOMDP` to the
classic `DiscretePOMDP` (used by the single-factor algorithms). Throws if
the model has multiple state factors or modalities.
"""
function DiscretePOMDP(m::MultiFactorDiscretePOMDP)
    length(m.D) == 1 || throw(ArgumentError(
        "Cannot convert multi-factor MultiFactorDiscretePOMDP (F=$(length(m.D))) to single-factor DiscretePOMDP"))
    length(m.C) == 1 || throw(ArgumentError(
        "Cannot convert multi-modality MultiFactorDiscretePOMDP (M=$(length(m.C))) to single-modality DiscretePOMDP"))
    return DiscretePOMDP(m.A[1], m.B[1], m.C[1], m.D[1];
                         E    = m.E,
                         pA   = m.pA   === nothing ? nothing : m.pA[1],
                         pB   = m.pB   === nothing ? nothing : m.pB[1],
                         pD   = m.pD   === nothing ? nothing : m.pD[1],
                         goal = m.goal === nothing ? nothing : m.goal[1],
                         check = false)
end

"""
    MultiFactorDiscretePOMDP(m::DiscretePOMDP)

Lift a single-factor `DiscretePOMDP` to the multi-factor representation.
Useful for passing to multi-factor algorithms while keeping the rest of the
agent unchanged.
"""
function MultiFactorDiscretePOMDP(m::DiscretePOMDP)
    return MultiFactorDiscretePOMDP(
        [m.A], [m.B], [m.C], [m.D];
        E    = m.E,
        pA   = m.pA   === nothing ? nothing : [m.pA],
        pB   = m.pB   === nothing ? nothing : [m.pB],
        pD   = m.pD   === nothing ? nothing : [m.pD],
        goal = m.goal === nothing ? nothing : [m.goal],
        check = false,
    )
end
