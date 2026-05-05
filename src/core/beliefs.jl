# Belief representation built on Distributions.jl.
#
# A "belief" in this package is just a `Distributions.Distribution` —
# typically `Categorical` for single-factor discrete state, a product of
# `Categorical`s for multi-factor discrete, `MvNormal` for continuous, or
# user-supplied. We add a thin `belief_style` trait for invariant testing
# (it lets the conformance suite know which checks to apply) and a few
# convenience helpers.

import Distributions
using Distributions: Distribution, Categorical, MvNormal, Normal
using Distributions: probs, ncategories, support, logpdf, pdf, mean, var, cov
using Distributions: UnivariateDistribution, MultivariateDistribution
using Distributions: product_distribution

"""
    BeliefStyle

Trait abstract type used by the conformance test suite to decide which
mathematical invariants apply to a given belief representation. NOT used
for runtime dispatch in inference algorithms — those receive concrete
`Distribution` types.

Provided styles: `CategoricalStyle`, `GaussianStyle`, `ProductStyle{S}`,
`ParticleStyle`. Users can add new styles by defining a method:

    Aifc.belief_style(::Type{MyDistribution}) = MyStyle()
"""
abstract type BeliefStyle end

struct CategoricalStyle <: BeliefStyle end
struct GaussianStyle <: BeliefStyle end
struct ProductStyle{S<:BeliefStyle} <: BeliefStyle
    inner::S
end
struct ParticleStyle <: BeliefStyle end
struct UnknownStyle <: BeliefStyle end

"""
    belief_style(d::Distribution) -> BeliefStyle

Return the trait classifying which invariant tests apply to `d`. Default is
`UnknownStyle()`, in which case only generic invariants (e.g. `pdf ≥ 0`)
will be checked.
"""
belief_style(::Categorical) = CategoricalStyle()
belief_style(::Normal) = GaussianStyle()
belief_style(::MvNormal) = GaussianStyle()
belief_style(p::Distributions.Product) = ProductStyle(belief_style(first(p.v)))
belief_style(::Distribution) = UnknownStyle()

"""
    isnormalized(d::Distribution; atol=1e-8) -> Bool

Sanity check that probability mass sums to 1 within tolerance. For built-in
Distributions.jl types this is trivially true by construction; the check is
useful primarily when a `Categorical` has been built from a hand-rolled
probability vector that may have drifted under repeated arithmetic.
"""
isnormalized(d::Categorical; atol::Real=1e-8) = isapprox(sum(probs(d)), 1; atol=atol)
isnormalized(p::Distributions.Product; atol::Real=1e-8) =
    all(isnormalized(c; atol=atol) for c in p.v)
isnormalized(::Distribution; atol::Real=1e-8) = true

"""
    marginal(d::Distribution, i::Integer)

Return the `i`-th marginal of a factorized belief. Defined for
`Distributions.Product`; falls back to `d` itself for univariate
distributions when `i == 1`.
"""
marginal(p::Distributions.Product, i::Integer) = p.v[i]
function marginal(d::UnivariateDistribution, i::Integer)
    i == 1 || throw(ArgumentError("marginal: univariate distribution has only factor 1, got $i"))
    return d
end

"""
    nfactors(d::Distribution) -> Int

Number of mean-field factors. `nfactors(p::Product) == length(p.v)`,
`nfactors(d::UnivariateDistribution) == 1`, otherwise 1 by default.
"""
nfactors(p::Distributions.Product) = length(p.v)
nfactors(::UnivariateDistribution) = 1
nfactors(::Distribution) = 1

"""
    factor_probs(d::Distribution, i::Integer) -> Vector{<:Real}

Probability vector of the `i`-th categorical marginal. Convenience wrapper
around `probs(marginal(d, i))`.
"""
factor_probs(d::Distribution, i::Integer) = probs(marginal(d, i))

"""
    categorical_belief(p::AbstractVector{<:Real}) -> Categorical

Construct a `Categorical` belief from a probability vector, asserting that
the input is non-negative and sums to 1 within `1e-8`. Use this at API
boundaries where you want a safety net; internal hot loops can pass raw
vectors and skip the check.
"""
function categorical_belief(p::AbstractVector{<:Real}; atol::Real=1e-8)
    all(>=(zero(eltype(p))), p) ||
        throw(ArgumentError("categorical_belief: probability vector contains negative entries"))
    s = sum(p)
    isapprox(s, 1; atol=atol) ||
        throw(ArgumentError("categorical_belief: probability vector sums to $(s), expected 1 (within atol=$atol)"))
    return Categorical(collect(float.(p)))
end

"""
    product_belief(factors::AbstractVector{<:UnivariateDistribution})

Construct a factorized belief from a list of marginals. Thin wrapper over
`Distributions.product_distribution` that accepts a vector of arbitrary
univariate distributions (typically `Categorical`s in active-inference
settings).
"""
product_belief(factors::AbstractVector{<:UnivariateDistribution}) =
    product_distribution(factors)
