# Tests for belief helpers: marginal, nfactors, factor_probs,
# categorical_belief validation, product_belief, belief_style.

using Distributions: Categorical, MvNormal, Normal, probs

@testset "categorical_belief: validates input" begin
    # Valid input
    b = categorical_belief([0.3, 0.7])
    @test b isa Categorical
    @test probs(b) == [0.3, 0.7]

    # Reject negative entries
    @test_throws ArgumentError categorical_belief([-0.1, 1.1])

    # Reject non-normalized
    @test_throws ArgumentError categorical_belief([0.3, 0.5])
    @test_throws ArgumentError categorical_belief([0.5, 0.5, 0.5])

    # Custom atol
    @test categorical_belief([0.3, 0.7 + 1e-10]; atol=1e-9) isa Categorical
    @test_throws ArgumentError categorical_belief([0.3, 0.7 + 1e-6]; atol=1e-9)
end

@testset "product_belief from a vector of Categoricals" begin
    cats = [Categorical([0.5, 0.5]), Categorical([0.3, 0.4, 0.3])]
    p = product_belief(cats)
    @test nfactors(p) == 2
    @test marginal(p, 1) == cats[1]
    @test marginal(p, 2) == cats[2]
end

@testset "marginal: univariate fallback" begin
    c = Categorical([0.4, 0.6])
    # Univariate's only factor is itself
    @test marginal(c, 1) === c
    @test_throws ArgumentError marginal(c, 2)
    @test_throws ArgumentError marginal(c, 0)
end

@testset "nfactors / factor_probs" begin
    # Univariate
    c = Categorical([0.5, 0.5])
    @test nfactors(c) == 1
    @test factor_probs(c, 1) == [0.5, 0.5]

    # Multivariate Product
    p = product_belief([Categorical([0.2, 0.8]), Categorical([0.4, 0.6])])
    @test nfactors(p) == 2
    @test factor_probs(p, 1) == [0.2, 0.8]
    @test factor_probs(p, 2) == [0.4, 0.6]

    # MvNormal: nfactors falls back to 1 (single multivariate "factor")
    mvn = MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0])
    @test nfactors(mvn) == 1
end

@testset "belief_style trait" begin
    @test belief_style(Categorical([0.5, 0.5])) isa CategoricalStyle
    @test belief_style(Normal()) isa GaussianStyle
    @test belief_style(MvNormal([0.0], [1.0;;])) isa GaussianStyle

    p = product_belief([Categorical([0.5, 0.5]), Categorical([0.3, 0.7])])
    @test belief_style(p) isa ProductStyle
    @test belief_style(p).inner isa CategoricalStyle

    # Unknown distribution falls back to UnknownStyle
    using Distributions: Beta
    @test belief_style(Beta(1, 1)) isa UnknownStyle
end

@testset "isnormalized" begin
    @test isnormalized(Categorical([0.5, 0.5]))
    @test isnormalized(product_belief([Categorical([0.4, 0.6]),
                                          Categorical([0.5, 0.5])]))
    # Built-in Distributions are trivially normalized — fallback returns true
    @test isnormalized(Normal())
    @test isnormalized(MvNormal([0.0], [1.0;;]))
end
