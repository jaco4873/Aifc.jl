# Multi-factor EnumerativeEFE tests.
#
# - F=1 multi-factor planning matches single-factor on every quantity (sanity)
# - Decomposition identity G = -(prag + epi) holds in multi-factor
# - Cross-validation: 4×2 multi-factor T-maze gives the same total EFE per
#   policy as 8-state single-factor T-maze (since the posterior factorizes
#   for T-maze, mean-field is exact)

using Distributions: Categorical, MultivariateDistribution, probs

@testset "F=1 multi-factor matches single-factor EFE" begin
    rng = Xoshiro(0xE1)
    A = zeros(3, 4)
    for s in 1:4
        A[:, s] = softmax(randn(rng, 3))
    end
    B = zeros(4, 4, 2)
    for a in 1:2, s in 1:4
        B[:, s, a] = softmax(randn(rng, 4))
    end
    C = randn(rng, 3)
    D = softmax(randn(rng, 4))

    sf = DiscretePOMDP(A, B, C, D)
    mf = MultiFactorDiscretePOMDP(sf)

    p_sf = EnumerativeEFE(γ=8.0, horizon=2)
    p_mf = EnumerativeEFE(γ=8.0, horizon=2)

    qs_sf = state_prior(sf)
    qs_mf = state_prior(mf)

    # Compare component-by-component on a few hand-picked policies
    for π_sf in [[1, 1], [1, 2], [2, 1], [2, 2]]
        π_mf = [[a] for a in π_sf]   # multi-factor wraps each action

        prag_sf = pragmatic_value(p_sf, sf, qs_sf, π_sf)
        prag_mf = pragmatic_value(p_mf, mf, qs_mf, π_mf)
        @test prag_sf ≈ prag_mf atol=1e-10

        epi_sf = epistemic_value(p_sf, sf, qs_sf, π_sf)
        epi_mf = epistemic_value(p_mf, mf, qs_mf, π_mf)
        @test epi_sf ≈ epi_mf atol=1e-10

        G_sf = expected_free_energy(p_sf, sf, qs_sf, π_sf)
        G_mf = expected_free_energy(p_mf, mf, qs_mf, π_mf)
        @test G_sf ≈ G_mf atol=1e-10
    end
end

@testset "EFE decomposition holds in multi-factor" begin
    rng = Xoshiro(0xE2)
    A = zeros(3, 4, 2)
    for s1 in 1:4, s2 in 1:2
        A[:, s1, s2] = softmax(randn(rng, 3))
    end
    B1 = zeros(4, 4, 2)
    for a in 1:2, s in 1:4
        B1[:, s, a] = softmax(randn(rng, 4))
    end
    B2 = zeros(2, 2, 1)
    B2[1, 1, 1] = 1; B2[2, 2, 1] = 1
    C  = [randn(rng, 3)]
    D  = [softmax(randn(rng, 4)), softmax(randn(rng, 2))]
    m  = MultiFactorDiscretePOMDP([A], [B1, B2], C, D)

    p = EnumerativeEFE(γ=4.0, horizon=2)
    qs = state_prior(m)

    for a1_seq in 1:2, a2_seq in 1:2
        π = [[a1_seq, 1], [a2_seq, 1]]
        G    = expected_free_energy(p, m, qs, π)
        prag = pragmatic_value(p, m, qs, π)
        epi  = epistemic_value(p, m, qs, π)
        @test G ≈ -(prag + epi) atol=1e-10
        @test epi >= -1e-10
    end
end

@testset "T-maze cross-validation: every policy's total matches" begin
    sf = tmaze_model(cue_reliability=0.95, preference=4.0)
    mf = tmaze_model_multi_factor(cue_reliability=0.95, preference=4.0)

    p_sf = EnumerativeEFE(γ=8.0, horizon=2)
    p_mf = EnumerativeEFE(γ=8.0, horizon=2)

    qs_sf = state_prior(sf)
    qs_mf = state_prior(mf)

    # All 16 single-factor 2-step policies (4 actions × 2 steps)
    for a1 in 1:4, a2 in 1:4
        π_sf = [a1, a2]
        π_mf = [[a1, 1], [a2, 1]]    # wrap with passive context action

        prag_sf = pragmatic_value(p_sf, sf, qs_sf, π_sf)
        prag_mf = pragmatic_value(p_mf, mf, qs_mf, π_mf)
        @test prag_sf ≈ prag_mf atol=1e-9

        epi_sf = epistemic_value(p_sf, sf, qs_sf, π_sf)
        epi_mf = epistemic_value(p_mf, mf, qs_mf, π_mf)
        @test epi_sf ≈ epi_mf atol=1e-9
    end
end

@testset "Best vanilla 2-step total = 2·log(2) (multi-factor)" begin
    mf = tmaze_model_multi_factor(cue_reliability=0.95, preference=4.0)
    p  = EnumerativeEFE(γ=8.0, horizon=2)
    qs = state_prior(mf)

    pols = [[ [a1, 1], [a2, 1] ] for a1 in 1:4 for a2 in 1:4]
    totals = [pragmatic_value(p, mf, qs, π) + epistemic_value(p, mf, qs, π) for π in pols]
    best_total = maximum(totals)

    @test best_total ≈ 2 * log(2) atol=1e-6
end
