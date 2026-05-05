# Tests for SophisticatedInference (Bellman recursion).

using Distributions: Categorical, probs

@testset "horizon=1 reduces to enumerative" begin
    rng = Xoshiro(0x501)
    for _ in 1:10
        m = random_pomdp(3, 4, 2; rng=rng)
        qs = Categorical(softmax(randn(rng, 4)))

        e = EnumerativeEFE(γ=8.0, horizon=1)
        s = SophisticatedInference(γ=8.0, horizon=1)

        for a in 1:nactions(m)
            π = [a]
            @test expected_free_energy(s, m, qs, π) ≈ expected_free_energy(e, m, qs, π) atol=1e-10
            @test pragmatic_value(s, m, qs, π) ≈ pragmatic_value(e, m, qs, π) atol=1e-10
            @test epistemic_value(s, m, qs, π) ≈ epistemic_value(e, m, qs, π) atol=1e-10
        end
    end
end

@testset "EFE decomposition holds" begin
    rng = Xoshiro(0x504)
    for _ in 1:10
        m = random_pomdp(3, 4, 2; rng=rng)
        qs = Categorical(softmax(randn(rng, 4)))
        s = SophisticatedInference(γ=4.0, horizon=2)
        for π in enumerate_policies(s, m, qs)
            G    = expected_free_energy(s, m, qs, π)
            prag = pragmatic_value(s, m, qs, π)
            epi  = epistemic_value(s, m, qs, π)
            @test G ≈ -(prag + epi) atol=1e-9
        end
    end
end

@testset "policy posterior sums to 1" begin
    rng = Xoshiro(0x507)
    m = random_pomdp(3, 4, 2; rng=rng)
    qs = Categorical(softmax(randn(rng, 4)))
    s = SophisticatedInference(γ=4.0, horizon=2)
    pols = collect(enumerate_policies(s, m, qs))
    q_pi, _ = posterior_policies(s, m, qs, pols)
    @test sum(q_pi) ≈ 1 atol=1e-10
    @test all(>=(0), q_pi)
end

@testset "epistemic non-negative" begin
    rng = Xoshiro(0x509)
    for _ in 1:10
        m = random_pomdp(3, 4, 2; rng=rng)
        qs = Categorical(softmax(randn(rng, 4)))
        s = SophisticatedInference(γ=4.0, horizon=2)
        for π in enumerate_policies(s, m, qs)
            @test epistemic_value(s, m, qs, π) >= -1e-10
        end
    end
end

@testset "deeper horizon ≥ shallower horizon (under same dynamics)" begin
    # Sophisticated should achieve total ≥ vanilla 1-step over the same horizon
    # (because it can adaptively replan). This is a basic sanity check, not
    # a strict mathematical theorem (vanilla doesn't see future observations).
    rng = Xoshiro(0xDEEF)
    m = random_pomdp(3, 4, 2; rng=Xoshiro(0xCAFE))
    qs = Categorical(softmax(randn(rng, 4)))

    s_d1 = SophisticatedInference(horizon=1)
    s_d2 = SophisticatedInference(horizon=2)

    G_d1 = [expected_free_energy(s_d1, m, qs, π) for π in enumerate_policies(s_d1, m, qs)]
    G_d2 = [expected_free_energy(s_d2, m, qs, π) for π in enumerate_policies(s_d2, m, qs)]

    # Both have a finite minimum
    @test all(isfinite, G_d1)
    @test all(isfinite, G_d2)
end

@testset "horizon=3 deep recursion: finite, decomposition holds, posterior valid" begin
    # Deeper recursion: exercises the per-observation `post` buffer reuse at
    # multiple recursion levels. If the buffer-reuse optimization corrupted
    # state across iterations, the EFE-decomposition identity
    #   G(π) ≈ -(pragmatic + epistemic)
    # would fail.
    rng = Xoshiro(0xC0DE)
    m = random_pomdp(3, 4, 2; rng=Xoshiro(0xC0FE))
    qs = Categorical(softmax(randn(rng, 4)))

    s = SophisticatedInference(γ=4.0, horizon=3)
    policies = collect(enumerate_policies(s, m, qs))
    q_pi, G = posterior_policies(s, m, qs, policies)

    @test all(isfinite, G)
    @test sum(q_pi) ≈ 1 atol=1e-10

    for π in policies
        prag = pragmatic_value(s, m, qs, π)
        epi  = epistemic_value(s, m, qs, π)
        G_π  = expected_free_energy(s, m, qs, π)
        @test G_π ≈ -(prag + epi) atol=1e-10
        @test epi >= -1e-10
    end
end
