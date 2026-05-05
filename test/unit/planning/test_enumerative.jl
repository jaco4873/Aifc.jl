# Tests for EnumerativeEFE planning.

using Distributions: Categorical, probs

# A canonical small POMDP used throughout these tests:
#   - 2 states, 2 observations, 2 actions
#   - A = [0.9 0.1; 0.1 0.9]   (informative observation)
#   - Action 1 = identity, action 2 = swap
function _two_state_pomdp(; preference::Real=2.0)
    A = [0.9 0.1; 0.1 0.9]
    B = zeros(2, 2, 2)
    B[1, 1, 1] = 1; B[2, 2, 1] = 1
    B[2, 1, 2] = 1; B[1, 2, 2] = 1
    C = [Float64(preference), -Float64(preference)]
    D = [0.5, 0.5]
    return DiscretePOMDP(A, B, C, D)
end

@testset "enumerate_policies" begin
    m = _two_state_pomdp()
    qs = state_prior(m)

    p = EnumerativeEFE(horizon=1)
    pols = collect(enumerate_policies(p, m, qs))
    @test length(pols) == 2
    @test sort([first(π) for π in pols]) == [1, 2]

    p2 = EnumerativeEFE(horizon=2)
    pols2 = collect(enumerate_policies(p2, m, qs))
    @test length(pols2) == 4   # 2^2
    @test all(length(π) == 2 for π in pols2)

    p3 = EnumerativeEFE(horizon=3)
    pols3 = collect(enumerate_policies(p3, m, qs))
    @test length(pols3) == 8
end

@testset "EFE decomposition: G = -(pragmatic + epistemic)" begin
    rng = Xoshiro(0x6F)
    for _ in 1:30
        m = random_pomdp(3, 4, 2; rng=rng)
        qs = Categorical(softmax(randn(rng, 4)))
        p = EnumerativeEFE(γ=8.0, horizon=2)
        for π in enumerate_policies(p, m, qs)
            G    = expected_free_energy(p, m, qs, π)
            prag = pragmatic_value(p, m, qs, π)
            epi  = epistemic_value(p, m, qs, π)
            @test G ≈ -(prag + epi) atol=1e-10
        end
    end
end

@testset "epistemic non-negative" begin
    rng = Xoshiro(0xE17)
    for _ in 1:30
        m = random_pomdp(3, 4, 2; rng=rng)
        qs = Categorical(softmax(randn(rng, 4)))
        p = EnumerativeEFE(horizon=2)
        for π in enumerate_policies(p, m, qs)
            @test epistemic_value(p, m, qs, π) >= -1e-10
        end
    end
end

@testset "policy posterior sums to 1" begin
    rng = Xoshiro(0xC1)
    for _ in 1:20
        m = random_pomdp(3, 4, 2; rng=rng)
        qs = Categorical(softmax(randn(rng, 4)))
        for h in 1:3
            p = EnumerativeEFE(γ=4.0, horizon=h)
            pols = collect(enumerate_policies(p, m, qs))
            q_pi, G = posterior_policies(p, m, qs, pols)
            @test sum(q_pi) ≈ 1 atol=1e-10
            @test all(>=(zero(eltype(q_pi))), q_pi)
            @test length(G) == length(pols)
        end
    end
end

@testset "γ → ∞ collapses to argmin G" begin
    m = _two_state_pomdp(preference=4.0)
    # NOTE: with a symmetric prior [0.5, 0.5] and the symmetric two-state
    # model, all policies have the same G (no policy-dependent information).
    # Use an asymmetric prior to break the symmetry.
    qs = Categorical([0.9, 0.1])
    p_sharp = EnumerativeEFE(γ=1e6, horizon=2)
    pols = collect(enumerate_policies(p_sharp, m, qs))
    q_pi, G = posterior_policies(p_sharp, m, qs, pols)
    @test maximum(q_pi) > 0.999
    @test argmax(q_pi) == argmin(G)
end

@testset "γ = 0 → uniform" begin
    m = _two_state_pomdp()
    qs = state_prior(m)
    p_flat = EnumerativeEFE(γ=0.0, horizon=2)
    pols = collect(enumerate_policies(p_flat, m, qs))
    q_pi, _ = posterior_policies(p_flat, m, qs, pols)
    @test all(≈(1 / length(pols)), q_pi)
end

@testset "use_pragmatic / use_epistemic toggles" begin
    m = _two_state_pomdp(preference=3.0)
    qs = state_prior(m)
    π = [1, 2]

    p_full = EnumerativeEFE()
    p_no_prag = EnumerativeEFE(use_pragmatic=false)
    p_no_epi  = EnumerativeEFE(use_epistemic=false)

    @test pragmatic_value(p_no_prag, m, qs, π) == 0.0
    @test epistemic_value(p_no_epi,  m, qs, π) == 0.0
    # Full G = sum of components
    @test expected_free_energy(p_full, m, qs, π) ≈
          -(pragmatic_value(p_full, m, qs, π) + epistemic_value(p_full, m, qs, π))
end

@testset "Conformance" begin
    rng = Xoshiro(0xC0)
    m = random_pomdp(3, 4, 2; rng=Xoshiro(0xC1))
    p = EnumerativeEFE(horizon=2)
    qs = Categorical(softmax(randn(rng, 4)))
    pols = collect(enumerate_policies(p, m, qs))
    Aifc.Testing.test_policy_inference(p, m, qs, pols; rng=rng)
end
