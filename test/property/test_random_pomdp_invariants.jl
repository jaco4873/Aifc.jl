# Property-based / fuzz testing on randomly-generated POMDPs.
#
# We generate many random `DiscretePOMDP`s across a wide configuration
# space and assert that Aifc's mathematical invariants
# hold on every one. This catches edge-case bugs that single seeded
# scenarios miss — extreme dimensions, near-degenerate likelihoods,
# zero-mass priors, etc.
#
# Implemented with seeded `Xoshiro` rather than full Supposition.jl-style
# property testing — adequate for our needs without adding a dep.

using Distributions: Categorical, probs
using Random: Xoshiro

const N_TRIALS = 200

function _random_pomdp(rng::Xoshiro)
    num_obs    = rand(rng, 2:8)
    num_states = rand(rng, 2:8)
    num_actions = rand(rng, 2:4)
    return random_pomdp(num_obs, num_states, num_actions; rng=rng), num_obs, num_states, num_actions
end

@testset "FPI posterior is a valid Categorical for arbitrary models" begin
    for trial in 1:N_TRIALS
        rng = Xoshiro(trial * 1009)
        m, num_obs, num_states, _ = _random_pomdp(rng)
        prior = Categorical(softmax(randn(rng, num_states)))
        obs = rand(rng, 1:num_obs)
        q = infer_states(FixedPointIteration(), m, prior, obs)

        @test q isa Categorical
        @test isnormalized(q)
        @test all(>=(0), probs(q))
    end
end

@testset "Free energy at FPI's posterior obeys Gibbs inequality" begin
    for trial in 1:N_TRIALS
        rng = Xoshiro(trial * 2011)
        m, num_obs, num_states, _ = _random_pomdp(rng)
        prior = Categorical(softmax(randn(rng, num_states)))
        obs = rand(rng, 1:num_obs)

        alg = FixedPointIteration()
        q = infer_states(alg, m, prior, obs)
        F = free_energy(alg, m, prior, obs, q)

        # Compute log evidence directly: log P(o) = log Σ_s A[o,s] · prior[s]
        prior_p = probs(prior)
        log_evidence = log(sum(m.A[obs, :] .* prior_p))
        # Gibbs: F[q*] = -log P(o) (within tolerance)
        @test F ≈ -log_evidence atol=1e-9
    end
end

@testset "EFE decomposition holds on random POMDPs at random horizons" begin
    for trial in 1:N_TRIALS
        rng = Xoshiro(trial * 3019)
        m, _, num_states, _ = _random_pomdp(rng)
        qs = Categorical(softmax(randn(rng, num_states)))

        H = rand(rng, 1:3)
        p = EnumerativeEFE(γ=2.0 + 6.0 * rand(rng), horizon=H)
        for π in enumerate_policies(p, m, qs)
            G = expected_free_energy(p, m, qs, π)
            prag = pragmatic_value(p, m, qs, π)
            epi = epistemic_value(p, m, qs, π)
            @test G ≈ -(prag + epi) atol=1e-9
            @test epi >= -1e-10        # information non-negativity
        end
    end
end

@testset "Policy posterior is normalized and non-negative" begin
    for trial in 1:N_TRIALS
        rng = Xoshiro(trial * 4027)
        m, _, num_states, _ = _random_pomdp(rng)
        qs = Categorical(softmax(randn(rng, num_states)))

        H = rand(rng, 1:3)
        γ = exp(2 * randn(rng))   # γ ∈ (eps, large)
        p = EnumerativeEFE(γ=γ, horizon=H)
        pols = collect(enumerate_policies(p, m, qs))
        q_pi, G = posterior_policies(p, m, qs, pols)

        @test all(>=(0), q_pi)
        @test sum(q_pi) ≈ 1 atol=1e-10
        @test all(isfinite, G)
    end
end

@testset "Sophisticated decomposition (G = -(prag + epi)) on random POMDPs" begin
    # Smaller N because Sophisticated at depth 2+ is expensive
    for trial in 1:30
        rng = Xoshiro(trial * 5039)
        m, _, num_states, _ = _random_pomdp(rng)
        qs = Categorical(softmax(randn(rng, num_states)))

        for depth in 1:2
            p = SophisticatedInference(γ=4.0, horizon=depth)
            for π in enumerate_policies(p, m, qs)
                G = expected_free_energy(p, m, qs, π)
                prag = pragmatic_value(p, m, qs, π)
                epi = epistemic_value(p, m, qs, π)
                @test G ≈ -(prag + epi) atol=1e-8
            end
        end
    end
end

@testset "Stochastic action distribution sums to 1 across random configs" begin
    for trial in 1:N_TRIALS
        rng = Xoshiro(trial * 6043)
        m, _, num_states, num_actions = _random_pomdp(rng)
        qs = Categorical(softmax(randn(rng, num_states)))
        p = EnumerativeEFE(γ=4.0, horizon=rand(rng, 1:2))
        pols = collect(enumerate_policies(p, m, qs))
        q_pi, _ = posterior_policies(p, m, qs, pols)

        α = exp(2 * randn(rng))   # broad range
        sel = Stochastic(α=α)
        d = action_distribution(sel, q_pi, pols)
        ps = probs(d)
        @test sum(ps) ≈ 1 atol=1e-9
        @test all(>=(0), ps)
    end
end
