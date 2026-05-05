# Tests for InductiveEFE — goal-state-prior weighting.

using Distributions: Categorical, probs
using LinearAlgebra: dot

@testset "no goal: identical to base planner" begin
    rng = Xoshiro(0x1)
    m = random_pomdp(3, 4, 2; rng=rng)   # no goal_state_prior set
    qs = Categorical(softmax(randn(rng, 4)))

    base = EnumerativeEFE(γ=4.0, horizon=2)
    induct = InductiveEFE(base; goal_weight=2.0)

    for π in enumerate_policies(induct, m, qs)
        @test expected_free_energy(induct, m, qs, π) ≈
              expected_free_energy(base, m, qs, π) atol=1e-12
    end
end

@testset "goal_weight=0 reduces to base" begin
    A = [0.9 0.1; 0.1 0.9]
    B = zeros(2, 2, 2)
    B[1, 1, 1] = 1; B[2, 2, 1] = 1
    B[2, 1, 2] = 1; B[1, 2, 2] = 1
    C = [0.0, 0.0]
    D = [0.5, 0.5]
    m = DiscretePOMDP(A, B, C, D; goal=[0.0, 1.0])

    base = EnumerativeEFE(horizon=2)
    induct = InductiveEFE(base; goal_weight=0.0)

    qs = state_prior(m)
    for π in enumerate_policies(induct, m, qs)
        @test expected_free_energy(induct, m, qs, π) ≈
              expected_free_energy(base, m, qs, π) atol=1e-12
    end
end

@testset "goal-reaching policies are favored" begin
    # 2-state model: action 1 = stay, action 2 = swap. Goal = state 2.
    # Starting from state 1, action 2 reaches the goal; action 1 doesn't.
    A = [1.0 0.0; 0.0 1.0]    # perfect observation
    B = zeros(2, 2, 2)
    B[1, 1, 1] = 1; B[2, 2, 1] = 1   # action 1 = stay
    B[2, 1, 2] = 1; B[1, 2, 2] = 1   # action 2 = swap
    C = [0.0, 0.0]
    D = [1.0, 0.0]            # certainty in state 1
    m = DiscretePOMDP(A, B, C, D; goal=[0.0, 1.0])

    base = EnumerativeEFE(γ=4.0, horizon=1, use_pragmatic=false)
    induct = InductiveEFE(base; goal_weight=4.0)

    qs = state_prior(m)
    pols = collect(enumerate_policies(induct, m, qs))
    q_pi, G = posterior_policies(induct, m, qs, pols)

    # Find the policy that uses action 2 (the swap, which reaches the goal)
    a2_idx = findfirst(π -> first(π) == 2, pols)
    a1_idx = findfirst(π -> first(π) == 1, pols)
    @test q_pi[a2_idx] > q_pi[a1_idx]
end

@testset "decomposition still holds (goal weighting rolled into pragmatic)" begin
    A = [0.9 0.1; 0.1 0.9]
    B = zeros(2, 2, 2)
    B[1,1,1]=1; B[2,2,1]=1
    B[2,1,2]=1; B[1,2,2]=1
    C = [1.0, -1.0]
    D = [0.5, 0.5]
    m = DiscretePOMDP(A, B, C, D; goal=[0.3, 0.7])

    induct = InductiveEFE(EnumerativeEFE(horizon=2); goal_weight=2.0)
    qs = state_prior(m)
    for π in enumerate_policies(induct, m, qs)
        G = expected_free_energy(induct, m, qs, π)
        prag = pragmatic_value(induct, m, qs, π)
        epi = epistemic_value(induct, m, qs, π)
        @test G ≈ -(prag + epi) atol=1e-12
    end
end

@testset "wraps SophisticatedInference too" begin
    A = [0.9 0.1; 0.1 0.9]
    B = zeros(2, 2, 2)
    B[1,1,1]=1; B[2,2,1]=1
    B[2,1,2]=1; B[1,2,2]=1
    C = [0.0, 0.0]
    D = [1.0, 0.0]
    m = DiscretePOMDP(A, B, C, D; goal=[0.0, 1.0])

    induct = InductiveEFE(SophisticatedInference(horizon=2); goal_weight=2.0)
    qs = state_prior(m)
    pols = collect(enumerate_policies(induct, m, qs))
    q_pi, G = posterior_policies(induct, m, qs, pols)
    @test sum(q_pi) ≈ 1 atol=1e-10
    @test all(isfinite, G)
end
