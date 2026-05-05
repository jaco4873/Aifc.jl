# Tests for FixedPointIteration on DiscretePOMDP.

using Distributions: Categorical, probs

@testset "single-factor closed form" begin
    # For a single-factor POMDP, FPI's posterior is exactly:
    #   q(s) ∝ A[o, s] · prior[s]
    #   (i.e. softmax(log A[o, :] + log prior))
    A = [0.9 0.1; 0.1 0.9]
    B = zeros(2, 2, 2)
    B[1, 1, 1] = 1; B[2, 2, 1] = 1     # action 1 = identity
    B[2, 1, 2] = 1; B[1, 2, 2] = 1     # action 2 = swap
    C = [0.0, 0.0]
    D = [0.5, 0.5]
    m = DiscretePOMDP(A, B, C, D)

    alg = FixedPointIteration()

    # Observation 1 with uniform prior: posterior should be A[1, :] normalized
    q = infer_states(alg, m, Categorical([0.5, 0.5]), 1)
    raw = [0.9 * 0.5, 0.1 * 0.5]   # A[1, s] · prior[s]
    expected = raw ./ sum(raw)
    @test probs(q) ≈ expected atol=1e-12

    # Observation 2 with informed prior
    q = infer_states(alg, m, Categorical([0.7, 0.3]), 2)
    raw = [0.1 * 0.7, 0.9 * 0.3]   # A[2, s] · prior[s]
    expected = raw ./ sum(raw)
    @test probs(q) ≈ expected atol=1e-12
end

@testset "free energy = -log P(o) at the exact posterior" begin
    # Gibbs' inequality: F = -log P(o) when q* matches the exact posterior
    rng = Xoshiro(1011)
    for _ in 1:20
        m = random_pomdp(4, 5, 2; rng=rng)
        prior_p = softmax(randn(rng, 5))
        prior = Categorical(prior_p)
        o = rand(rng, 1:4)

        alg = FixedPointIteration()
        q = infer_states(alg, m, prior, o)
        F = free_energy(alg, m, prior, o, q)

        # log P(o) = log Σ_s A[o, s] · prior[s]
        log_evidence = log(sum(m.A[o, :] .* prior_p))
        @test F ≈ -log_evidence atol=1e-10
    end
end

@testset "F at non-exact q is greater" begin
    rng = Xoshiro(2024)
    m = random_pomdp(3, 4, 2; rng=rng)
    prior = Categorical(softmax(randn(rng, 4)))
    o = 2
    alg = FixedPointIteration()

    q_star = infer_states(alg, m, prior, o)
    F_star = free_energy(alg, m, prior, o, q_star)

    for _ in 1:10
        q_random = Categorical(softmax(randn(rng, 4)))
        F_random = free_energy(alg, m, prior, o, q_random)
        @test F_random >= F_star - 1e-10
    end
end

@testset "interface capability queries" begin
    alg = FixedPointIteration()
    @test supports_states(alg)
    @test !supports_parameters(alg)
    @test :states in supported_targets(alg)
end

@testset "Conformance" begin
    rng = Xoshiro(31)
    m = random_pomdp(3, 4, 2; rng=Xoshiro(33))
    Aifc.Testing.test_state_inference(FixedPointIteration(), m; rng=rng)
end
