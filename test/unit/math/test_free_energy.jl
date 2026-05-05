# Tests for variational_free_energy / accuracy / complexity.
#
# Coverage:
#   1. Decomposition: F = -accuracy + complexity (always)
#   2. Gibbs' inequality: F ≥ -log P(o), equality at exact posterior
#   3. complexity(q, log_prior) ≡ kl_divergence(q, exp.(log_prior))
#   4. Hand-computed special cases
#   5. DimensionMismatch on size disagreement

@testset "decomposition: F = -accuracy + complexity" begin
    rng = Xoshiro(0xCAFE)
    for _ in 1:100
        n = rand(rng, 2:20)
        q = softmax(randn(rng, n))
        log_lik = randn(rng, n)
        log_prior = logsoftmax(randn(rng, n))   # normalized prior (in log domain)

        F = variational_free_energy(q, log_lik, log_prior)
        a = accuracy(q, log_lik)
        c = complexity(q, log_prior)

        @test F ≈ -a + c atol=1e-10
    end
end

@testset "Gibbs' inequality: F ≥ -log P(o)" begin
    # The exact posterior q*[s] ∝ P(o|s) P(s) achieves F = -log P(o).
    # Any other q has F ≥ -log P(o), with strict inequality unless q = q*.
    rng = Xoshiro(0x6125)
    for _ in 1:50
        n = rand(rng, 2:10)
        log_prior = logsoftmax(randn(rng, n))
        log_lik = randn(rng, n)

        # Exact posterior: log q* = log_lik + log_prior - log P(o)
        log_joint = log_lik .+ log_prior
        log_evidence = _logsumexp_test(log_joint)  # = log P(o)
        q_star = exp.(log_joint .- log_evidence)
        @test sum(q_star) ≈ 1 atol=1e-12   # sanity

        F_star = variational_free_energy(q_star, log_lik, log_prior)
        @test F_star ≈ -log_evidence atol=1e-10

        # F at any other q is >= F_star (within numerical tolerance)
        for _ in 1:5
            q_other = softmax(randn(rng, n))
            F_other = variational_free_energy(q_other, log_lik, log_prior)
            @test F_other >= F_star - 1e-10
        end
    end
end

@testset "complexity ≡ KL(q, exp.(log_prior))" begin
    rng = Xoshiro(0x12321)
    for _ in 1:50
        n = rand(rng, 2:20)
        q = softmax(randn(rng, n))
        log_prior = logsoftmax(randn(rng, n))
        prior = exp.(log_prior)
        @test complexity(q, log_prior) ≈ kl_divergence(q, prior) atol=1e-10
    end
end

@testset "accuracy and complexity individually" begin
    # Hand-computed.
    # If q = δ_1 (mass on state 1), accuracy = log P(o|s_1).
    log_lik = log.([0.7, 0.3])
    @test accuracy([1.0, 0.0], log_lik) ≈ log(0.7)
    @test accuracy([0.0, 1.0], log_lik) ≈ log(0.3)
    @test accuracy([0.5, 0.5], log_lik) ≈ 0.5*log(0.7) + 0.5*log(0.3)

    # If q = prior, complexity = 0
    log_prior = log.([0.4, 0.6])
    prior = exp.(log_prior)
    @test complexity(prior, log_prior) ≈ 0 atol=1e-12

    # If q = δ_1 with prior(1) = 0.4, complexity = log(1/0.4) = -log(0.4)
    @test complexity([1.0, 0.0], log_prior) ≈ -log(0.4)
end

@testset "edge cases" begin
    # Delta posterior on state 1 with non-trivial likelihood and prior.
    log_prior = log.([0.5, 0.5])
    log_lik = log.([0.7, 0.3])
    q = [1.0, 0.0]
    # F = -log(0.7) + log(1.0/0.5) = log(2) - log(0.7)
    @test variational_free_energy(q, log_lik, log_prior) ≈ log(2) - log(0.7) atol=1e-12

    # Posterior == prior, perfect likelihood: F = -E_q[log P(o|s)]
    log_prior_unif = log.([0.5, 0.5])
    log_lik_certain = log.([1.0, 1.0])      # uninformative likelihood
    q_unif = [0.5, 0.5]
    F = variational_free_energy(q_unif, log_lik_certain, log_prior_unif)
    @test F ≈ 0 atol=1e-12   # accuracy = 0, complexity = 0
end

@testset "DimensionMismatch" begin
    @test_throws DimensionMismatch variational_free_energy(
        [0.5, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0])
    @test_throws DimensionMismatch accuracy([0.5, 0.5], [0.0, 0.0, 0.0])
    @test_throws DimensionMismatch complexity([0.5, 0.5], [0.0, 0.0, 0.0])
end
