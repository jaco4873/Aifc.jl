# Tests for Stochastic and Deterministic action selectors.

@testset "Stochastic: action_marginals" begin
    # 4 policies, 3 actions
    policies = [[1, 1], [2, 1], [3, 1], [1, 2]]
    q_pi = [0.4, 0.3, 0.2, 0.1]
    sel = Stochastic(α=1.0)
    p = exp.(log_action_probabilities(sel, q_pi, policies))
    # Marginal: action 1 = 0.4 + 0.1 = 0.5; action 2 = 0.3; action 3 = 0.2
    expected_marg = [0.5, 0.3, 0.2]
    # With α=1, p ≈ marginal (modulo eps stabilization)
    @test p ≈ expected_marg atol=1e-6
end

@testset "Stochastic: α → ∞ matches Deterministic" begin
    rng = Xoshiro(0xACE)
    for _ in 1:20
        n_actions = rand(rng, 2:5)
        n_policies = n_actions  # one policy per action
        policies = [[a] for a in 1:n_actions]
        q_pi = softmax(randn(rng, n_policies))

        sharp = Stochastic(α=1e6)
        det   = Deterministic()

        a_sharp = sample_action(sharp, q_pi, policies, Xoshiro(0))
        a_det   = sample_action(det,   q_pi, policies, Xoshiro(0))
        @test a_sharp == a_det
        @test a_det == argmax(q_pi)
    end
end

@testset "Deterministic: argmax of marginal" begin
    policies = [[1], [2], [3]]
    q_pi = [0.2, 0.7, 0.1]
    sel = Deterministic()
    @test sample_action(sel, q_pi, policies, Xoshiro(0)) == 2

    # Multi-step policies — only first action counts for marginal
    policies2 = [[1, 1, 1], [2, 1, 1], [2, 2, 2]]
    q_pi2 = [0.3, 0.4, 0.3]
    # marginal: action 1 = 0.3, action 2 = 0.7
    @test sample_action(sel, q_pi2, policies2, Xoshiro(0)) == 2
end

@testset "Stochastic: sampling distribution converges to p(a)" begin
    # Run many samples; verify empirical frequencies approach the action marginal
    policies = [[1], [2], [3]]
    q_pi = [0.5, 0.3, 0.2]
    sel = Stochastic(α=1.0)
    rng = Xoshiro(0xFF)
    counts = zeros(Int, 3)
    n = 50_000
    for _ in 1:n
        a = sample_action(sel, q_pi, policies, rng)
        counts[a] += 1
    end
    empirical = counts ./ n
    @test empirical ≈ q_pi atol=0.01
end

@testset "Conformance" begin
    policies = [[1], [2], [3]]
    q_pi = [0.4, 0.4, 0.2]

    Aifc.Testing.test_action_selector(Stochastic(α=4.0), q_pi, policies)
    Aifc.Testing.test_action_selector(Deterministic(), q_pi, policies)
end
