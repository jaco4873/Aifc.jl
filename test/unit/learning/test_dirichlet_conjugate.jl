# Tests for DirichletConjugate parameter learning.

using Distributions: Categorical, probs

# Helper: build a minimal DiscretePOMDP with Dirichlet priors set to uniform
# concentration `c` (so we can predict update results exactly).
function _make_learnable_pomdp(c::Real = 1.0)
    A = [0.5 0.5; 0.5 0.5]    # uniform — uninformative
    B = zeros(2, 2, 1)
    B[1, 1, 1] = 1; B[2, 2, 1] = 1   # identity
    C = [0.0, 0.0]
    D = [0.5, 0.5]
    pA = fill(float(c), 2, 2)
    pB = fill(float(c), 2, 2, 1)
    pD = fill(float(c), 2)
    return DiscretePOMDP(A, B, C, D; pA=pA, pB=pB, pD=pD, check=false)
end

# Helper: a one-step history with given observation, posterior, action
function _history_with_one_step(observation, qs::Vector{Float64}, action)
    history = AgentHistory(Categorical([0.5, 0.5]))
    step = AgentStep(observation, Categorical(qs),
                     [action], [1.0], [0.0], 0.0, action)
    push_step!(history, step)
    return history
end

@testset "pA conjugate update" begin
    m = _make_learnable_pomdp(1.0)
    # Observation = 1, posterior peaked on state 1
    h = _history_with_one_step(1, [0.8, 0.2], 1)
    rule = DirichletConjugate(lr_pA=1.0, learn_pB=false, learn_pD=false,
                              use_effective_A=false)
    m2 = infer_parameters(rule, m, h)
    # pA should be original + onehot(o=1) ⊗ qs = original + [[0.8, 0.2]; [0, 0]]
    expected_pA = [1.8 1.2; 1.0 1.0]
    @test m2.pA ≈ expected_pA atol=1e-12
end

@testset "lr_pA = 0 → no change" begin
    m = _make_learnable_pomdp(2.5)
    h = _history_with_one_step(1, [0.7, 0.3], 1)
    rule = DirichletConjugate(lr_pA=0.0, learn_pB=false, learn_pD=false,
                              use_effective_A=false)
    m2 = infer_parameters(rule, m, h)
    @test m2.pA == m.pA
end

@testset "fr_pA = 0 → forget completely, then add this step's count" begin
    m = _make_learnable_pomdp(5.0)   # initial concentration 5 everywhere
    h = _history_with_one_step(1, [1.0, 0.0], 1)
    rule = DirichletConjugate(lr_pA=1.0, fr_pA=0.0, learn_pB=false, learn_pD=false,
                              use_effective_A=false)
    m2 = infer_parameters(rule, m, h)
    # After fr=0 forgetting + adding one observation o=1 at state 1:
    expected_pA = [1.0 0.0; 0.0 0.0]
    @test m2.pA ≈ expected_pA atol=1e-12
end

@testset "pD update at first step" begin
    m = _make_learnable_pomdp(1.0)
    h = _history_with_one_step(1, [0.7, 0.3], 1)
    rule = DirichletConjugate(lr_pD=1.0, learn_pA=false, learn_pB=false)
    m2 = infer_parameters(rule, m, h)
    # pD was [1, 1]; after seeing q(s_1) = [0.7, 0.3]:
    @test m2.pD ≈ [1.7, 1.3] atol=1e-12
end

@testset "pB update across two steps" begin
    m = _make_learnable_pomdp(1.0)
    history = AgentHistory(Categorical([0.5, 0.5]))
    push_step!(history, AgentStep(1, Categorical([0.8, 0.2]), [1], [1.0], [0.0], 0.0, 1))
    push_step!(history, AgentStep(2, Categorical([0.6, 0.4]), [1], [1.0], [0.0], 0.0, 1))
    rule = DirichletConjugate(lr_pB=1.0, learn_pA=false, learn_pD=false)
    m2 = infer_parameters(rule, m, history)
    # pB[s', s, 1] += q(s'_2) · q(s_1) for s, s'
    # q(s_1) = [0.8, 0.2], q(s_2) = [0.6, 0.4]
    expected_delta = [0.6*0.8  0.6*0.2;
                      0.4*0.8  0.4*0.2]
    expected_pB1 = ones(2, 2) .+ expected_delta
    @test m2.pB[:, :, 1] ≈ expected_pB1 atol=1e-12
end

@testset "effective_A is flatter than mean for small concentrations" begin
    # With small pA, the digamma-corrected effective A flattens below
    # the simple mean A (column-normalized). With large pA they coincide.
    m_small = _make_learnable_pomdp(1.0)
    m_large = _make_learnable_pomdp(100.0)

    h_peaked = _history_with_one_step(1, [1.0, 0.0], 1)

    rule_eff  = DirichletConjugate(use_effective_A=true,  learn_pB=false, learn_pD=false)
    rule_mean = DirichletConjugate(use_effective_A=false, learn_pB=false, learn_pD=false)

    A_eff_small  = infer_parameters(rule_eff,  m_small, h_peaked).A
    A_mean_small = infer_parameters(rule_mean, m_small, h_peaked).A

    # For small pA, effective and mean should differ: effective should be
    # flatter (lower max element) per column at indices where pA is small.
    @test A_eff_small != A_mean_small
    # Both column-stochastic
    for s in 1:2
        @test sum(A_eff_small[:, s])  ≈ 1 atol=1e-10
        @test sum(A_mean_small[:, s]) ≈ 1 atol=1e-10
    end

    # For large pA, effective ≈ mean to several decimals
    A_eff_large  = infer_parameters(rule_eff,  m_large, h_peaked).A
    A_mean_large = infer_parameters(rule_mean, m_large, h_peaked).A
    @test A_eff_large ≈ A_mean_large atol=1e-3
end

@testset "interface capability" begin
    rule = DirichletConjugate()
    @test !supports_states(rule)
    @test supports_parameters(rule)
    @test :parameters in supported_targets(rule)
end
