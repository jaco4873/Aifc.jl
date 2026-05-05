# Multi-factor DirichletConjugate tests.
#
# Per-factor / per-modality conjugate Dirichlet updates:
#   pA[m][o, s_1, ..., s_F] += lr · onehot(o, obs[m]) · ∏_f q_f(s_f)
#   pB[f][s', s, a_f]       += lr · q_f(s'_τ) · q_f(s_{τ-1})
#   pD[f][s_f]              += lr · q_f(s_1)

using Distributions: Categorical, probs

# Helper: build a 2-factor learnable model with uniform priors
function _make_learnable_mf_pomdp(; pA_concentration=1.0, pB_concentration=1.0,
                                     pD_concentration=1.0)
    # 2 state factors of sizes (2, 2); 1 modality of size 3
    A1 = zeros(3, 2, 2)
    for s1 in 1:2, s2 in 1:2
        A1[:, s1, s2] = [1/3, 1/3, 1/3]
    end
    B1 = zeros(2, 2, 1)
    B1[1, 1, 1] = 1; B1[2, 2, 1] = 1
    B2 = zeros(2, 2, 1)
    B2[1, 1, 1] = 1; B2[2, 2, 1] = 1
    C  = [zeros(3)]
    D  = [[0.5, 0.5], [0.5, 0.5]]

    pA = [fill(Float64(pA_concentration), 3, 2, 2)]
    pB = [fill(Float64(pB_concentration), 2, 2, 1),
          fill(Float64(pB_concentration), 2, 2, 1)]
    pD = [fill(Float64(pD_concentration), 2),
          fill(Float64(pD_concentration), 2)]

    return MultiFactorDiscretePOMDP([A1], [B1, B2], C, D;
                                      pA=pA, pB=pB, pD=pD, check=false)
end

# Helper: build a one-step history with given observation, belief, action
function _mf_history_one_step(observation::AbstractVector{<:Integer},
                                qs_factors::AbstractVector{<:AbstractVector{<:Real}},
                                action_vec::AbstractVector{<:Integer},
                                model::MultiFactorDiscretePOMDP)
    initial = state_prior(model)
    history = AgentHistory(initial)
    qs_dist = product_belief([Categorical(q) for q in qs_factors])
    step = AgentStep(observation, qs_dist,
                      [action_vec], [1.0], [0.0], 0.0, action_vec)
    push_step!(history, step)
    return history
end

# Helper: single-factor history with one step (parallel to _mf_history_one_step)
function _history_with_one_step_for_D(observation::Integer, qs::Vector{Float64},
                                        action::Integer, model::DiscretePOMDP)
    h = AgentHistory(state_prior(model))
    push_step!(h, AgentStep(observation, Categorical(qs),
                              [action], [1.0], [0.0], 0.0, action))
    return h
end

@testset "pA conjugate update: multi-factor / multi-modality" begin
    m = _make_learnable_mf_pomdp(pA_concentration=1.0)
    qs = [[0.7, 0.3], [0.4, 0.6]]
    h = _mf_history_one_step([2], qs, [1, 1], m)

    rule = DirichletConjugate(lr_pA=1.0, learn_pB=false, learn_pD=false,
                                use_effective_A=false)
    m2 = infer_parameters(rule, m, h)

    # Expected pA update for modality 1:
    #   pA[1][o=2, s_1, s_2] += 1.0 · q_1(s_1) · q_2(s_2)
    expected = ones(3, 2, 2)
    for s1 in 1:2, s2 in 1:2
        expected[2, s1, s2] += qs[1][s1] * qs[2][s2]
    end
    @test m2.pA[1] ≈ expected atol=1e-12

    # pB and pD untouched
    @test m2.pB == m.pB
    @test m2.pD == m.pD
end

@testset "pD update at first step: per-factor" begin
    m = _make_learnable_mf_pomdp(pD_concentration=1.0)
    qs = [[0.7, 0.3], [0.4, 0.6]]
    h = _mf_history_one_step([2], qs, [1, 1], m)

    rule = DirichletConjugate(lr_pD=1.0, learn_pA=false, learn_pB=false)
    m2 = infer_parameters(rule, m, h)

    # pD[f] += q_f(s_1)
    @test m2.pD[1] ≈ [1.7, 1.3] atol=1e-12
    @test m2.pD[2] ≈ [1.4, 1.6] atol=1e-12
end

@testset "pB update across two steps: per-factor" begin
    m = _make_learnable_mf_pomdp(pB_concentration=1.0)

    initial = state_prior(m)
    history = AgentHistory(initial)

    qs1 = [[0.8, 0.2], [0.5, 0.5]]
    a1  = [1, 1]
    push_step!(history, AgentStep([1],
        product_belief([Categorical(q) for q in qs1]),
        [a1], [1.0], [0.0], 0.0, a1))

    qs2 = [[0.6, 0.4], [0.3, 0.7]]
    a2  = [1, 1]
    push_step!(history, AgentStep([2],
        product_belief([Categorical(q) for q in qs2]),
        [a2], [1.0], [0.0], 0.0, a2))

    rule = DirichletConjugate(lr_pB=1.0, learn_pA=false, learn_pD=false)
    m2 = infer_parameters(rule, m, history)

    # pB[f][s', s, a] += q_f(s'_2) · q_f(s_1)   for action a = a1[f] = 1
    expected_pB1 = ones(2, 2, 1)
    for s1 in 1:2, s2 in 1:2
        expected_pB1[s2, s1, 1] += qs2[1][s2] * qs1[1][s1]
    end
    expected_pB2 = ones(2, 2, 1)
    for s1 in 1:2, s2 in 1:2
        expected_pB2[s2, s1, 1] += qs2[2][s2] * qs1[2][s1]
    end
    @test m2.pB[1] ≈ expected_pB1 atol=1e-12
    @test m2.pB[2] ≈ expected_pB2 atol=1e-12
end

@testset "lr=0 → no change (multi-factor)" begin
    m = _make_learnable_mf_pomdp(pA_concentration=2.5)
    qs = [[0.7, 0.3], [0.4, 0.6]]
    h = _mf_history_one_step([1], qs, [1, 1], m)

    rule = DirichletConjugate(lr_pA=0.0, learn_pB=false, learn_pD=false,
                                use_effective_A=false)
    m2 = infer_parameters(rule, m, h)
    @test m2.pA == m.pA
end

@testset "fr=0 forgets, then re-accumulates this step's count" begin
    m = _make_learnable_mf_pomdp(pA_concentration=5.0)
    qs = [[1.0, 0.0], [1.0, 0.0]]
    h = _mf_history_one_step([1], qs, [1, 1], m)

    rule = DirichletConjugate(lr_pA=1.0, fr_pA=0.0, learn_pB=false, learn_pD=false,
                                use_effective_A=false)
    m2 = infer_parameters(rule, m, h)

    # After fr=0 + lr=1 with q peaked: only pA[1, 1, 1] = 1, rest = 0
    expected = zeros(3, 2, 2)
    expected[1, 1, 1] = 1.0
    @test m2.pA[1] ≈ expected atol=1e-12
end

@testset "use_effective_A=true: multi-factor digamma correction" begin
    # Previously `use_effective_A=true` silently degraded to the Dirichlet
    # mean on multi-factor models (the digamma correction was implemented
    # only for single-factor). Closes that semantic gap.
    #
    # Asymptotically `digamma(x) - digamma(s) → log(x/s)` as concentrations
    # grow large, so effective and mean coincide at high concentration.
    # At low concentration the two differ — exactly the regime where the
    # parameter-uncertainty coupling matters (Friston/Da Costa 2020).

    m_small = _make_learnable_mf_pomdp(pA_concentration=1.0)
    m_large = _make_learnable_mf_pomdp(pA_concentration=100.0)

    qs = [[1.0, 0.0], [1.0, 0.0]]
    h_peaked       = _mf_history_one_step([1], qs, [1, 1], m_small)
    h_peaked_large = _mf_history_one_step([1], qs, [1, 1], m_large)

    rule_eff  = DirichletConjugate(use_effective_A=true,  learn_pB=false, learn_pD=false)
    rule_mean = DirichletConjugate(use_effective_A=false, learn_pB=false, learn_pD=false)

    A_eff_small  = infer_parameters(rule_eff,  m_small, h_peaked).A[1]
    A_mean_small = infer_parameters(rule_mean, m_small, h_peaked).A[1]

    # 1. At low concentration the two differ — the digamma correction is
    #    actually being computed (no longer silently the mean).
    @test A_eff_small != A_mean_small

    # 2. Both column-stochastic at every (s_1, s_2). This is what would
    #    silently break if the digamma correction were buggy at the
    #    multi-dim indexing.
    for s1 in 1:2, s2 in 1:2
        @test sum(A_eff_small[:, s1, s2])  ≈ 1 atol=1e-10
        @test sum(A_mean_small[:, s1, s2]) ≈ 1 atol=1e-10
    end

    # 3. At high concentration the two converge.
    A_eff_large  = infer_parameters(rule_eff,  m_large, h_peaked_large).A[1]
    A_mean_large = infer_parameters(rule_mean, m_large, h_peaked_large).A[1]
    @test A_eff_large ≈ A_mean_large atol=1e-3

    # 4. Multi-factor effective A matches the single-factor effective A
    #    for the trivial F=M=1 case (sanity that the multi-dim
    #    implementation isn't doing something different from the original).
    A_sf = [0.5 0.5; 0.5 0.5]
    B_sf = zeros(2, 2, 1); B_sf[1,1,1]=1; B_sf[2,2,1]=1
    pA_sf = [2.0  1.0;
             1.0  3.0]
    m_sf = DiscretePOMDP(A_sf, B_sf, [0.0, 0.0], [0.5, 0.5];
                          pA=pA_sf, check=false)
    m_mf = MultiFactorDiscretePOMDP(m_sf)

    h_sf = _history_with_one_step_for_D(1, [0.6, 0.4], 1, m_sf)
    h_mf = _mf_history_one_step([1], [[0.6, 0.4]], [1], m_mf)

    A_sf_eff = infer_parameters(rule_eff, m_sf, h_sf).A
    A_mf_eff = infer_parameters(rule_eff, m_mf, h_mf).A[1]
    @test A_sf_eff ≈ A_mf_eff atol=1e-12
end
