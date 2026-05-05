# Tests for the *incremental* / per-step semantics of `infer_parameters`,
# which is how `Agent.step!` calls it.
#
# Bug being guarded against: the previous implementation rebuilt the
# Dirichlet posterior by iterating the *entire* history on each call,
# starting from `m.pA` (the already-updated posterior). When `Agent.step!`
# called it once per step, each step's contribution got re-added every
# subsequent step, producing
#
#     pA[T] = pA_0 + T·step_1 + (T-1)·step_2 + … + step_T
#
# instead of the correct
#
#     pA[T] = pA_0 + step_1 + step_2 + … + step_T.
#
# The same multi-counting hits pB and pD.
#
# These tests pin down the correct behavior by simulating Agent.step!'s
# call pattern directly: call `infer_parameters` on each step's history
# slice in succession, threading the returned model.

using Distributions: Categorical, probs

function _make_learnable_pomdp_2state(c::Real = 1.0)
    A = [0.5 0.5; 0.5 0.5]
    B = zeros(2, 2, 1)
    B[1, 1, 1] = 1; B[2, 2, 1] = 1
    C = [0.0, 0.0]
    D = [0.5, 0.5]
    pA = fill(float(c), 2, 2)
    pB = fill(float(c), 2, 2, 1)
    pD = fill(float(c), 2)
    return DiscretePOMDP(A, B, C, D; pA=pA, pB=pB, pD=pD, check=false)
end

# Drive infer_parameters the way Agent.step! does: append one step at a
# time, call infer_parameters after each push, thread the returned model.
function _drive_per_step(alg, m, steps)
    history = AgentHistory(Categorical([0.5, 0.5]))
    cur = m
    for step in steps
        push_step!(history, step)
        cur = infer_parameters(alg, cur, history)
    end
    return cur, history
end

@testset "pA: per-step replay does not double-count earlier steps" begin
    m = _make_learnable_pomdp_2state(1.0)
    rule = DirichletConjugate(lr_pA=1.0, fr_pA=1.0, learn_pB=false, learn_pD=false,
                                use_effective_A=false)

    # Two distinct steps with peaked posteriors, so the contribution is unique.
    s1 = AgentStep(1, Categorical([1.0, 0.0]), [1], [1.0], [0.0], 0.0, 1)
    s2 = AgentStep(2, Categorical([0.0, 1.0]), [1], [1.0], [0.0], 0.0, 1)

    cur, _ = _drive_per_step(rule, m, [s1, s2])

    # Correct conjugate update: each step contributes ONCE.
    # Step 1: pA[1, 1] += 1, pA[1, 2] += 0
    # Step 2: pA[2, 1] += 0, pA[2, 2] += 1
    expected_pA = [2.0  1.0;
                   1.0  2.0]
    @test cur.pA ≈ expected_pA atol=1e-12
end

@testset "pA: forgetting compounds correctly across per-step calls" begin
    # With fr<1 the recurrence is pA[t] = fr·pA[t-1] + lr·count_t.
    m = _make_learnable_pomdp_2state(0.0)   # start from zero so we can see the recurrence cleanly
    rule = DirichletConjugate(lr_pA=1.0, fr_pA=0.5, learn_pB=false, learn_pD=false,
                                use_effective_A=false)

    s1 = AgentStep(1, Categorical([1.0, 0.0]), [1], [1.0], [0.0], 0.0, 1)
    s2 = AgentStep(1, Categorical([1.0, 0.0]), [1], [1.0], [0.0], 0.0, 1)
    s3 = AgentStep(1, Categorical([1.0, 0.0]), [1], [1.0], [0.0], 0.0, 1)

    cur, _ = _drive_per_step(rule, m, [s1, s2, s3])

    # Manual recurrence at pA[1, 1]:
    #   t=1: 0.5·0 + 1 = 1
    #   t=2: 0.5·1 + 1 = 1.5
    #   t=3: 0.5·1.5 + 1 = 1.75
    @test cur.pA[1, 1] ≈ 1.75 atol=1e-12
end

@testset "pB: per-step replay does not double-count earlier transitions" begin
    m = _make_learnable_pomdp_2state(1.0)
    rule = DirichletConjugate(learn_pA=false, learn_pB=true, learn_pD=false, lr_pB=1.0, fr_pB=1.0)

    # Three steps: q(s_1)=[1,0], q(s_2)=[0,1], q(s_3)=[1,0], action 1 throughout.
    s1 = AgentStep(1, Categorical([1.0, 0.0]), [1], [1.0], [0.0], 0.0, 1)
    s2 = AgentStep(1, Categorical([0.0, 1.0]), [1], [1.0], [0.0], 0.0, 1)
    s3 = AgentStep(1, Categorical([1.0, 0.0]), [1], [1.0], [0.0], 0.0, 1)

    cur, _ = _drive_per_step(rule, m, [s1, s2, s3])

    # Two transitions: (s_1=1 → s_2=2) and (s_2=2 → s_3=1).
    # pB[s', s, a=1] should pick up exactly:
    #   pB[2, 1, 1] += 1   (from transition 1→2)
    #   pB[1, 2, 1] += 1   (from transition 2→1)
    expected_delta = [0.0  1.0;
                      1.0  0.0]
    expected_pB1 = ones(2, 2) .+ expected_delta
    @test cur.pB[:, :, 1] ≈ expected_pB1 atol=1e-12
end

@testset "pD: only updates on the very first step" begin
    m = _make_learnable_pomdp_2state(1.0)
    rule = DirichletConjugate(learn_pA=false, learn_pB=false, learn_pD=true,
                                lr_pD=1.0, fr_pD=1.0)

    s1 = AgentStep(1, Categorical([0.7, 0.3]), [1], [1.0], [0.0], 0.0, 1)
    s2 = AgentStep(2, Categorical([0.4, 0.6]), [1], [1.0], [0.0], 0.0, 1)
    s3 = AgentStep(1, Categorical([0.5, 0.5]), [1], [1.0], [0.0], 0.0, 1)

    cur, _ = _drive_per_step(rule, m, [s1, s2, s3])

    # pD should be original + q(s_1) once. Subsequent steps don't touch it
    # (the prior over the *initial* state can't be re-evidenced by later steps).
    @test cur.pD ≈ [1.7, 1.3] atol=1e-12
end

@testset "Agent.step! preserves correct DirichletConjugate semantics" begin
    # End-to-end: actually run Agent.step! a few times and check pA.
    using Random: Xoshiro
    m = _make_learnable_pomdp_2state(1.0)
    agent = Agent(m,
                  FixedPointIteration(),
                  EnumerativeEFE(γ=4.0, horizon=1),
                  Stochastic(α=4.0);
                  parameter_learning = DirichletConjugate(lr_pA=1.0, fr_pA=1.0,
                                                            learn_pB=false, learn_pD=false,
                                                            use_effective_A=false),
                  rng = Xoshiro(0))

    obs_seq = [1, 2, 1, 2]
    for o in obs_seq
        step!(agent, o)
    end

    # After 4 steps the trace of pA should equal the trace of m.pA plus the
    # number of steps (each step's q(s) is a probability vector, so
    # Σ_o pA_delta[o, s] sums to q(s) per state, and across states sums to 1).
    @test sum(agent.model.pA) ≈ sum(m.pA) + length(obs_seq) atol=1e-10
end
