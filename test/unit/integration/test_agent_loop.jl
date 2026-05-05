# Integration tests for the full Agent.step! loop on DiscretePOMDP.

using Distributions: Categorical, probs

@testset "single step on a simple model" begin
    A = [0.9 0.1; 0.1 0.9]
    B = zeros(2, 2, 2)
    B[1,1,1]=1; B[2,2,1]=1
    B[2,1,2]=1; B[1,2,2]=1
    C = [2.0, -2.0]
    D = [0.5, 0.5]
    m = DiscretePOMDP(A, B, C, D)

    agent = Agent(m,
                  FixedPointIteration(),
                  EnumerativeEFE(γ=16.0, horizon=1),
                  Deterministic())

    # Observe obs=1 (preferred). Belief should peak on state 1.
    a = step!(agent, 1)
    qs = current_belief(agent.history)
    @test probs(qs)[1] > 0.85

    # Action should be 1 (identity) since we're already in preferred state
    @test a == 1
end

@testset "multi-step trajectory" begin
    # An agent that prefers obs 1 should converge to state 1 over time.
    A = [0.9 0.1; 0.1 0.9]
    B = zeros(2, 2, 2)
    B[1,1,1]=1; B[2,2,1]=1
    B[2,1,2]=1; B[1,2,2]=1
    C = [3.0, -3.0]
    D = [0.5, 0.5]
    m = DiscretePOMDP(A, B, C, D)

    agent = Agent(m,
                  FixedPointIteration(),
                  EnumerativeEFE(γ=16.0, horizon=2),
                  Deterministic())

    # Simulate environment: starts in state 2, agent has to switch via action 2
    true_state = 2
    for t in 1:3
        # Observation likelihood
        obs_probs = A[:, true_state]
        # Generate observation deterministically (most likely)
        obs = argmax(obs_probs)
        a = step!(agent, obs)
        # Apply action to true state
        true_state = argmax(B[:, true_state, a])
    end

    @test length(agent.history) == 3
    # By the end, the agent should believe it's in state 1
    @test probs(current_belief(agent.history))[1] > 0.5
end

@testset "reset!" begin
    m = random_pomdp(3, 4, 2; rng=Xoshiro(99))
    agent = Agent(m, FixedPointIteration(), EnumerativeEFE(), Stochastic())
    step!(agent, 1)
    step!(agent, 2)
    @test length(agent.history) == 2
    Aifc.reset!(agent)
    @test isempty(agent.history)
    @test current_belief(agent.history) == state_prior(m)
end

@testset "history fields populated" begin
    m = random_pomdp(3, 4, 2; rng=Xoshiro(11))
    agent = Agent(m, FixedPointIteration(), EnumerativeEFE(horizon=2), Stochastic())
    step!(agent, 1)

    s = agent.history.steps[1]
    @test s.observation == 1
    @test s.belief isa Categorical
    @test length(s.policies) == 4   # 2 actions × horizon 2
    @test length(s.policy_posterior) == 4
    @test sum(s.policy_posterior) ≈ 1
    @test length(s.expected_free_energies) == 4
    @test isfinite(s.free_energy)
    @test 1 <= s.action <= 2
end

@testset "agent loop with Dirichlet learning" begin
    A0 = [0.5 0.5; 0.5 0.5]   # Uninformative initial likelihood
    B = zeros(2, 2, 1)         # Single action, identity
    B[1, 1, 1] = 1; B[2, 2, 1] = 1
    C = [0.0, 0.0]
    D = [0.5, 0.5]
    pA = ones(2, 2)             # Weak Dirichlet prior
    m = DiscretePOMDP(A0, B, C, D; pA=pA, check=false)

    learner = DirichletConjugate(lr_pA=1.0, learn_pB=false, learn_pD=false,
                                  use_effective_A=false)
    agent = Agent(m, FixedPointIteration(), EnumerativeEFE(), Stochastic();
                  parameter_learning=learner, rng=Xoshiro(0))

    # Feed many observations of obs=1 — pA should accumulate counts
    for _ in 1:20
        step!(agent, 1)
    end

    # After learning, pA should have higher concentration on row 1
    @test agent.model.pA[1, 1] > agent.model.pA[2, 1]
    @test agent.model.pA[1, 2] > agent.model.pA[2, 2]
end

@testset "errors on broken inference (missing infer_states impl)" begin
    # An Inference subtype that forgets to implement infer_states should
    # error at first step!. (The constructor accepts any `<: Inference`;
    # method-level dispatch catches the missing implementation at use.)
    struct _NoOpInference <: Aifc.Inference end
    m = random_pomdp(2, 2, 2; rng=Xoshiro(0))
    agent = Agent(m, _NoOpInference(), EnumerativeEFE(), Stochastic())
    @test_throws ErrorException step!(agent, 1)
end
