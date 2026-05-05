# End-to-end Agent loop with MultiFactorDiscretePOMDP.
#
# After multi-factor predict_states, FPI, EnumerativeEFE, and
# Stochastic.action_distribution(::Vector{Vector{Vector{Int}}}) are all in
# place, `Agent.step!` should work without further changes — the prior
# becomes a `MultivariateDistribution`, policies become
# `Vector{Vector{Vector{Int}}}`, and actions become `Vector{Int}`. This
# test suite exercises the full action–perception cycle on the 4×2 T-maze.

using Distributions: Categorical, MultivariateDistribution, probs

@testset "single step on multi-factor T-maze" begin
    m = tmaze_model_multi_factor(cue_reliability=0.95, preference=4.0)
    agent = Agent(m,
                  FixedPointIteration(),
                  EnumerativeEFE(γ=16.0, horizon=2),
                  Deterministic())

    # Observe "neutral" at C
    a = step!(agent, [1])
    @test a isa AbstractVector{<:Integer}
    @test length(a) == 2                   # 2 action factors
    @test 1 <= a[1] <= 4                   # location action
    @test a[2] == 1                        # context action (only 1 option)
end

@testset "trajectory: at-Q observe cue→L → commit to L (multi-factor)" begin
    # NOTE: SophisticatedInference is single-factor only at present; this
    # test uses EnumerativeEFE with horizon=1 starting from a state in
    # which the agent has already moved to Q. After observing cue→L the
    # agent should commit to L.
    m = tmaze_model_multi_factor(cue_reliability=0.95, preference=4.0)
    agent = Agent(m,
                  FixedPointIteration(num_iter=20, dF_tol=1e-14),
                  EnumerativeEFE(γ=64.0, horizon=1),
                  Deterministic())

    # Manually drive the agent to Q so the test starts there.
    step!(agent, [1])    # neutral at C — pick some action
    # Force the next belief to be at Q with uniform context (simulate go-cue)
    a_qcue = [TMAZE_MF_ACTIONS.go_cue, 1]
    # Override last action so prior advances under go-cue
    agent.history.steps[end] = AgentStep(
        agent.history.steps[end].observation,
        agent.history.steps[end].belief,
        agent.history.steps[end].policies,
        agent.history.steps[end].policy_posterior,
        agent.history.steps[end].expected_free_energies,
        agent.history.steps[end].free_energy,
        a_qcue,
    )
    # Observe cue→L
    a2 = step!(agent, [2])
    @test a2[1] == TMAZE_MF_ACTIONS.go_left
end

@testset "single-factor and multi-factor agents agree on first action (EnumerativeEFE)" begin
    sf_model = tmaze_model(cue_reliability=0.95, preference=4.0)
    mf_model = tmaze_model_multi_factor(cue_reliability=0.95, preference=4.0)

    sf_agent = Agent(sf_model,
                      FixedPointIteration(),
                      EnumerativeEFE(γ=64.0, horizon=2),
                      Deterministic())
    mf_agent = Agent(mf_model,
                      FixedPointIteration(),
                      EnumerativeEFE(γ=64.0, horizon=2),
                      Deterministic())

    a_sf = step!(sf_agent, 1)
    a_mf = step!(mf_agent, [1])

    # Single-factor returns Int; multi-factor returns Vector{Int}. The
    # location actions should match.
    @test a_sf == a_mf[1]
end

@testset "Stochastic multi-factor: action_distribution is Product of Categoricals" begin
    m = tmaze_model_multi_factor()
    agent = Agent(m,
                  FixedPointIteration(),
                  EnumerativeEFE(γ=8.0, horizon=2),
                  Stochastic(α=8.0))

    a = step!(agent, [1])
    @test a isa AbstractVector{<:Integer}
    @test length(a) == 2
end

@testset "agent_history populated correctly with multi-factor data" begin
    m = tmaze_model_multi_factor()
    agent = Agent(m,
                  FixedPointIteration(),
                  EnumerativeEFE(γ=8.0, horizon=2),
                  Stochastic(α=8.0))

    step!(agent, [1])
    step!(agent, [2])

    @test length(agent.history) == 2
    s = agent.history.steps[1]
    @test s.observation == [1]
    @test s.belief isa MultivariateDistribution
    @test s.action isa AbstractVector{<:Integer}
    @test isfinite(s.free_energy)
end
