# Multi-factor ActionModels.jl integration tests.
#
# Verifies that a multi-factor MultiFactorDiscretePOMDP wraps cleanly into
# an ActionModels.ActionModel with per-factor Action declarations and per-
# modality Observation declarations.

using ActionModels
using ActionModels: ActionModel, Parameter, init_agent, observe!, simulate!
using ActionModels: get_parameters, set_parameters!, get_states, reset!
using Distributions: Categorical, probs

@testset "multi-factor active_inference_actionmodel construction" begin
    m = tmaze_model_multi_factor(cue_reliability=0.95, preference=4.0)
    am = active_inference_actionmodel(
        model = m,
        state_inference = FixedPointIteration(),
        policy_inference = EnumerativeEFE(γ=8.0, horizon=2),
        action_selection = Stochastic(α=8.0),
    )
    @test am isa ActionModel

    # F=2 → 2 action declarations, M=1 → 1 observation declaration
    @test length(am.actions) == 2
    @test length(am.observations) == 1
    @test :action_1 in keys(am.actions)
    @test :action_2 in keys(am.actions)
    @test :observation_1 in keys(am.observations)
end

@testset "init_agent + observe! one step on multi-factor T-maze" begin
    m = tmaze_model_multi_factor(cue_reliability=0.95, preference=4.0)
    am = active_inference_actionmodel(
        model = m,
        state_inference = FixedPointIteration(),
        policy_inference = EnumerativeEFE(γ=64.0, horizon=2),
        action_selection = Stochastic(α=64.0),
    )
    agent = init_agent(am; save_history=true)

    # Single observation (M=1)
    actions = observe!(agent, 1)
    @test actions isa Tuple
    @test length(actions) == 2          # F=2
    @test 1 <= actions[1] <= 4          # location action
    @test actions[2] == 1               # context (only 1 option)
end

@testset "simulate! a 3-step trajectory through multi-factor T-maze" begin
    m = tmaze_model_multi_factor(cue_reliability=0.95, preference=4.0)
    am = active_inference_actionmodel(
        model = m,
        state_inference = FixedPointIteration(),
        policy_inference = EnumerativeEFE(γ=64.0, horizon=2),
        action_selection = Stochastic(α=64.0),
    )
    agent = init_agent(am; save_history=true)

    obs_seq = [1, 2, 4]                 # neutral, cue→L, reward
    actions = simulate!(agent, obs_seq)
    @test length(actions) == 3
    for a in actions
        @test a isa Tuple
        @test length(a) == 2
        @test a[2] == 1
    end
end

# NOTE: a "go-cue at depth 2" test would require multi-factor
# `SophisticatedInference` (deferred). With `EnumerativeEFE` the agent
# picks go-arm at horizon=2 since vanilla doesn't see the post-cue replan.

@testset "get_parameters / set_parameters! still work for multi-factor" begin
    m = tmaze_model_multi_factor()
    am = active_inference_actionmodel(
        model = m,
        state_inference = FixedPointIteration(),
        policy_inference = EnumerativeEFE(γ=4.0, horizon=2),
        action_selection = Stochastic(α=4.0),
    )
    agent = init_agent(am)

    p = get_parameters(agent)
    @test p.action_precision == 4.0
    @test p.policy_precision == 4.0

    set_parameters!(agent, :action_precision, 16.0)
    @test get_parameters(agent, :action_precision) == 16.0
end
