# Integration tests for the ActionModels.jl package extension.
#
# Verifies that Aifc agents work as ActionModels submodels:
#   - Construction via active_inference_actionmodel
#   - init_agent / observe! / simulate! pathway
#   - get_parameters / set_parameters! introspection
#   - reset! returns the agent to its initial state
#   - History tracking through ActionModels' save_history mechanism
#
# Out of scope: Turing-based fitting (create_model + sample_posterior!).
# That requires propagating Dual numbers through our inference pipeline,
# which is a follow-up.

using ActionModels
using ActionModels: ActionModel, Parameter, init_agent, observe!, simulate!
using ActionModels: get_parameters, set_parameters!, get_states, reset!
using Distributions: Categorical, probs

@testset "active_inference_actionmodel construction" begin
    m = tmaze_model(cue_reliability=0.95, preference=4.0)
    am = active_inference_actionmodel(
        model = m,
        state_inference = FixedPointIteration(),
        policy_inference = SophisticatedInference(γ=8.0, horizon=2),
        action_selection = Stochastic(α=8.0),
    )

    @test am isa ActionModel

    # Parameters should expose action_precision and policy_precision
    @test :action_precision in keys(am.parameters)
    @test :policy_precision in keys(am.parameters)
    @test am.parameters.action_precision.value == 8.0
    @test am.parameters.policy_precision.value == 8.0
end

@testset "init_agent + observe! one step" begin
    m = tmaze_model()
    am = active_inference_actionmodel(
        model = m,
        state_inference = FixedPointIteration(),
        policy_inference = SophisticatedInference(γ=16.0, horizon=2),
        action_selection = Stochastic(α=16.0),
    )
    agent = init_agent(am; save_history=true)

    a = observe!(agent, 1)
    @test a in 1:4
end

@testset "simulate! over a trajectory" begin
    m = tmaze_model(cue_reliability=0.95, preference=4.0)
    am = active_inference_actionmodel(
        model = m,
        state_inference = FixedPointIteration(),
        policy_inference = SophisticatedInference(γ=16.0, horizon=2),
        action_selection = Stochastic(α=16.0),
    )
    agent = init_agent(am; save_history=true)

    obs_seq = [1, 2, 4]
    actions = simulate!(agent, obs_seq)

    @test length(actions) == 3
    @test all(a in 1:4 for a in actions)
end

@testset "get_parameters / set_parameters!" begin
    m = tmaze_model()
    am = active_inference_actionmodel(
        model = m,
        state_inference = FixedPointIteration(),
        policy_inference = EnumerativeEFE(γ=4.0, horizon=2),
        action_selection = Stochastic(α=4.0),
    )
    agent = init_agent(am)

    params = get_parameters(agent)
    @test params.action_precision == 4.0
    @test params.policy_precision == 4.0

    set_parameters!(agent, :action_precision, 16.0)
    @test get_parameters(agent, :action_precision) == 16.0

    set_parameters!(agent, (:action_precision, :policy_precision), (2.0, 2.0))
    p = get_parameters(agent)
    @test p.action_precision == 2.0
    @test p.policy_precision == 2.0
end

@testset "reset! returns to initial state" begin
    m = tmaze_model()
    am = active_inference_actionmodel(
        model = m,
        state_inference = FixedPointIteration(),
        policy_inference = EnumerativeEFE(γ=4.0, horizon=1),
        action_selection = Stochastic(α=4.0),
    )
    agent = init_agent(am; save_history=:free_energy)

    observe!(agent, 1)
    observe!(agent, 2)

    reset!(agent)
    # After reset!, free_energy state should be back to its initial 0.0
    @test get_states(agent, :free_energy) == 0.0
end

@testset "free_energy state is tracked across timesteps" begin
    m = tmaze_model()
    am = active_inference_actionmodel(
        model = m,
        state_inference = FixedPointIteration(),
        policy_inference = EnumerativeEFE(γ=4.0, horizon=1),
        action_selection = Stochastic(α=4.0),
    )
    agent = init_agent(am; save_history=:free_energy)

    observe!(agent, 1)
    observe!(agent, 2)

    F_history = agent.history.free_energy
    @test length(F_history) >= 2
    @test all(isfinite, F_history)
end

@testset "T-maze: ActionModels-driven agent picks go-cue at depth 2" begin
    # Reproduces our T-maze test through the ActionModels harness:
    # with strong preferences and sophisticated planning, the first action
    # at C should be go-cue.
    m = tmaze_model(cue_reliability=0.95, preference=4.0)
    am = active_inference_actionmodel(
        model = m,
        state_inference = FixedPointIteration(),
        policy_inference = SophisticatedInference(γ=64.0, horizon=2),
        action_selection = Stochastic(α=64.0),
    )
    agent = init_agent(am; save_history=true)

    a = observe!(agent, 1)
    @test a == 2   # go-cue
end

@testset "Parameter changes affect agent behavior" begin
    # With γ → 0 the policy posterior becomes uniform; over many trials, all
    # 4 actions should appear at least sometimes.
    m = tmaze_model(preference=4.0)
    am = active_inference_actionmodel(
        model = m,
        state_inference = FixedPointIteration(),
        policy_inference = EnumerativeEFE(γ=64.0, horizon=2),
        action_selection = Stochastic(α=1.0),
    )
    agent = init_agent(am)

    set_parameters!(agent, :policy_precision, 0.0)
    actions = Set{Int}()
    for _ in 1:50
        reset!(agent)
        push!(actions, observe!(agent, 1))
    end
    @test length(actions) >= 2
end
