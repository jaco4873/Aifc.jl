# Tests for AgentHistory accessor methods.

using Distributions: Categorical, probs

@testset "AgentHistory accessors on empty history" begin
    h = AgentHistory(Categorical([0.5, 0.5]))
    @test length(h) == 0
    @test isempty(h)
    @test current_belief(h) == Categorical([0.5, 0.5])
    @test last_action(h) === nothing
    @test isempty(observation_history(h))
    @test isempty(action_history(h))
    @test isempty(free_energy_history(h))
    # belief_history includes initial_belief
    @test length(belief_history(h)) == 1
end

@testset "AgentHistory accessors after several steps" begin
    h = AgentHistory(Categorical([0.5, 0.5]))
    push_step!(h, AgentStep(1, Categorical([0.7, 0.3]),
                              [[1]], [1.0], [0.0], 0.5, 1))
    push_step!(h, AgentStep(2, Categorical([0.4, 0.6]),
                              [[2]], [1.0], [0.0], 0.3, 2))
    push_step!(h, AgentStep(1, Categorical([0.6, 0.4]),
                              [[1]], [1.0], [0.0], 0.8, 1))

    @test length(h) == 3
    @test !isempty(h)
    @test observation_history(h) == [1, 2, 1]
    @test action_history(h) == [1, 2, 1]
    @test free_energy_history(h) == [0.5, 0.3, 0.8]
    @test current_belief(h) == Categorical([0.6, 0.4])
    @test last_action(h) == 1

    # belief_history has initial + 3 step beliefs = 4 entries
    bh = belief_history(h)
    @test length(bh) == 4
    @test bh[1] == Categorical([0.5, 0.5])
    @test bh[end] == Categorical([0.6, 0.4])
end

@testset "AgentHistory iteration + indexing" begin
    h = AgentHistory(Categorical([0.5, 0.5]))
    push_step!(h, AgentStep(1, Categorical([0.7, 0.3]),
                              [[1]], [1.0], [0.0], 0.5, 1))
    push_step!(h, AgentStep(2, Categorical([0.4, 0.6]),
                              [[2]], [1.0], [0.0], 0.3, 2))

    # getindex
    @test h[1].observation == 1
    @test h[2].observation == 2
    @test h[lastindex(h)].observation == 2

    # iterate
    obs_collected = Int[]
    for step in h
        push!(obs_collected, step.observation)
    end
    @test obs_collected == [1, 2]
end
