# Tests for DiscretePOMDP construction and the GenerativeModel interface.

using Distributions: Categorical, probs

@testset "construction & validation" begin
    A = [0.7 0.2; 0.3 0.8]
    B = reshape([1.0 0.0  0.0 1.0;
                 0.0 1.0  1.0 0.0], 2, 2, 2)   # 2 actions, identity & swap
    C = [1.0, 0.0]
    D = [0.5, 0.5]

    m = DiscretePOMDP(A, B, C, D)
    @test m isa DiscretePOMDP
    @test nstates(m) == 2
    @test nobservations(m) == 2
    @test nactions(m) == 2

    # Reject mis-shaped B
    B_bad = reshape([1.0, 0.0, 0.5, 0.5], 2, 2, 1)   # only 1 action — but A doesn't constrain
    @test DiscretePOMDP(A, B_bad, C, D) isa DiscretePOMDP   # OK, just 1 action

    # Reject non-stochastic A
    A_bad = [0.7 0.2; 0.5 0.8]   # column 1 sums to 1.2
    @test_throws ArgumentError DiscretePOMDP(A_bad, B, C, D)

    # Reject D not summing to 1
    @test_throws ArgumentError DiscretePOMDP(A, B, C, [0.3, 0.5])

    # check=false skips validation
    @test DiscretePOMDP(A_bad, B, C, D; check=false) isa DiscretePOMDP

    # Goal validation
    @test DiscretePOMDP(A, B, C, D; goal=[0.4, 0.6]) isa DiscretePOMDP
    @test_throws ArgumentError DiscretePOMDP(A, B, C, D; goal=[0.5, 0.4])
end

@testset "GenerativeModel interface" begin
    A = [0.9 0.1; 0.1 0.9]
    B = reshape([1.0 0.0  0.0 1.0;
                 0.0 1.0  1.0 0.0], 2, 2, 2)
    C = [2.0, -1.0]
    D = [0.6, 0.4]

    m = DiscretePOMDP(A, B, C, D)

    # state_prior
    sp = state_prior(m)
    @test sp isa Categorical
    @test probs(sp) == D

    # observation_distribution
    od1 = observation_distribution(m, 1)
    @test probs(od1) == [0.9, 0.1]
    od2 = observation_distribution(m, 2)
    @test probs(od2) == [0.1, 0.9]
    @test_throws BoundsError observation_distribution(m, 3)

    # transition_distribution
    td_a1_s1 = transition_distribution(m, 1, 1)
    @test probs(td_a1_s1) == [1.0, 0.0]    # B[:,1,1]
    td_a2_s1 = transition_distribution(m, 1, 2)
    @test probs(td_a2_s1) == [0.0, 1.0]    # B[:,1,2]

    # log_preferences
    @test log_preferences(m, 1) == 2.0
    @test log_preferences(m, 2) == -1.0

    # action_space
    @test collect(action_space(m)) == [1, 2]
end

@testset "predict_states" begin
    A = [0.9 0.1; 0.1 0.9]
    # Action 1 = identity, action 2 = swap
    B = zeros(2, 2, 2)
    B[1, 1, 1] = 1; B[2, 2, 1] = 1
    B[2, 1, 2] = 1; B[1, 2, 2] = 1
    C = [0.0, 0.0]
    D = [0.7, 0.3]

    m = DiscretePOMDP(A, B, C, D)
    prior = Categorical([0.7, 0.3])

    # Action 1 (identity): unchanged
    @test probs(predict_states(m, prior, 1)) ≈ [0.7, 0.3]

    # Action 2 (swap): flipped
    @test probs(predict_states(m, prior, 2)) ≈ [0.3, 0.7]

    # Accepts raw vector
    @test probs(predict_states(m, [0.5, 0.5], 1)) ≈ [0.5, 0.5]
end

@testset "random_pomdp" begin
    rng = Xoshiro(42)
    m = random_pomdp(4, 6, 3; rng=rng)
    @test nobservations(m) == 4
    @test nstates(m) == 6
    @test nactions(m) == 3

    # A column-stochastic
    for s in 1:6
        @test sum(m.A[:, s]) ≈ 1
    end
    # B column-stochastic per action
    for a in 1:3, s in 1:6
        @test sum(m.B[:, s, a]) ≈ 1
    end

    # D uniform
    @test all(m.D .≈ 1/6)

    # C zeros (no preferences)
    @test all(iszero, m.C)
end

@testset "goal_state_prior" begin
    A = [0.5 0.5; 0.5 0.5]
    B = ones(2, 2, 1) ./ 2
    C = zeros(2)
    D = [0.5, 0.5]
    m = DiscretePOMDP(A, B, C, D)
    @test goal_state_prior(m) === nothing

    m_goal = DiscretePOMDP(A, B, C, D; goal=[0.0, 1.0])
    g = goal_state_prior(m_goal)
    @test g isa Categorical
    @test probs(g) == [0.0, 1.0]
end

@testset "Conformance" begin
    # Pass the GenerativeModel conformance suite
    rng = Xoshiro(7)
    m = random_pomdp(3, 4, 2; rng=Xoshiro(11))
    Aifc.Testing.test_generative_model(m; rng=rng)
end
