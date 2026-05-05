# Multi-factor POMDP validation + edge-case tests.
#
# Targets the validation branches and convenience overloads in
# `src/models/multi_factor_pomdp.jl` that the existing tests don't
# exercise. Goal: push file coverage from 59% closer to 95%.

using Distributions: Categorical, probs

function _mf_random_col_stoch(shape::NTuple{N,Int}; rng=Xoshiro(0)) where N
    out = zeros(Float64, shape...)
    n_obs = shape[1]
    for idx in CartesianIndices(shape[2:end])
        out[:, idx] = softmax(randn(rng, n_obs))
    end
    return out
end

@testset "construction validation: number of factors mismatch" begin
    A = [_mf_random_col_stoch((4, 3))]      # 1 factor in A
    B1 = _mf_random_col_stoch((3, 3, 2))
    B2 = _mf_random_col_stoch((2, 2, 1))
    C  = [zeros(4)]
    D  = [fill(1/3, 3), fill(1/2, 2)]       # 2 factors in D — mismatch with A's rank
    @test_throws DimensionMismatch MultiFactorDiscretePOMDP(A, [B1, B2], C, D)

    # |B| != |D|
    A2 = [_mf_random_col_stoch((4, 3, 2))]
    @test_throws DimensionMismatch MultiFactorDiscretePOMDP(A2, [B1], C, D)

    # |A| != |C|
    Cmm = [zeros(4), zeros(2)]
    @test_throws DimensionMismatch MultiFactorDiscretePOMDP(A2, [B1, B2], Cmm, D)
end

@testset "construction validation: non-stochastic" begin
    A = _mf_random_col_stoch((4, 3, 2))
    B1 = _mf_random_col_stoch((3, 3, 2))
    B2 = _mf_random_col_stoch((2, 2, 1))
    C  = [zeros(4)]
    D  = [fill(1/3, 3), fill(1/2, 2)]

    # B with non-stochastic column
    B_bad = copy(B1)
    B_bad[:, 1, 1] .*= 1.5
    @test_throws ArgumentError MultiFactorDiscretePOMDP([A], [B_bad, B2], C, D)

    # check=false skips validation
    @test MultiFactorDiscretePOMDP([A], [B_bad, B2], C, D; check=false) isa MultiFactorDiscretePOMDP

    # D with non-normalized factor (right shape, wrong sum)
    D_bad = [[0.4, 0.4, 0.1], [0.5, 0.5]]   # first factor sums to 0.9
    @test_throws ArgumentError MultiFactorDiscretePOMDP([A], [B1, B2], C, D_bad)
end

@testset "construction validation: optional fields shape mismatch" begin
    A = _mf_random_col_stoch((4, 3, 2))
    B1 = _mf_random_col_stoch((3, 3, 2))
    B2 = _mf_random_col_stoch((2, 2, 1))
    C  = [zeros(4)]
    D  = [fill(1/3, 3), fill(1/2, 2)]

    # goal of wrong number of factors
    goal_wrong_count = [[0.0, 1.0]]   # only 1 factor; D has 2
    @test_throws DimensionMismatch MultiFactorDiscretePOMDP(
        [A], [B1, B2], C, D; goal=goal_wrong_count)

    # goal factor of wrong size
    goal_wrong_size = [[0.0, 0.5, 0.5], [0.5, 0.5, 0.0]]   # second factor has 3 entries
    @test_throws DimensionMismatch MultiFactorDiscretePOMDP(
        [A], [B1, B2], C, D; goal=goal_wrong_size)

    # goal not summing to 1
    goal_unnorm = [[0.5, 0.4, 0.0], [0.5, 0.5]]
    @test_throws ArgumentError MultiFactorDiscretePOMDP(
        [A], [B1, B2], C, D; goal=goal_unnorm)
end

@testset "convenience overloads: scalar observation/action for F=M=1" begin
    A = _mf_random_col_stoch((4, 3))
    B = _mf_random_col_stoch((3, 3, 2))
    C = zeros(4)
    D = fill(1/3, 3)
    sf = DiscretePOMDP(A, B, C, D)
    mf = MultiFactorDiscretePOMDP(sf)        # lift F=M=1

    # Scalar overloads delegate to vector overloads
    od = observation_distribution(mf, 2)
    @test od isa Categorical

    td = transition_distribution(mf, 2, 1)
    @test td isa Categorical

    @test log_preferences(mf, 1) == C[1]

    # Errors on scalar input when F>1 or M>1
    A2 = _mf_random_col_stoch((4, 3, 2))
    B1 = _mf_random_col_stoch((3, 3, 2))
    B2 = _mf_random_col_stoch((2, 2, 1))
    mf2 = MultiFactorDiscretePOMDP([A2], [B1, B2], [zeros(4)],
                                     [fill(1/3, 3), fill(1/2, 2)])
    @test_throws ArgumentError observation_distribution(mf2, 2)
    @test_throws ArgumentError transition_distribution(mf2, 2, 1)
end

@testset "predict_states overloads (Categorical input)" begin
    sf = DiscretePOMDP([0.9 0.1; 0.1 0.9],
                        let B = zeros(2, 2, 1); B[1,1,1] = 1; B[2,2,1] = 1; B end,
                        [0.0, 0.0], [0.5, 0.5])
    mf = MultiFactorDiscretePOMDP(sf)

    # F=1 with Categorical prior, vector action
    qs = predict_states(mf, Categorical([0.5, 0.5]), [1])
    @test qs isa Categorical

    # F=1 with Categorical prior, scalar action
    qs2 = predict_states(mf, Categorical([0.5, 0.5]), 1)
    @test qs2 isa Categorical
    @test probs(qs) == probs(qs2)

    # Scalar action requires F=1
    A2 = _mf_random_col_stoch((4, 3, 2))
    B1 = _mf_random_col_stoch((3, 3, 2))
    B2 = _mf_random_col_stoch((2, 2, 1))
    mf2 = MultiFactorDiscretePOMDP([A2], [B1, B2], [zeros(4)],
                                     [fill(1/3, 3), fill(1/2, 2)])
    @test_throws ArgumentError predict_states(mf2, Categorical([0.5, 0.5]), 1)
end

@testset "transition_distribution dimension mismatch" begin
    A = _mf_random_col_stoch((4, 3, 2))
    B1 = _mf_random_col_stoch((3, 3, 2))
    B2 = _mf_random_col_stoch((2, 2, 1))
    mf = MultiFactorDiscretePOMDP([A], [B1, B2], [zeros(4)],
                                    [fill(1/3, 3), fill(1/2, 2)])

    # Wrong-length state vector
    @test_throws DimensionMismatch transition_distribution(mf, [1], [1, 1])
    # Wrong-length action vector
    @test_throws DimensionMismatch transition_distribution(mf, [1, 1], [1])
end

@testset "log_preferences dimension mismatch + scalar requires M=1" begin
    A1 = _mf_random_col_stoch((4, 3))
    A2 = _mf_random_col_stoch((2, 3))
    B = _mf_random_col_stoch((3, 3, 2))
    mf = MultiFactorDiscretePOMDP([A1, A2], [B], [zeros(4), zeros(2)], [fill(1/3, 3)])

    # Scalar observation requires M=1
    @test_throws ArgumentError log_preferences(mf, 1)

    # Vector observation of wrong length
    @test_throws DimensionMismatch log_preferences(mf, [1, 1, 1])
end

@testset "observation_distribution dimension mismatch" begin
    A = _mf_random_col_stoch((4, 3, 2))
    B1 = _mf_random_col_stoch((3, 3, 2))
    B2 = _mf_random_col_stoch((2, 2, 1))
    mf = MultiFactorDiscretePOMDP([A], [B1, B2], [zeros(4)],
                                    [fill(1/3, 3), fill(1/2, 2)])

    # Wrong-length state vector
    @test_throws DimensionMismatch observation_distribution(mf, [1])
end

@testset "DiscretePOMDP(::MultiFactorDiscretePOMDP): refuses non-trivial cases" begin
    A = _mf_random_col_stoch((4, 3, 2))
    B1 = _mf_random_col_stoch((3, 3, 2))
    B2 = _mf_random_col_stoch((2, 2, 1))
    mf = MultiFactorDiscretePOMDP([A], [B1, B2], [zeros(4)],
                                    [fill(1/3, 3), fill(1/2, 2)])
    @test_throws ArgumentError DiscretePOMDP(mf)

    A1 = _mf_random_col_stoch((4, 3))
    A2 = _mf_random_col_stoch((2, 3))
    mf_mm = MultiFactorDiscretePOMDP([A1, A2], [_mf_random_col_stoch((3, 3, 2))],
                                       [zeros(4), zeros(2)], [fill(1/3, 3)])
    @test_throws ArgumentError DiscretePOMDP(mf_mm)
end

@testset "action_space: F=1 returns range; F>1 returns CartesianIndices" begin
    A = _mf_random_col_stoch((4, 3))
    B = _mf_random_col_stoch((3, 3, 2))
    mf = MultiFactorDiscretePOMDP([A], [B], [zeros(4)], [fill(1/3, 3)])
    space = action_space(mf)
    @test collect(space) == [[1], [2]]    # F=1 wraps each scalar action in a vector

    A2 = _mf_random_col_stoch((4, 3, 2))
    B1 = _mf_random_col_stoch((3, 3, 2))
    B2 = _mf_random_col_stoch((2, 2, 1))
    mf2 = MultiFactorDiscretePOMDP([A2], [B1, B2], [zeros(4)],
                                     [fill(1/3, 3), fill(1/2, 2)])
    space2 = action_space(mf2)
    @test length(collect(space2)) == 2 * 1   # 2 location actions × 1 context action
end
