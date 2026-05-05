# Tests for FunctionalModel — user-supplied closures.

using Distributions: Categorical, probs

@testset "construction & interface" begin
    A = [0.9 0.1; 0.1 0.9]
    B1 = [1.0 0.0; 0.0 1.0]   # action 1: identity
    B2 = [0.0 1.0; 1.0 0.0]   # action 2: swap
    Bs = [B1, B2]
    C = [1.0, -1.0]
    D = [0.5, 0.5]

    m = FunctionalModel(
        state_prior = () -> Categorical(D),
        observation_distribution = s -> Categorical(A[:, s]),
        transition_distribution  = (s, a) -> Categorical(Bs[a][:, s]),
        log_preferences          = o -> C[o],
        action_space             = () -> [1, 2],
        predict_states           = (q, a) -> Categorical(Bs[a] * probs(q))
    )

    sp = state_prior(m)
    @test probs(sp) == D
    @test probs(observation_distribution(m, 1)) == A[:, 1]
    @test probs(transition_distribution(m, 2, 1)) == B1[:, 2]
    @test log_preferences(m, 1) == 1.0
    @test collect(action_space(m)) == [1, 2]

    qs_pred = predict_states(m, Categorical([1.0, 0.0]), 2)
    @test probs(qs_pred) == [0.0, 1.0]   # swap from state 1 → state 2
end

@testset "FunctionalModel passes GenerativeModel conformance" begin
    A = [0.9 0.1; 0.1 0.9]
    B1 = [1.0 0.0; 0.0 1.0]
    B2 = [0.0 1.0; 1.0 0.0]
    Bs = [B1, B2]

    m = FunctionalModel(
        state_prior = () -> Categorical([0.5, 0.5]),
        observation_distribution = s -> Categorical(A[:, s]),
        transition_distribution  = (s, a) -> Categorical(Bs[a][:, s]),
        log_preferences          = o -> [0.0, 0.0][o],
        action_space             = () -> [1, 2],
        predict_states           = (q, a) -> Categorical(Bs[a] * probs(q))
    )

    Aifc.Testing.test_generative_model(m; rng=Xoshiro(42))
end

@testset "goal_state_prior support" begin
    A = [1.0 0.0; 0.0 1.0]
    B = [1.0 0.0; 0.0 1.0]
    m = FunctionalModel(
        state_prior = () -> Categorical([0.5, 0.5]),
        observation_distribution = s -> Categorical(A[:, s]),
        transition_distribution  = (s, a) -> Categorical(B[:, s]),
        log_preferences          = o -> 0.0,
        action_space             = () -> [1],
        predict_states           = (q, a) -> q,
        goal_state_prior         = Categorical([0.0, 1.0])
    )
    @test goal_state_prior(m) isa Categorical
    @test probs(goal_state_prior(m)) == [0.0, 1.0]
end
