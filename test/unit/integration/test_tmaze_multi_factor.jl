# Cross-validation: 4×2 multi-factor T-maze ≡ 8-state single-factor T-maze.
#
# The two representations encode the same task:
#   - Single-factor: 8 states = (location, context) flattened
#   - Multi-factor:  2 factors of sizes (4, 2)
#
# State inference and free energy should agree EXACTLY (modulo factorization
# constraints), and the marginals over location and context should match.
# This is a strong correctness test: it verifies that the multi-factor FPI
# implementation matches the single-factor closed-form on a model where
# the true posterior happens to factorize (which it does for T-maze, since
# location is fully observable and context is independent).

using Distributions: Categorical, probs

# Map a single-factor 8-state belief to (location, context) marginals
function _sf_marginals(b::Categorical)
    p = probs(b)
    @assert length(p) == 8
    location = zeros(Float64, 4)
    context  = zeros(Float64, 2)
    for s in 1:8
        loc = (s - 1) ÷ 2 + 1
        ctx = (s - 1) % 2 + 1
        location[loc] += p[s]
        context[ctx]  += p[s]
    end
    return location, context
end

@testset "T-maze structural correspondence" begin
    sf = tmaze_model(cue_reliability=0.95, preference=4.0)
    mf = tmaze_model_multi_factor(cue_reliability=0.95, preference=4.0)

    @test nstates(sf) == nstates(mf) == 8
    @test nobservations(sf) == nobservations(mf) == 5
    @test state_factors(mf) == (4, 2)
    @test observation_modalities(mf) == (5,)
end

@testset "state_prior marginals match" begin
    sf = tmaze_model()
    mf = tmaze_model_multi_factor()

    sf_loc, sf_ctx = _sf_marginals(state_prior(sf))
    mf_loc = probs(marginal(state_prior(mf), 1))
    mf_ctx = probs(marginal(state_prior(mf), 2))

    @test sf_loc ≈ mf_loc atol=1e-12
    @test sf_ctx ≈ mf_ctx atol=1e-12
end

@testset "cue→L observation: posterior factorizes; marginals match" begin
    sf = tmaze_model()
    mf = tmaze_model_multi_factor()
    alg = FixedPointIteration(num_iter=20, dF_tol=1e-14)

    # Predict prior under [go-cue] (single-factor a=2; multi-factor [2, 1])
    prior_sf = predict_states(sf, state_prior(sf), TMAZE_ACTIONS.go_cue)
    prior_mf = predict_states(mf, state_prior(mf), [TMAZE_MF_ACTIONS.go_cue, 1])

    # Observe cue→L (obs index 2 in single-modality)
    q_sf = infer_states(alg, sf, prior_sf, 2)
    q_mf = infer_states(alg, mf, prior_mf, [2])

    sf_loc, sf_ctx = _sf_marginals(q_sf)
    mf_loc = probs(marginal(q_mf, 1))
    mf_ctx = probs(marginal(q_mf, 2))

    @test sf_loc ≈ mf_loc atol=1e-10
    @test sf_ctx ≈ mf_ctx atol=1e-10

    # The posterior should put high mass on ctx=L (since cue reliability=0.95)
    @test mf_ctx[1] ≈ 0.95 atol=1e-6
end

@testset "free energy at posterior matches across representations" begin
    sf = tmaze_model()
    mf = tmaze_model_multi_factor()
    alg = FixedPointIteration(num_iter=20, dF_tol=1e-14)

    prior_sf = predict_states(sf, state_prior(sf), TMAZE_ACTIONS.go_cue)
    prior_mf = predict_states(mf, state_prior(mf), [TMAZE_MF_ACTIONS.go_cue, 1])

    q_sf = infer_states(alg, sf, prior_sf, 2)
    q_mf = infer_states(alg, mf, prior_mf, [2])

    F_sf = free_energy(alg, sf, prior_sf, 2, q_sf)
    F_mf = free_energy(alg, mf, prior_mf, [2], q_mf)

    # The free energies should be EQUAL (mean-field is exact when the
    # posterior factorizes — which it does for the T-maze).
    @test F_sf ≈ F_mf atol=1e-9
end

@testset "trajectory: walk to Q then commit; marginals track" begin
    sf = tmaze_model()
    mf = tmaze_model_multi_factor()
    alg = FixedPointIteration(num_iter=20, dF_tol=1e-14)

    # Step 1: at C, observe neutral
    q_sf = infer_states(alg, sf, state_prior(sf), 1)
    q_mf = infer_states(alg, mf, state_prior(mf), [1])
    sf_loc, sf_ctx = _sf_marginals(q_sf)
    mf_loc = probs(marginal(q_mf, 1))
    mf_ctx = probs(marginal(q_mf, 2))
    @test sf_loc ≈ mf_loc atol=1e-10
    @test sf_ctx ≈ mf_ctx atol=1e-10

    # Step 2: take go-cue, then observe cue→L
    prior2_sf = predict_states(sf, q_sf, TMAZE_ACTIONS.go_cue)
    prior2_mf = predict_states(mf, q_mf, [TMAZE_MF_ACTIONS.go_cue, 1])
    q2_sf = infer_states(alg, sf, prior2_sf, 2)
    q2_mf = infer_states(alg, mf, prior2_mf, [2])

    sf_loc2, sf_ctx2 = _sf_marginals(q2_sf)
    mf_loc2 = probs(marginal(q2_mf, 1))
    mf_ctx2 = probs(marginal(q2_mf, 2))

    @test sf_loc2 ≈ mf_loc2 atol=1e-10
    @test sf_ctx2 ≈ mf_ctx2 atol=1e-10

    # The agent should be confident it's at Q with ctx=L
    @test mf_loc2[2] ≈ 1.0 atol=1e-9       # location = Q
    @test mf_ctx2[1] ≈ 0.95 atol=1e-6      # context = L
end
