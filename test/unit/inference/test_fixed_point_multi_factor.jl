# Multi-factor FixedPointIteration tests.
#
# Properties verified:
#   1. F=1 multi-factor inference matches single-factor closed-form (sanity)
#   2. F>1 mean-field iteration: variational free energy decreases monotonically
#   3. F>1 mean-field on a factorized-posterior model: converges to the exact
#      posterior (since mean-field is exact when the true posterior factorizes)
#   4. Multi-modality: multiple modalities sum log-likelihoods correctly

using Distributions: Categorical, MultivariateDistribution, probs

# Helper from the multi-factor model tests
function _random_col_stoch_tensor_fpi(shape::NTuple{N,Int}; rng=Xoshiro(0)) where N
    out = zeros(Float64, shape...)
    n_obs = shape[1]
    inner_shape = shape[2:end]
    for idx in CartesianIndices(inner_shape)
        out[:, idx] = softmax(randn(rng, n_obs))
    end
    return out
end

@testset "F=1 multi-factor matches single-factor closed-form" begin
    rng = Xoshiro(0x101)
    A = _random_col_stoch_tensor_fpi((4, 3); rng=rng)
    B = _random_col_stoch_tensor_fpi((3, 3, 2); rng=rng)
    C = zeros(4)
    D = [0.5, 0.3, 0.2]

    sf = DiscretePOMDP(A, B, C, D)
    mf = MultiFactorDiscretePOMDP(sf)

    alg = FixedPointIteration()

    for obs in 1:4
        q_sf = infer_states(alg, sf, state_prior(sf), obs)
        q_mf = infer_states(alg, mf, state_prior(mf), [obs])
        @test probs(q_sf) ≈ probs(q_mf) atol=1e-12

        F_sf = free_energy(alg, sf, state_prior(sf), obs, q_sf)
        F_mf = free_energy(alg, mf, state_prior(mf), [obs], q_mf)
        @test F_sf ≈ F_mf atol=1e-12
    end
end

@testset "F=2 mean-field: free energy decreases monotonically" begin
    rng = Xoshiro(0x202)
    # 2 state factors of sizes (3, 4); 1 modality of size 5
    A1 = _random_col_stoch_tensor_fpi((5, 3, 4); rng=rng)
    B1 = _random_col_stoch_tensor_fpi((3, 3, 2); rng=rng)
    B2 = _random_col_stoch_tensor_fpi((4, 4, 1); rng=rng)
    C  = [zeros(5)]
    D  = [fill(1/3, 3), fill(1/4, 4)]
    m  = MultiFactorDiscretePOMDP([A1], [B1, B2], C, D)

    # Use a HIGH iteration count and tight tolerance so we can observe descent
    alg = FixedPointIteration(num_iter=10, dF_tol=1e-12)

    # We need access to F at each iteration for the monotonicity check.
    # Use the internal `infer_states_with_history` (see below) if exposed,
    # or run multiple `infer_states` calls with bounded iter counts and verify
    # F at each.
    F_traj = Float64[]
    for k in 1:10
        a = FixedPointIteration(num_iter=k, dF_tol=0.0)
        q = infer_states(a, m, state_prior(m), [3])
        F = free_energy(a, m, state_prior(m), [3], q)
        push!(F_traj, F)
    end
    # F should be non-increasing across iterations
    diffs = diff(F_traj)
    @test all(diffs .<= 1e-9)
end

@testset "factorized-posterior model: mean-field is exact" begin
    # A model where each modality depends on only ONE state factor
    # → the posterior factorizes; mean-field is exact.
    rng = Xoshiro(0x303)
    # Construct A1 such that it depends only on factor 1 (uniform over factor 2)
    # i.e. A1[o, s1, s2] = A1[o, s1, s2'] for all s2, s2'
    base1 = zeros(4, 3)
    for s1 in 1:3
        base1[:, s1] = softmax(randn(rng, 4))
    end
    # Replicate over factor 2's dim
    A1 = zeros(4, 3, 2)
    for s1 in 1:3, s2 in 1:2
        A1[:, s1, s2] = base1[:, s1]
    end
    B1 = _random_col_stoch_tensor_fpi((3, 3, 2); rng=rng)
    B2 = _random_col_stoch_tensor_fpi((2, 2, 1); rng=rng)
    C  = [zeros(4)]
    D  = [softmax(randn(rng, 3)), softmax(randn(rng, 2))]
    m  = MultiFactorDiscretePOMDP([A1], [B1, B2], C, D)

    alg = FixedPointIteration(num_iter=20, dF_tol=1e-14)
    q = infer_states(alg, m, state_prior(m), [2])

    # Posterior should be factorized:
    #   q_1(s_1) ∝ A1_base[2, s_1] * D[1][s_1]
    #   q_2(s_2) = D[2][s_2] (uninformative observation)
    expected_q1 = let p = base1[2, :] .* D[1]
                      p ./ sum(p)
                  end
    expected_q2 = D[2]
    @test probs(marginal(q, 1)) ≈ expected_q1 atol=1e-10
    @test probs(marginal(q, 2)) ≈ expected_q2 atol=1e-10
end

@testset "multi-modality: log-likelihoods sum across modalities" begin
    # 1 state factor, 2 modalities; verify q ∝ exp(Σ_m log A[m][o[m], :] + log prior)
    rng = Xoshiro(0x404)
    A1 = _random_col_stoch_tensor_fpi((3, 4); rng=rng)
    A2 = _random_col_stoch_tensor_fpi((2, 4); rng=rng)
    B  = _random_col_stoch_tensor_fpi((4, 4, 2); rng=rng)
    C  = [zeros(3), zeros(2)]
    D  = [softmax(randn(rng, 4))]
    m  = MultiFactorDiscretePOMDP([A1, A2], [B], C, D)

    alg = FixedPointIteration()
    obs = [2, 1]
    q = infer_states(alg, m, state_prior(m), obs)

    expected = let
        log_lik = log.(A1[obs[1], :]) .+ log.(A2[obs[2], :])
        log_prior = log.(D[1])
        softmax(log_lik .+ log_prior)
    end
    @test probs(q) ≈ expected atol=1e-12
end
