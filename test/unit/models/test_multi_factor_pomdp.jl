# Multi-factor MultiFactorDiscretePOMDP tests.
#
# Internal representation is ALWAYS multi-factor: A is `Vector{Array{T}}`
# (one tensor per observation modality, shape (n_obs[m], factor sizes...)),
# B is `Vector{Array{T,3}}` (one per state factor, shape
# (n_states_f, n_states_f, n_actions_f)), C and D are `Vector{Vector{T}}`.
#
# The single-factor / single-modality construction syntax that we use
# elsewhere remains backward-compatible — see `test_discrete_pomdp.jl`.

using Distributions: Categorical, probs
using Distributions: MultivariateDistribution, ContinuousMultivariateDistribution

# Helper: column-stochastic 3D tensor with given shape (rng-dependent)
function _random_col_stoch_tensor(shape::NTuple{N,Int}; rng=Xoshiro(0)) where N
    out = zeros(Float64, shape...)
    n_obs = shape[1]
    inner_shape = shape[2:end]
    for idx in CartesianIndices(inner_shape)
        out[:, idx] = softmax(randn(rng, n_obs))
    end
    return out
end

@testset "construct 2-factor / single-modality model" begin
    # 2 state factors of sizes (3, 2); 1 modality of size 4; 2 action factors
    # (factor 1 has 3 actions; factor 2 has 1 action — uncontrollable).
    rng = Xoshiro(0xCAFE)

    A1 = _random_col_stoch_tensor((4, 3, 2); rng=rng)        # 4 × 3 × 2
    A = [A1]

    B1 = _random_col_stoch_tensor((3, 3, 3); rng=rng)         # factor 1: 3 actions
    B2 = _random_col_stoch_tensor((2, 2, 1); rng=rng)         # factor 2: 1 action (passive)
    B = [B1, B2]

    C = [zeros(4)]
    D = [fill(1/3, 3), fill(1/2, 2)]

    m = MultiFactorDiscretePOMDP(A, B, C, D)
    @test m isa MultiFactorDiscretePOMDP
    @test state_factors(m)         == (3, 2)
    @test observation_modalities(m) == (4,)
    @test nstates(m)        == 6   # product of factor sizes
    @test nobservations(m)  == 4   # product of modality sizes
end

@testset "construct 1-factor / 2-modality model" begin
    rng = Xoshiro(0xBEEF)
    A1 = _random_col_stoch_tensor((4, 3); rng=rng)             # modality 1: 4 outcomes
    A2 = _random_col_stoch_tensor((2, 3); rng=rng)             # modality 2: 2 outcomes
    A  = [A1, A2]

    B1 = _random_col_stoch_tensor((3, 3, 2); rng=rng)
    B  = [B1]

    C  = [zeros(4), zeros(2)]
    D  = [fill(1/3, 3)]

    m = MultiFactorDiscretePOMDP(A, B, C, D)
    @test state_factors(m)          == (3,)
    @test observation_modalities(m) == (4, 2)
    @test nstates(m)        == 3
    @test nobservations(m)  == 8
end

@testset "construct 2-factor / 2-modality model" begin
    rng = Xoshiro(0x6F75)
    A1 = _random_col_stoch_tensor((4, 3, 2); rng=rng)         # depends on both factors
    A2 = _random_col_stoch_tensor((2, 3, 2); rng=rng)
    A  = [A1, A2]

    B1 = _random_col_stoch_tensor((3, 3, 3); rng=rng)
    B2 = _random_col_stoch_tensor((2, 2, 1); rng=rng)
    B  = [B1, B2]

    C = [zeros(4), zeros(2)]
    D = [fill(1/3, 3), fill(1/2, 2)]

    m = MultiFactorDiscretePOMDP(A, B, C, D)
    @test state_factors(m)          == (3, 2)
    @test observation_modalities(m) == (4, 2)
end

@testset "construction validation: shape mismatches caught" begin
    A1 = _random_col_stoch_tensor((4, 3, 2))
    B1 = _random_col_stoch_tensor((3, 3, 2))
    B2 = _random_col_stoch_tensor((2, 2, 1))
    C  = [zeros(4)]
    D  = [fill(1/3, 3), fill(1/2, 2)]

    # A's rank must equal 1 + F (number of state factors)
    A_bad_rank = _random_col_stoch_tensor((4, 3))    # rank 2, F=2 → mismatch
    @test_throws DimensionMismatch MultiFactorDiscretePOMDP([A_bad_rank], [B1, B2], C, D)

    # A's trailing dims must match D factor sizes
    A_bad_dims = _random_col_stoch_tensor((4, 3, 3))  # last dim 3 ≠ |D[2]| = 2
    @test_throws DimensionMismatch MultiFactorDiscretePOMDP([A_bad_dims], [B1, B2], C, D)

    # B[f] first two dims must equal |D[f]|
    B_bad = _random_col_stoch_tensor((4, 4, 2))      # both dims 4 ≠ |D[1]| = 3
    @test_throws DimensionMismatch MultiFactorDiscretePOMDP([A1], [B_bad, B2], C, D)

    # C[m] length must match A[m] first dim
    C_bad = [zeros(5)]
    @test_throws DimensionMismatch MultiFactorDiscretePOMDP([A1], [B1, B2], C_bad, D)

    # Number of state factors mismatches: |D| must equal length(B) and trailing rank of A
    D_bad = [fill(1/3, 3)]
    @test_throws DimensionMismatch MultiFactorDiscretePOMDP([A1], [B1, B2], C, D_bad)
end

@testset "stochasticity validation" begin
    A1 = _random_col_stoch_tensor((4, 3))
    B1 = _random_col_stoch_tensor((3, 3, 2))
    C  = [zeros(4)]
    D  = [fill(1/3, 3)]

    # Non-stochastic A column should be rejected when check=true
    A_bad = copy(A1)
    A_bad[:, 1] .*= 1.5    # column 1 now sums to 1.5
    @test_throws ArgumentError MultiFactorDiscretePOMDP([A_bad], [B1], C, D)

    # check=false skips validation
    @test MultiFactorDiscretePOMDP([A_bad], [B1], C, D; check=false) isa MultiFactorDiscretePOMDP
end

@testset "lift single-factor DiscretePOMDP to MultiFactor + round-trip" begin
    A = [0.9 0.1; 0.1 0.9]
    B = zeros(2, 2, 2)
    B[1, 1, 1] = 1; B[2, 2, 1] = 1
    B[2, 1, 2] = 1; B[1, 2, 2] = 1
    C = [1.0, -1.0]
    D = [0.5, 0.5]
    m_sf = DiscretePOMDP(A, B, C, D)

    # Lift
    m_mf = MultiFactorDiscretePOMDP(m_sf)
    @test state_factors(m_mf)          == (2,)
    @test observation_modalities(m_mf) == (2,)
    @test nstates(m_mf)        == 2
    @test nobservations(m_mf)  == 2
    @test m_mf.A[1] == A
    @test m_mf.B[1] == B
    @test m_mf.C[1] == C
    @test m_mf.D[1] == D

    # Round-trip back to single-factor
    m_sf2 = DiscretePOMDP(m_mf)
    @test m_sf2.A == A
    @test m_sf2.B == B
    @test m_sf2.C == C
    @test m_sf2.D == D
end

@testset "state_prior: F=1 returns Categorical, F>1 returns multivariate" begin
    # F=1 (single-factor: use the lifted form via [A], [B], [C], [D])
    A1 = [0.5 0.5; 0.5 0.5]
    B1 = ones(2, 2, 1) ./ 2
    C1 = zeros(2)
    D1 = [0.7, 0.3]
    m1 = MultiFactorDiscretePOMDP([A1], [B1], [C1], [D1])
    sp1 = state_prior(m1)
    @test sp1 isa Categorical
    @test probs(sp1) == [0.7, 0.3]

    # F=2: state_prior is multivariate (Product or similar)
    A = _random_col_stoch_tensor((3, 3, 2); rng=Xoshiro(1))
    B1 = _random_col_stoch_tensor((3, 3, 2); rng=Xoshiro(2))
    B2 = _random_col_stoch_tensor((2, 2, 1); rng=Xoshiro(3))
    Cm = [zeros(3)]
    Dm = [[0.2, 0.3, 0.5], [0.4, 0.6]]
    m2 = MultiFactorDiscretePOMDP([A], [B1, B2], Cm, Dm)
    sp2 = state_prior(m2)
    @test sp2 isa MultivariateDistribution
    # Each factor's marginal matches D[f]
    @test probs(marginal(sp2, 1)) == [0.2, 0.3, 0.5]
    @test probs(marginal(sp2, 2)) == [0.4, 0.6]
end
