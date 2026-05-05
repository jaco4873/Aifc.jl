# Tests that AgentStep does not concretize element types.
#
# Bug being guarded against: previously `AgentStep` declared
#
#     policy_posterior::Vector{Float64}
#     expected_free_energies::Vector{Float64}
#     free_energy::Float64
#
# Under Turing/ForwardDiff fitting `q_pi`, `G`, `F` arrive as
# `Vector{Dual{...}}` / `Dual{...}` (because they're computed from sampled
# precision parameters that propagate through the planner and selector).
# The concrete-Float64 declaration forced an `InexactError` on
# `convert(Float64, ::Dual)`, killing fitting at the recording step. The
# fix is to parameterize on T<:Real and let promotion handle eltypes.

using ForwardDiff
using ForwardDiff: Dual
using Distributions: Categorical

@testset "AgentStep accepts ForwardDiff.Dual eltypes" begin
    DualT = Dual{Nothing, Float64, 1}
    q_pi = DualT[Dual(0.6, 1.0), Dual(0.4, -1.0)]
    G    = DualT[Dual(-0.2, 0.1), Dual(0.5, -0.1)]
    F    = Dual(0.3, 0.05)

    step = AgentStep(1, Categorical([0.6, 0.4]),
                     [[1], [2]], q_pi, G, F, 1)

    # Eltypes preserved (no Float64 coercion)
    @test eltype(step.policy_posterior) <: Dual
    @test eltype(step.expected_free_energies) <: Dual
    @test step.free_energy isa Dual

    # Values intact
    @test ForwardDiff.value(step.policy_posterior[1]) == 0.6
    @test ForwardDiff.partials(step.free_energy)[1] == 0.05
end

@testset "AgentStep with Float64 inputs still works (regression)" begin
    step = AgentStep(1, Categorical([0.5, 0.5]),
                     [[1], [2]], [0.7, 0.3], [-0.1, 0.2], 0.4, 1)
    @test step.policy_posterior == [0.7, 0.3]
    @test step.expected_free_energies == [-0.1, 0.2]
    @test step.free_energy == 0.4
end

@testset "AgentStep: heterogeneous q_pi and G eltypes promote" begin
    # E.g. q_pi may be Vector{Dual} (depends on α) but G may stay Vector{Float64}
    # (depends only on the model). Promotion should give a Dual-eltype storage.
    DualT = Dual{Nothing, Float64, 1}
    q_pi = DualT[Dual(0.5, 0.1), Dual(0.5, -0.1)]
    G    = [-0.2, 0.5]    # Float64
    F    = 0.3            # Float64

    step = AgentStep(1, Categorical([0.5, 0.5]),
                     [[1], [2]], q_pi, G, F, 1)

    T = eltype(step.policy_posterior)
    @test T <: Dual
    @test eltype(step.expected_free_energies) === T
    @test typeof(step.free_energy) === T
end

@testset "Agent.step! propagates Dual α through to recorded step" begin
    # End-to-end: ForwardDiff.derivative through one Agent.step! call
    # against α should yield non-zero gradient on the recorded q_pi.
    using Random: Xoshiro
    A = [0.9 0.1; 0.1 0.9]
    B = zeros(2, 2, 2)
    B[1, 1, 1] = 1; B[2, 2, 1] = 1     # action 1: stay
    B[2, 1, 2] = 1; B[1, 2, 2] = 1     # action 2: swap
    C = [1.0, 0.0]
    D = [0.5, 0.5]
    m = DiscretePOMDP(A, B, C, D)

    function recorded_q_pi_first(α::Real)
        agent = Agent(m,
                      FixedPointIteration(),
                      EnumerativeEFE(γ=4.0, horizon=1),
                      Stochastic(α=α);
                      rng = Xoshiro(0))
        step!(agent, 1)
        # The first recorded policy posterior, first entry — depends on α
        # via the action-marginal softmax inside Stochastic.
        return agent.history.steps[1].policy_posterior[1]
    end

    # Just a smoke test: this would have thrown InexactError before the fix.
    # Concretely: derivative non-NaN.
    g = ForwardDiff.derivative(recorded_q_pi_first, 4.0)
    @test !isnan(g)
    @test isfinite(g)
end
