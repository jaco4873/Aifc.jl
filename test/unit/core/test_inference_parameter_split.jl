# Pin the Inference / ParameterLearning split.
#
# Prior to this refactor, both state inference (FixedPointIteration) and
# parameter learning (DirichletConjugate) were `<: Inference`, with a
# `supports_states` / `supports_parameters` capability-flag system to
# signal which methods each subtype actually implemented. That was a
# Friston-aesthetic ("all variational inference is one operation")
# decision that bloated the Inference interface in practice — every
# Inference subtype carried error-fallbacks for methods it didn't use.
#
# Cleaner design: two abstract types.
#   - `Inference` for state inference (`infer_states`, `free_energy`)
#   - `ParameterLearning` for trajectory-level updates (`infer_parameters`)
#
# `Agent` now takes one of each — the type system enforces what each role
# is allowed to do.

using Aifc: Inference, ParameterLearning, FixedPointIteration, DirichletConjugate

@testset "Type hierarchy: Inference and ParameterLearning are distinct" begin
    # FixedPointIteration is state inference only
    @test FixedPointIteration <: Inference
    @test !(FixedPointIteration <: ParameterLearning)

    # DirichletConjugate is parameter learning only
    @test DirichletConjugate <: ParameterLearning
    @test !(DirichletConjugate <: Inference)

    # The two abstract types are unrelated — no shared supertype between them
    # (other than Any). Concretely, neither is a subtype of the other.
    @test !(Inference <: ParameterLearning)
    @test !(ParameterLearning <: Inference)
end

@testset "Agent constructor: type-enforced split" begin
    using Distributions: Categorical
    A = [0.9 0.1; 0.1 0.9]
    B = zeros(2, 2, 1); B[1,1,1] = 1; B[2,2,1] = 1
    C = [0.0, 0.0]; D = [0.5, 0.5]
    pA = ones(2, 2); pB = ones(2, 2, 1); pD = ones(2)
    m = DiscretePOMDP(A, B, C, D; pA=pA, pB=pB, pD=pD, check=false)

    fpi    = FixedPointIteration()
    planner = EnumerativeEFE(γ=4.0, horizon=1)
    sel    = Stochastic(α=4.0)
    learner = DirichletConjugate()

    # Valid: state inference is an Inference, learner is a ParameterLearning
    agent = Agent(m, fpi, planner, sel; parameter_learning=learner)
    @test agent isa Agent

    # Invalid: passing a ParameterLearning where state_inference is expected
    # — no constructor method matches because the positional slot is typed
    # `::Inference` and DirichletConjugate is `::ParameterLearning`.
    @test_throws MethodError Agent(m, learner, planner, sel)

    # Invalid: passing an Inference where parameter_learning is expected
    # — kwarg type assertion `::Union{Nothing, ParameterLearning}` rejects.
    @test_throws TypeError Agent(m, fpi, planner, sel; parameter_learning=fpi)
end
