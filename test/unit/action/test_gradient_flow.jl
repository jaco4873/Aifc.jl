# Verify that gradients flow through parametric `Stochastic` / `EnumerativeEFE`
# / `DirichletConjugate` configs.
#
# Without parametric typing (`α::Float64`), constructing
# `Stochastic(α=Dual_value)` would coerce to `Float64`, dropping the
# gradient. With `α::T<:Real`, the constructed `Stochastic{Dual{...}}`
# preserves the type and gradient flows through subsequent operations.
#
# We compare ForwardDiff-via-the-typed-config to FiniteDifferences as
# a self-consistency check.

using FiniteDifferences
using Distributions: logpdf, Categorical
import ForwardDiff

const FDM_grad = central_fdm(5, 1)

@testset "Stochastic α: gradient flow" begin
    # A 4-policy scenario where the action distribution depends nontrivially on α
    q_pi = [0.4, 0.3, 0.2, 0.1]
    policies = [[1], [2], [3], [1]]    # action 1 has aggregated mass 0.5

    # Loss: log probability of action 2 under Stochastic(α)
    f(α) = let
        sel = Stochastic(α=α)
        d = action_distribution(sel, q_pi, policies)
        logpdf(d, 2)
    end

    # ForwardDiff gradient (uses Dual numbers)
    g_ad = ForwardDiff.derivative(f, 2.5)

    # FiniteDifferences reference
    g_fd = grad(FDM_grad, f, 2.5)[1]

    @test isfinite(g_ad)
    @test g_ad ≈ g_fd atol=1e-6
end

@testset "EnumerativeEFE γ: gradient flow into policy posterior" begin
    # γ flows into softmax over -G(π); we differentiate the log of q(π=1)
    # wrt γ and verify ForwardDiff gradient matches finite differences.
    rng = Xoshiro(0xF1)
    A = zeros(3, 4)
    for s in 1:4
        A[:, s] = softmax(randn(rng, 3))
    end
    B = zeros(4, 4, 2)
    for a in 1:2, s in 1:4
        B[:, s, a] = softmax(randn(rng, 4))
    end
    C = randn(rng, 3)
    D = softmax(randn(rng, 4))
    m = DiscretePOMDP(A, B, C, D)
    qs = state_prior(m)

    f(γ) = let
        p = EnumerativeEFE(γ=γ, horizon=1)
        pols = collect(enumerate_policies(p, m, qs))
        q_pi, _ = posterior_policies(p, m, qs, pols)
        log(q_pi[1])
    end

    g_ad = ForwardDiff.derivative(f, 4.0)
    g_fd = grad(FDM_grad, f, 4.0)[1]

    @test isfinite(g_ad)
    @test g_ad ≈ g_fd atol=1e-6
end

@testset "Composed: log P(action=1) wrt α through Stochastic + EnumerativeEFE" begin
    # End-to-end gradient: action distribution depends on α (Stochastic) and
    # policy posterior depends on γ (EnumerativeEFE). Differentiate wrt α only.
    rng = Xoshiro(0xF2)
    m = random_pomdp(3, 4, 2; rng=rng)
    qs = state_prior(m)
    pol = EnumerativeEFE(γ=4.0, horizon=2)
    pols = collect(enumerate_policies(pol, m, qs))
    q_pi, _ = posterior_policies(pol, m, qs, pols)

    f(α) = let
        sel = Stochastic(α=α)
        d = action_distribution(sel, q_pi, pols)
        logpdf(d, 1)
    end

    g_ad = ForwardDiff.derivative(f, 8.0)
    g_fd = grad(FDM_grad, f, 8.0)[1]
    @test g_ad ≈ g_fd atol=1e-6
end

@testset "DirichletConjugate lr_pA: gradient flow through pA update" begin
    # Build a learnable POMDP, take one step, differentiate the post-update
    # pA[1, 1] wrt lr_pA.
    A = [0.5 0.5; 0.5 0.5]
    B = zeros(2, 2, 1)
    B[1, 1, 1] = 1; B[2, 2, 1] = 1
    C = [0.0, 0.0]
    D = [0.5, 0.5]
    pA = ones(2, 2)
    m = DiscretePOMDP(A, B, C, D; pA=pA, check=false)

    history = AgentHistory(state_prior(m))
    push_step!(history, AgentStep(1, Categorical([0.8, 0.2]),
                                    [1], [1.0], [0.0], 0.0, 1))

    f(lr) = let
        rule = DirichletConjugate(lr_pA=lr, learn_pB=false, learn_pD=false,
                                    use_effective_A=false)
        m_new = infer_parameters(rule, m, history)
        m_new.pA[1, 1]
    end

    g_ad = ForwardDiff.derivative(f, 1.0)
    g_fd = grad(FDM_grad, f, 1.0)[1]

    @test g_ad ≈ g_fd atol=1e-7
end
