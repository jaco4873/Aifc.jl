# Cross-backend gradient validation via DifferentiationInterface.jl.
#
# Computes the same agent log-likelihood gradient under multiple AD backends
# and asserts they agree. Validates the package's "AD-backend-agnostic via
# parametric <:Real types" claim through DI directly (parallel to the
# Turing-NUTS validation in test/sbc/test_parameter_recovery.jl, but at the
# gradient level rather than the posterior level).
#
# Backends checked:
#   - AutoForwardDiff   (Dual numbers)
#   - AutoReverseDiff   (TrackedReal tape)
#   - FiniteDifferences (numerical, as a sanity ground truth)
#
# What's being differentiated: the log-likelihood of an observed action
# sequence under an active-inference agent, as a function of the action
# precision α. This is exactly the function Turing-NUTS optimizes during
# parameter fitting — so cross-backend agreement here is the gradient-level
# evidence that the parametric <:Real design works.

using DifferentiationInterface
using DifferentiationInterface: AutoForwardDiff, AutoReverseDiff, AutoFiniteDifferences
using Distributions: Categorical, logpdf
using Random: Xoshiro
import ForwardDiff
import ReverseDiff
import FiniteDifferences

# Build a small POMDP and a synthetic action sequence at fixed α_true,
# returning a log-likelihood-as-a-function-of-α closure.
function _build_loglik_closure(; α_true::Float64 = 8.0, n_steps::Int = 30,
                                  rng_seed = 0xABCD)
    # Use the T-maze model — known from the SBC parameter recovery test to
    # produce informative likelihood curves over α.
    m = tmaze_model(cue_reliability=0.95, preference=4.0)

    # Simulate at α_true to get a fixed observation+action sequence.
    rng = Xoshiro(rng_seed)
    sel_true = Stochastic(α=α_true)
    planner   = EnumerativeEFE(γ=8.0, horizon=2)
    fpi       = FixedPointIteration()

    obs_seq = [rand(rng, 1:5) for _ in 1:n_steps]
    actions = Int[]
    qs = state_prior(m)
    last_action = -1
    for t in 1:n_steps
        prior = last_action == -1 ?
            state_prior(m) : predict_states(m, qs, last_action)
        qs = infer_states(fpi, m, prior, obs_seq[t])
        policies = collect(enumerate_policies(planner, m, qs))
        q_pi, _ = posterior_policies(planner, m, qs, policies)
        d = action_distribution(sel_true, q_pi, policies)
        a = rand(rng, d)
        push!(actions, a)
        last_action = a
    end

    # Log-likelihood of (obs_seq, actions) as a function of α — vector input
    # because DifferentiationInterface's `gradient` API expects vector inputs.
    function loglik(αvec::AbstractVector{<:Real})
        α = αvec[1]
        sel = Stochastic(α=α)
        qs = state_prior(m)
        last_action = -1
        ll = zero(α)
        for t in eachindex(obs_seq)
            prior = last_action == -1 ?
                state_prior(m) : predict_states(m, qs, last_action)
            qs = infer_states(fpi, m, prior, obs_seq[t])
            policies = collect(enumerate_policies(planner, m, qs))
            q_pi, _ = posterior_policies(planner, m, qs, policies)
            d = action_distribution(sel, q_pi, policies)
            ll += logpdf(d, actions[t])
            last_action = actions[t]
        end
        return ll
    end

    return loglik
end

@testset "DI: ForwardDiff and ReverseDiff agree on agent log-likelihood gradient" begin
    loglik = _build_loglik_closure()

    # Probe the gradient at several α values across the support of a typical
    # LogNormal(log(8), 0.5) prior. At least one of these will have a
    # non-trivial gradient on a 30-step trial; agreement at all of them is
    # what we care about.
    for α in (3.0, 6.0, 8.0, 12.0, 20.0)
        αvec = [α]
        g_fd = gradient(loglik, AutoForwardDiff(), αvec)
        g_rd = gradient(loglik, AutoReverseDiff(), αvec)
        # Use atol because the gradient can be ~0 near the MLE, and rtol
        # against zero is meaningless. Agreement to ~1e-10 absolute is the
        # standard "two AD systems compute the same number" precision.
        @test g_fd ≈ g_rd  atol=1e-10
    end
end

@testset "DI: AD agrees with finite differences (sanity ground truth)" begin
    loglik = _build_loglik_closure()

    for α in (3.0, 6.0, 12.0, 20.0)
        αvec = [α]
        g_fd  = gradient(loglik, AutoForwardDiff(), αvec)
        g_num = gradient(loglik,
                          AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(5, 1)),
                          αvec)
        # Finite differences ~1e-6 noise floor; absolute tolerance 1e-4
        # comfortably covers it without hiding real disagreement.
        @test g_fd ≈ g_num  atol=1e-4
    end
end

@testset "DI: gradient sign tracks likelihood geometry at extreme α" begin
    # At α_true=8, log-likelihood peaks (or near-peaks). At very low α the
    # gradient should be non-negative (pull α up); at very high α
    # non-positive (pull α down). We probe with extreme values to ensure the
    # signal is large enough to be unambiguous across rng-dependent
    # trajectories.
    loglik = _build_loglik_closure(α_true=8.0)
    g_low  = gradient(loglik, AutoForwardDiff(), [1.0])
    g_high = gradient(loglik, AutoForwardDiff(), [50.0])

    @test g_low[1]  >= -1e-6   # positive or near-zero (numerical wiggle)
    @test g_high[1] <=  1e-6   # negative or near-zero
end
