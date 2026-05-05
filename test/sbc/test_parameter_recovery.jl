# Parameter recovery sanity check.
#
# Simulate behavior at known α_true, fit α via Turing-NUTS through a
# direct `@model` (bypassing ActionModels' create_model, which has an
# API quirk for single-parameter fits), check posterior 95% credible
# interval contains α_true.
#
# Run twice — once with ForwardDiff (Turing's default), once with
# ReverseDiff — to validate the package's claim of being AD-backend-
# agnostic via parametric `<:Real` types.
#
# This is the precondition for full SBC. If parameter recovery doesn't
# work for one synthetic trial, SBC won't either. And if it works under
# ForwardDiff but breaks under ReverseDiff, the parametric-types story
# is incomplete.

using ActionModels: NUTS, @model, sample, Chains, MCMCThreads
using Distributions: LogNormal
using Statistics: mean as dmean, std as dstd, quantile
using Random: Xoshiro
using ADTypes: AutoForwardDiff, AutoReverseDiff
import Distributions
import ReverseDiff   # loaded so AutoReverseDiff has the backend available

# Direct Turing model: prior on α, simulates the agent forward, observes actions
@model function aif_alpha_fit(model::DiscretePOMDP, observations::Vector{Int},
                                actions::Vector{Int})
    α ~ LogNormal(log(8.0), 0.5)
    sel = Stochastic(α=α)
    planner = EnumerativeEFE(γ=8.0, horizon=2)
    fpi = FixedPointIteration()

    qs = state_prior(model)
    last_action = -1   # -1 = no previous action

    for t in eachindex(observations)
        prior = if last_action == -1
            state_prior(model)
        else
            predict_states(model, qs, last_action)
        end

        qs = infer_states(fpi, model, prior, observations[t])

        policies = collect(enumerate_policies(planner, model, qs))
        q_pi, _ = posterior_policies(planner, model, qs, policies)

        d = action_distribution(sel, q_pi, policies)
        actions[t] ~ d

        last_action = actions[t]
    end
end

# Generate a synthetic trajectory under known α_true. Returns (obs, actions).
function _simulate_trial(m::DiscretePOMDP, α_true::Real;
                          n_steps::Int = 30, rng_seed = 0xABCD)
    rng = Xoshiro(rng_seed)
    fpi = FixedPointIteration()
    planner = EnumerativeEFE(γ=8.0, horizon=2)
    sel_true = Stochastic(α=α_true)

    obs_seq = [rand(rng, 1:5) for _ in 1:n_steps]
    actions = Int[]
    qs = state_prior(m)
    last_action = -1

    for t in 1:n_steps
        prior = last_action == -1 ?
            state_prior(m) :
            predict_states(m, qs, last_action)
        qs = infer_states(fpi, m, prior, obs_seq[t])
        policies = collect(enumerate_policies(planner, m, qs))
        q_pi, _ = posterior_policies(planner, m, qs, policies)
        d = action_distribution(sel_true, q_pi, policies)
        a = rand(rng, d)
        push!(actions, a)
        last_action = a
    end
    return obs_seq, actions
end

# Fit α via Turing-NUTS using the supplied AD backend; return posterior samples.
function _fit_alpha(m::DiscretePOMDP, obs_seq::Vector{Int}, actions::Vector{Int};
                     adtype, n_samples::Int = 300)
    model = aif_alpha_fit(m, obs_seq, actions)
    chain = sample(model, NUTS(0.65; adtype=adtype), n_samples; progress=false)
    return vec(Array(chain[:α]))
end

# One run of "simulate at α_true → fit α → check CI contains truth".
function _check_recovers_alpha(adtype; α_true::Real = 8.0, n_steps::Int = 30,
                                  rng_seed = 0xABCD, n_samples::Int = 300)
    m = tmaze_model(cue_reliability=0.95, preference=4.0)
    obs_seq, actions = _simulate_trial(m, α_true; n_steps=n_steps, rng_seed=rng_seed)
    α_samples = _fit_alpha(m, obs_seq, actions; adtype=adtype, n_samples=n_samples)

    α_post_mean = dmean(α_samples)
    α_post_std  = dstd(α_samples)
    α_lo, α_hi  = quantile(α_samples, 0.025), quantile(α_samples, 0.975)

    @info "Parameter recovery" backend=string(adtype) α_true α_post_mean α_post_std α_lo α_hi

    return (α_lo=α_lo, α_hi=α_hi, α_true=α_true,
             α_post_mean=α_post_mean, α_post_std=α_post_std,
             samples=α_samples)
end

@testset "parameter recovery: α via ForwardDiff" begin
    result = _check_recovers_alpha(AutoForwardDiff())
    # 95% credible interval should contain α_true
    @test result.α_lo <= result.α_true <= result.α_hi
end

@testset "parameter recovery: α via ReverseDiff" begin
    # Validates the "backend-agnostic via parametric <:Real" claim. If this
    # breaks, either there's a Float64 lock somewhere (like the AgentStep
    # one we just fixed) or ReverseDiff is hitting a real issue with our
    # code. Either way, an actionable signal.
    result = _check_recovers_alpha(AutoReverseDiff(; compile=false))
    @test result.α_lo <= result.α_true <= result.α_hi
end
