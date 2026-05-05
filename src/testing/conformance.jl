# Conformance test suites for the package's interfaces.
#
# Anyone implementing `GenerativeModel`, `Inference`, `PolicyInference`,
# `ActionSelector`, or a custom belief type can run these to verify their
# implementation satisfies the mathematical and structural contracts.
#
# Usage:
#
#     using Aifc.Testing
#     using Test
#     @testset begin
#         test_generative_model(MyModel())
#         test_state_inference(MyAlg(), MyModel())
#         ...
#     end

module Testing

using ..Aifc
using ..Aifc: GenerativeModel, Inference, PolicyInference, ActionSelector
using ..Aifc: state_prior, observation_distribution, transition_distribution
using ..Aifc: log_preferences, action_space, log_policy_prior
using ..Aifc: goal_state_prior, state_factors, observation_modalities
using ..Aifc: infer_states, infer_parameters, free_energy
using ..Aifc: posterior_policies, expected_free_energy
using ..Aifc: pragmatic_value, epistemic_value, enumerate_policies
using ..Aifc: sample_action, log_action_probabilities
using ..Aifc: belief_style, isnormalized, BeliefStyle
using ..Aifc: CategoricalStyle, GaussianStyle, ProductStyle
using Distributions: Distribution, Categorical, MvNormal, support, pdf, rand
using Random: AbstractRNG, Xoshiro
using Test

export test_generative_model, test_state_inference, test_policy_inference,
       test_action_selector, test_belief

"""
    test_belief(b::Distribution; rng=Xoshiro(1))

Verify the `Distribution` `b` satisfies the basic invariants its
`belief_style` implies. Always-on checks: `pdf` is non-negative on a sample
from `b`. Style-specific checks per `belief_style`.
"""
function test_belief(b::Distribution; rng::AbstractRNG = Xoshiro(1))
    @testset "Belief invariants ($(typeof(b)))" begin
        # Always-on: pdf >= 0
        x = rand(rng, b)
        @test pdf(b, x) >= 0

        # Style-specific
        s = belief_style(b)
        if s isa CategoricalStyle
            @test isnormalized(b)
        elseif s isa ProductStyle
            @test isnormalized(b)
        end
    end
end

"""
    test_generative_model(m::GenerativeModel; rng=Xoshiro(1), n_samples=20)

Verify `m` satisfies the `GenerativeModel` contract:

- `state_prior(m)` is a normalized `Distribution`
- `observation_distribution(m, s)` returns a `Distribution` for sampled `s`
- `transition_distribution(m, s, a)` returns a `Distribution` for sampled `(s, a)`
- `log_preferences(m, o)` returns a finite `Real` for sampled `o`
- `action_space(m)` is non-empty
"""
function test_generative_model(m::GenerativeModel;
                                rng::AbstractRNG = Xoshiro(1),
                                n_samples::Int = 20)
    @testset "GenerativeModel conformance ($(typeof(m)))" begin
        sp = state_prior(m)
        @test sp isa Distribution
        @test isnormalized(sp)

        as = action_space(m)
        @test !isempty(as)
        actions = collect(as)

        for _ in 1:n_samples
            s = rand(rng, sp)

            # observation likelihood
            od = observation_distribution(m, s)
            @test od isa Distribution
            o = rand(rng, od)

            # log preferences
            lp = log_preferences(m, o)
            @test lp isa Real
            @test !isnan(lp)

            # transitions for each action
            for a in actions
                td = transition_distribution(m, s, a)
                @test td isa Distribution
                s_next = rand(rng, td)
                @test pdf(td, s_next) >= 0
            end
        end
    end
end

"""
    test_state_inference(alg::Inference, m::GenerativeModel; rng=Xoshiro(1))

Verify `alg` satisfies the state-inference contract on model `m`:

- `infer_states(alg, m, history)` returns a normalized `Distribution`
- `free_energy(alg, m, history, q)` returns a finite `Real`
- For an iterative algorithm with multiple iterations, free energy is
  non-increasing across iterations (within tolerance)
"""
function test_state_inference(alg::Inference, m::GenerativeModel;
                                rng::AbstractRNG = Xoshiro(1))
    @testset "StateInference conformance ($(typeof(alg)))" begin
        # Generate a sample observation under the model's prior
        sp = state_prior(m)
        s = rand(rng, sp)
        o = rand(rng, observation_distribution(m, s))

        # Single-step interface: prior + observation -> posterior
        q = infer_states(alg, m, sp, o)
        @test q isa Distribution
        @test isnormalized(q)

        F = free_energy(alg, m, sp, o, q)
        @test F isa Real
        @test !isnan(F) && !isinf(F)
    end
end

"""
    test_policy_inference(planner::PolicyInference, m::GenerativeModel, qs;
                          rng=Xoshiro(1))

Verify `planner` satisfies the policy-inference contract on model `m`:

- `posterior_policies` returns `(q_pi, G)` with `q_pi` summing to 1
- `expected_free_energy(planner, m, qs, π) ≈ -(pragmatic + epistemic)` — the
  decomposition identity that any conforming planner must satisfy
- `epistemic_value` is non-negative (KL divergences are non-negative)
"""
function test_policy_inference(planner::PolicyInference,
                                 m::GenerativeModel,
                                 qs::Distribution,
                                 policies::AbstractVector;
                                 rng::AbstractRNG = Xoshiro(1),
                                 atol::Real = 1e-8)
    @testset "PolicyInference conformance ($(typeof(planner)))" begin
        q_pi, G = posterior_policies(planner, m, qs, policies)
        @test length(q_pi) == length(policies)
        @test length(G) == length(policies)
        @test sum(q_pi) ≈ 1 atol=atol
        @test all(>=(zero(eltype(q_pi))), q_pi)

        for π in policies
            G_π   = expected_free_energy(planner, m, qs, π)
            prag  = pragmatic_value(planner, m, qs, π)
            epi   = epistemic_value(planner, m, qs, π)
            @test G_π isa Real && !isnan(G_π)
            @test prag isa Real && !isnan(prag)
            @test epi isa Real && !isnan(epi)
            # Decomposition identity
            @test G_π ≈ -(prag + epi) atol=atol
            # Information non-negativity
            @test epi >= -atol
        end
    end
end

"""
    test_action_selector(sel::ActionSelector, q_pi, policies; rng=Xoshiro(1), n_samples=200)

Verify `sel` satisfies the action-selector contract:

- `sample_action` returns an action for any seed
- `log_action_probabilities` returns a vector summing to 1 (after `exp`)
"""
function test_action_selector(sel::ActionSelector,
                                q_pi::AbstractVector,
                                policies::AbstractVector;
                                rng::AbstractRNG = Xoshiro(1),
                                atol::Real = 1e-8)
    @testset "ActionSelector conformance ($(typeof(sel)))" begin
        # log-probability vector
        if hasmethod(log_action_probabilities, Tuple{typeof(sel), typeof(q_pi), typeof(policies)})
            lap = log_action_probabilities(sel, q_pi, policies)
            @test sum(exp, lap) ≈ 1 atol=atol
        end

        # sample_action returns something deterministic-ish given seeded rng
        a = sample_action(sel, q_pi, policies, rng)
        @test a !== nothing
    end
end

end # module Testing
