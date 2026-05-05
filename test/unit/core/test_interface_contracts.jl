# Interface-contract tests.
#
# Each abstract type (`GenerativeModel`, `Inference`, `PolicyInference`,
# `ActionSelector`) defines a set of required methods that throw an error
# when called on a subtype that doesn't implement them. These tests:
#
#   1. Document the contract in test code (each test names a required method)
#   2. Verify the error message mentions the offending method (so users get
#      a useful diagnostic)
#   3. Push the abstract-stub files to ~100% line coverage (cosmetic but
#      makes the coverage report a sharper signal of real gaps)

using Aifc
using Aifc: GenerativeModel, Inference, PolicyInference, ActionSelector
using Aifc: state_prior, observation_distribution, transition_distribution
using Aifc: log_preferences, action_space, predict_states
using Aifc: infer_states, infer_parameters, free_energy
using Aifc: posterior_policies, expected_free_energy
using Aifc: pragmatic_value, epistemic_value, enumerate_policies
using Aifc: action_distribution, sample_action, log_action_probabilities
using Aifc: supports_states, supports_parameters
using Aifc: supported_targets

# Empty subtypes that don't implement anything
struct _MockModel <: GenerativeModel end
struct _MockInference <: Inference end
struct _MockPolicyInference <: PolicyInference end
struct _MockActionSelector <: ActionSelector end

@testset "GenerativeModel: required methods throw on unimplemented subtype" begin
    m = _MockModel()
    @test_throws Exception state_prior(m)
    @test_throws Exception observation_distribution(m, 1)
    @test_throws Exception transition_distribution(m, 1, 1)
    @test_throws Exception log_preferences(m, 1)
    @test_throws Exception action_space(m)
    @test_throws Exception predict_states(m, nothing, 1)

    # Optional methods have sensible defaults
    @test Aifc.log_policy_prior(m, [1]) == 0.0
    @test Aifc.goal_state_prior(m) === nothing
    @test Aifc.state_factors(m) == (1,)
    @test Aifc.observation_modalities(m) == (1,)
    @test Aifc.policy_horizon(m) == 1
    @test Aifc.nstates(m) == 1
    @test Aifc.nobservations(m) == 1
end

@testset "Inference: required methods throw on unimplemented subtype" begin
    alg = _MockInference()
    m = _MockModel()
    @test_throws Exception infer_states(alg, m, nothing, 1)
    @test_throws Exception free_energy(alg, m, nothing, 1, nothing)
    @test_throws Exception infer_parameters(alg, m, nothing)

    # Default capability flags
    @test supports_states(alg) == false
    @test supports_parameters(alg) == false
    @test isempty(supported_targets(alg))
end

@testset "PolicyInference: required methods throw on unimplemented subtype" begin
    p = _MockPolicyInference()
    m = _MockModel()
    @test_throws Exception posterior_policies(p, m, nothing, [])
    @test_throws Exception expected_free_energy(p, m, nothing, [])
    @test_throws Exception pragmatic_value(p, m, nothing, [])
    @test_throws Exception epistemic_value(p, m, nothing, [])
    @test_throws Exception enumerate_policies(p, m, nothing)
end

@testset "ActionSelector: required methods throw on unimplemented subtype" begin
    sel = _MockActionSelector()
    @test_throws Exception action_distribution(sel, [1.0], [[1]])
    @test_throws Exception sample_action(sel, [1.0], [[1]])
    @test_throws Exception log_action_probabilities(sel, [1.0], [[1]])
end
