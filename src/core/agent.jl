# `Agent` — composition of (model, inferences, policy_inference, action_selection).
#
# The agent is the top-level user-facing object. `step!` runs one full
# action-perception cycle: predict prior → perceive → plan → act → (learn).

using Random: AbstractRNG, default_rng

"""
    Agent(model, state_inference, policy_inference, action_selection;
          parameter_learning=nothing, rng=Random.default_rng())

Composition of the four (or five) building blocks of an active-inference agent.

- `model::GenerativeModel` — the joint distribution `P(o, s, a, π)`
- `state_inference::Inference` — must implement `infer_states` and `free_energy`
- `policy_inference::PolicyInference` — must implement `posterior_policies`
- `action_selection::ActionSelector` — must implement `sample_action`
- `parameter_learning::Union{Nothing, Inference}` — optional; if non-nothing,
  must implement `infer_parameters`. Run after action sampling at each step.
- `rng::AbstractRNG` — used by `action_selection`.

The agent owns its `AgentHistory`, initialized with the model's `state_prior`.
"""
mutable struct Agent{M<:GenerativeModel, SI, PI, AS, PL, R<:AbstractRNG}
    model::M
    state_inference::SI
    policy_inference::PI
    action_selection::AS
    parameter_learning::PL              # ::Inference or ::Nothing
    rng::R
    history::AgentHistory
end

function Agent(model::GenerativeModel,
               state_inference::Inference,
               policy_inference::PolicyInference,
               action_selection::ActionSelector;
               parameter_learning::Union{Nothing, Inference} = nothing,
               rng::AbstractRNG = default_rng())
    supports_states(state_inference) ||
        throw(ArgumentError("Agent: state_inference $(typeof(state_inference)) does not implement infer_states"))
    if parameter_learning !== nothing
        supports_parameters(parameter_learning) ||
            throw(ArgumentError("Agent: parameter_learning $(typeof(parameter_learning)) does not implement infer_parameters"))
    end
    history = AgentHistory(state_prior(model))
    return Agent(model, state_inference, policy_inference, action_selection,
                 parameter_learning, rng, history)
end

"""
    step!(agent::Agent, observation; policies=nothing) -> action

Run one full action-perception cycle:

1. Compute predicted prior over states (from previous posterior + previous
   action via `predict_states`, or `state_prior(model)` on the first step).
2. **Perceive.** Update posterior: `qs = infer_states(state_inference, model, prior, observation)`.
3. Compute free energy at the posterior.
4. **Plan.** `(q_π, G) = posterior_policies(policy_inference, model, qs, policies)`.
5. **Act.** `a = sample_action(action_selection, q_π, policies, rng)`.
6. Record an `AgentStep` in the history.
7. **Learn.** If `parameter_learning` is set, return a new model:
   `model = infer_parameters(parameter_learning, model, history)`.

The `policies` keyword:
- `nothing` (default): planner generates its own policies via `enumerate_policies`.
- `Vector{P}`: use the supplied set.

Returns the sampled action.
"""
function step!(agent::Agent, observation; policies = nothing)
    prior = _compute_prior(agent.model, agent.history)

    qs = infer_states(agent.state_inference, agent.model, prior, observation)
    F = free_energy(agent.state_inference, agent.model, prior, observation, qs)

    actual_policies = policies === nothing ?
        collect(enumerate_policies(agent.policy_inference, agent.model, qs)) :
        collect(policies)
    q_pi, G = posterior_policies(agent.policy_inference, agent.model, qs, actual_policies)

    action = sample_action(agent.action_selection, q_pi, actual_policies, agent.rng)

    step = AgentStep(observation, qs, actual_policies, q_pi, G, F, action)
    push_step!(agent.history, step)

    if agent.parameter_learning !== nothing
        agent.model = infer_parameters(agent.parameter_learning, agent.model, agent.history)
    end

    return action
end

# Predict the prior over states for the next step. On the first step, this is
# the model's state_prior; subsequently, it's `predict_states(m, last_belief, last_action)`.
function _compute_prior(m::GenerativeModel, h::AgentHistory)
    isempty(h) && return state_prior(m)
    return predict_states(m, current_belief(h), last_action(h))
end

"""
    reset!(agent::Agent)

Clear the agent's history and reinitialize the prior from the (possibly
parameter-updated) model. Useful between trials when running parameter recovery.
"""
function reset!(agent::Agent)
    agent.history = AgentHistory(state_prior(agent.model))
    return agent
end
