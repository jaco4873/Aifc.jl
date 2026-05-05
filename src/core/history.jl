# Agent history — typed per-step record.
#
# After each `step!`, an `AgentStep` is appended capturing what the agent
# observed, inferred, planned, and did. The history is a single source of
# truth for inference algorithms (which read the history) and is also
# what's serialized for offline analysis / parameter recovery.
#
# Design choice: ONE struct per step, atomic. We don't expose
# partially-constructed steps. If you need to inspect intermediate state
# during `step!`, use the algorithm-level callbacks (TODO in v0.2).

"""
    AgentStep

A complete record of one agent–environment interaction:

- `observation` — what the environment delivered
- `belief` — posterior over hidden states after seeing `observation`
- `policies` — the policies the planner evaluated
- `policy_posterior` — posterior `q(π)` over those policies
- `expected_free_energies` — `G(π)` for each evaluated policy
- `free_energy` — variational free energy at the inferred posterior
- `action` — the action sampled for execution

The numeric eltype `T<:Real` is parametric so that `Dual` / `TrackedReal`
values flow through unmodified during ForwardDiff / ReverseDiff fitting.
"""
struct AgentStep{O, B, P, A, T<:Real}
    observation::O
    belief::B
    policies::Vector{P}
    policy_posterior::Vector{T}
    expected_free_energies::Vector{T}
    free_energy::T
    action::A
end

function AgentStep(observation, belief, policies::AbstractVector,
                    q_pi::AbstractVector{<:Real},
                    G::AbstractVector{<:Real},
                    F::Real,
                    action)
    T = promote_type(eltype(q_pi), eltype(G), typeof(F))
    pols = collect(policies)
    return AgentStep{typeof(observation), typeof(belief), eltype(pols), typeof(action), T}(
        observation, belief, pols, Vector{T}(q_pi), Vector{T}(G), T(F), action,
    )
end

"""
    AgentHistory

Mutable chronological log. Maintains the agent's `initial_belief` (the
state prior at instantiation) and a vector of completed `AgentStep`s.
"""
mutable struct AgentHistory{B}
    initial_belief::B
    steps::Vector{AgentStep}
end

AgentHistory(initial_belief) = AgentHistory{typeof(initial_belief)}(initial_belief, AgentStep[])

Base.length(h::AgentHistory) = length(h.steps)
Base.isempty(h::AgentHistory) = isempty(h.steps)
Base.lastindex(h::AgentHistory) = length(h)
Base.getindex(h::AgentHistory, i) = h.steps[i]
Base.iterate(h::AgentHistory, args...) = iterate(h.steps, args...)

"""
    current_belief(h::AgentHistory)

Most recent posterior over hidden states. Returns `h.initial_belief` if no
step has been completed yet.
"""
current_belief(h::AgentHistory) = isempty(h) ? h.initial_belief : h.steps[end].belief

"""
    last_action(h::AgentHistory) -> Union{Action, Nothing}

Most recently executed action, or `nothing` if no step has been completed.
"""
last_action(h::AgentHistory) = isempty(h) ? nothing : h.steps[end].action

"""
    observation_history(h::AgentHistory) -> Vector

Time-ordered vector of observations.
"""
observation_history(h::AgentHistory) = [s.observation for s in h.steps]

"""
    belief_history(h::AgentHistory) -> Vector

Time-ordered vector of posteriors. Includes `h.initial_belief` as the first
entry (i.e., the belief BEFORE the first observation).
"""
function belief_history(h::AgentHistory)
    out = Any[h.initial_belief]
    for s in h.steps
        push!(out, s.belief)
    end
    return out
end

"""
    action_history(h::AgentHistory) -> Vector

Time-ordered vector of actions.
"""
action_history(h::AgentHistory) = [s.action for s in h.steps]

"""
    free_energy_history(h::AgentHistory) -> Vector{Float64}

Time series of variational free energies, one per step.
"""
free_energy_history(h::AgentHistory) = [s.free_energy for s in h.steps]

"""
    push_step!(h::AgentHistory, step::AgentStep)

Append a completed step to the history. Used internally by `Agent.step!`.
"""
push_step!(h::AgentHistory, step::AgentStep) = (push!(h.steps, step); h)
