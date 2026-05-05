# `FunctionalModel` — a `GenerativeModel` whose methods are user-supplied
# closures.
#
# Useful for:
#   - Rapid prototyping (no struct subtyping required)
#   - Custom dynamics that don't fit `DiscretePOMDP` cleanly
#   - Wrapping existing simulators
#   - Test fixtures
#
# All required `GenerativeModel` methods are passed as closures, plus
# `predict_states` (which the agent loop calls before each step). Optional
# methods fall back to the `GenerativeModel` defaults.

"""
    FunctionalModel(; state_prior, observation_distribution,
                       transition_distribution, log_preferences,
                       action_space, predict_states,
                       goal_state_prior = nothing,
                       state_factors = (1,),
                       observation_modalities = (1,))

A generic `GenerativeModel` defined by closures. Each closure has the
expected signature for the corresponding interface method:

- `state_prior() :: Distribution`
- `observation_distribution(s) :: Distribution`
- `transition_distribution(s, a) :: Distribution`
- `log_preferences(o) :: Real`
- `action_space() :: iterable`
- `predict_states(prior, action) :: Distribution`

# Example

```julia
m = FunctionalModel(
    state_prior            = () -> Categorical([0.5, 0.5]),
    observation_distribution = s -> Categorical(s == 1 ? [0.9, 0.1] : [0.1, 0.9]),
    transition_distribution  = (s, a) -> Categorical(a == 1 ? [s == 1 ? 1.0 : 0.0, s == 1 ? 0.0 : 1.0] : [0.5, 0.5]),
    log_preferences          = o -> o == 1 ? 1.0 : -1.0,
    action_space             = () -> [1, 2],
    predict_states           = (q, a) -> Categorical(probs(q)),  # placeholder
)
```
"""
struct FunctionalModel{F1, F2, F3, F4, F5, F6, GP, SF, OM} <: GenerativeModel
    state_prior_fn::F1
    observation_distribution_fn::F2
    transition_distribution_fn::F3
    log_preferences_fn::F4
    action_space_fn::F5
    predict_states_fn::F6
    goal_state_prior_value::GP
    state_factors_value::SF
    observation_modalities_value::OM
end

function FunctionalModel(; state_prior,
                            observation_distribution,
                            transition_distribution,
                            log_preferences,
                            action_space,
                            predict_states,
                            goal_state_prior = nothing,
                            state_factors    = (1,),
                            observation_modalities = (1,))
    return FunctionalModel(state_prior, observation_distribution,
                            transition_distribution, log_preferences,
                            action_space, predict_states,
                            goal_state_prior, state_factors, observation_modalities)
end

# --- Forward to user's closures ---

state_prior(m::FunctionalModel) = m.state_prior_fn()
observation_distribution(m::FunctionalModel, s) = m.observation_distribution_fn(s)
transition_distribution(m::FunctionalModel, s, a) = m.transition_distribution_fn(s, a)
log_preferences(m::FunctionalModel, o) = m.log_preferences_fn(o)
action_space(m::FunctionalModel) = m.action_space_fn()
predict_states(m::FunctionalModel, prior, action) = m.predict_states_fn(prior, action)

goal_state_prior(m::FunctionalModel) = m.goal_state_prior_value
state_factors(m::FunctionalModel) = m.state_factors_value
observation_modalities(m::FunctionalModel) = m.observation_modalities_value
