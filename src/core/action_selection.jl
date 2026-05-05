# `ActionSelector` interface — sample an action given the policy posterior.

using Random: AbstractRNG, default_rng

"""
    abstract type ActionSelector end

Given a posterior `q(π)` over policies and the corresponding policy set,
return an action. Concrete subtypes determine how the policy posterior is
mapped to the action distribution and whether sampling is stochastic or
deterministic.

# Required methods

- `action_distribution(sel, q_pi, policies)` → `Distribution` over actions
- `sample_action(sel, q_pi, policies, rng)` → action

The default `sample_action` implementation samples from `action_distribution`,
so concrete subtypes only need to implement the latter for both APIs to work.
The action distribution is the hook for `ActionModels.jl` / Turing-based
parameter fitting, where the observed action's likelihood under this
distribution is what's being maximized.

# Optional methods

- `log_action_probabilities(sel, q_pi, policies)` → vector of log-probs
  over the action space (a convenience wrapper around `logpdf` of the
  distribution at every support point)
"""
abstract type ActionSelector end

"""
    action_distribution(sel::ActionSelector, q_pi, policies) -> Distribution

Return the `Distribution` over actions induced by `q_pi` (posterior over
policies). Sampling from this distribution is what `sample_action` does.
"""
function action_distribution(sel::ActionSelector, q_pi, policies)
    error("action_distribution not implemented for $(typeof(sel))")
end

function sample_action(sel::ActionSelector, q_pi, policies, rng::AbstractRNG)
    return rand(rng, action_distribution(sel, q_pi, policies))
end

function sample_action(sel::ActionSelector, q_pi, policies)
    return sample_action(sel, q_pi, policies, default_rng())
end

function log_action_probabilities(sel::ActionSelector, q_pi, policies)
    error("log_action_probabilities not implemented for $(typeof(sel))")
end
