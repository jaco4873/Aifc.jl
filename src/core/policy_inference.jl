# `PolicyInference` interface — minimization of expected free energy over policies.
#
# This is structurally distinct from `Inference` (which minimizes VFE on
# present-time latents). Policy inference minimizes *expected future* free
# energy `G(π)` under the agent's predicted observation distribution
# `q(o|π)`, then forms a posterior `q(π) ∝ exp(γ·(-G(π)) + ln E(π))`.
#
# Required methods:
#   - `posterior_policies(planner, m, qs, policies)` -> (q_pi, G_values)
#   - `expected_free_energy(planner, m, qs, π)` -> Real
#   - `pragmatic_value(planner, m, qs, π)` -> Real
#   - `epistemic_value(planner, m, qs, π)` -> Real
#
# The decomposition methods (`pragmatic_value`, `epistemic_value`) are
# *mandatory* — they're how the conformance test suite verifies the
# implementation: G(π) ≈ -(pragmatic + epistemic) must hold.

"""
    abstract type PolicyInference end

Decision-theoretic algorithm: given a state posterior `qs`, return a
posterior over policies that minimizes expected free energy.

# Required methods

- `posterior_policies(planner, m, qs, policies)` →
   `(q_pi::Vector{<:Real}, G::Vector{<:Real})`

   `q_pi[k]` is the posterior probability of `policies[k]`, and `G[k]` is
   its expected free energy (in the convention where `G` is the quantity
   *minimized* — note this differs from the `total = -G` convention used
   in some teaching materials).

- `expected_free_energy(planner, m, qs, π)` → `Real`

   `G(π) = -pragmatic_value - epistemic_value`. Conformance tests verify
   this decomposition.

- `pragmatic_value(planner, m, qs, π)` → `Real`

   `E_q(o|π)[log P̃(o)]` — expected log-preference along the policy.

- `epistemic_value(planner, m, qs, π)` → `Real`

   Information gain about hidden states. Non-negative by construction.

# Optional methods

- `enumerate_policies(planner, m, qs)` → iterable of policies

   For planners that generate their own policy set (e.g. MCTS). Defaults to
   raising an error — most planners require an externally-supplied policy set.
"""
abstract type PolicyInference end

function posterior_policies(planner::PolicyInference, m::GenerativeModel, qs, policies)
    error("posterior_policies not implemented for $(typeof(planner))")
end

function expected_free_energy(planner::PolicyInference, m::GenerativeModel, qs, π)
    error("expected_free_energy not implemented for $(typeof(planner))")
end

function pragmatic_value(planner::PolicyInference, m::GenerativeModel, qs, π)
    error("pragmatic_value not implemented for $(typeof(planner))")
end

function epistemic_value(planner::PolicyInference, m::GenerativeModel, qs, π)
    error("epistemic_value not implemented for $(typeof(planner))")
end

"""
    enumerate_policies(planner::PolicyInference, m::GenerativeModel, qs)

Generate the set of policies to evaluate. By default this raises an error —
planners that require an externally-provided policy set should not implement
this method, and callers must pass `policies` explicitly. Planners that
generate their own (MCTS, sampled policies) override this.
"""
function enumerate_policies(planner::PolicyInference, m::GenerativeModel, qs)
    error("$(typeof(planner)) does not generate its own policy set; pass `policies` explicitly to `posterior_policies`")
end
