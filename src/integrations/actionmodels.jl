# Stubs for the ActionModels.jl integration.
#
# The actual implementation lives in
# `ext/AifcActionModelsExt.jl` and is loaded automatically when
# both `Aifc` and `ActionModels` are present in the user's
# environment (Julia 1.9+ package-extension mechanism).
#
# We forward-declare the user-facing entry points here so `using
# Aifc` always exposes the names. Calling them without
# `ActionModels` loaded gives a helpful error.

"""
    active_inference_actionmodel(; model, state_inference, policy_inference,
                                    action_selection,
                                    action_precision_prior=nothing,
                                    policy_precision_prior=nothing,
                                    expose=(:action_precision, :policy_precision))

Build an `ActionModels.ActionModel` that wraps an active-inference agent.
The returned `ActionModel` is suitable for ActionModels' `init_agent`,
`simulate!`, and Turing-based fitting via `create_model` /
`sample_posterior!`.

# Keyword arguments

- `model::GenerativeModel` — the underlying generative model
- `state_inference::Inference` — must support state inference
- `policy_inference::PolicyInference` — `EnumerativeEFE`,
  `SophisticatedInference`, or `InductiveEFE`
- `action_selection::ActionSelector` — `Stochastic` for fittable agents,
  `Deterministic` for simulation-only
- `action_precision_prior` — a `Parameter(α₀)` to override the default
  prior on the action precision; default uses the value in `action_selection`
- `policy_precision_prior` — same for the planner's γ
- `expose::Tuple{Vararg{Symbol}}` — which agent parameters become Turing-
  sampleable parameters of the resulting `ActionModel`. Default exposes
  `:action_precision` and `:policy_precision`.

# v0.1 limitations

- Single observation modality, single action factor (`Action(Categorical)`).
- For Turing fitting, `Stochastic` action selection only — `Deterministic`
  produces a degenerate likelihood.
- Online Dirichlet learning is not supported via this entry point — call
  `step!` on a learning agent directly instead.
- `Float64` parameters only; propagation of `ForwardDiff.Dual` numbers
  through inference is a follow-up.

# Requires ActionModels.jl

Calling this function requires `ActionModels.jl` to be loaded. The
implementation lives in the `AifcActionModelsExt` package
extension, which Julia loads automatically when both packages are present.

# Example

```julia
using Aifc, ActionModels

m = tmaze_model()
am = active_inference_actionmodel(
    model            = m,
    state_inference  = FixedPointIteration(),
    policy_inference = SophisticatedInference(γ=8.0, horizon=2),
    action_selection = Stochastic(α=8.0),
)

agent = init_agent(am; save_history=true)
actions = simulate!(agent, [1, 2, 3])
```
"""
function active_inference_actionmodel end
