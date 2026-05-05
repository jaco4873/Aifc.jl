# Package extension implementing the ActionModels.jl integration.
#
# Loaded automatically when both Aifc and ActionModels are
# present in the environment. Provides:
#
#   - `Aifc.active_inference_actionmodel(; ...)` — the user-
#     facing constructor stubbed in `src/integrations/actionmodels.jl`
#
# Internal types (`_ActiveInferenceSubmodel`, `_ActiveInferenceAttributes`)
# are private to this extension. Users interact with the integration through
# the public `active_inference_actionmodel` function and the standard
# ActionModels API (`init_agent`, `simulate!`, `observe!`, `get_parameters`,
# `set_parameters!`, `reset!`).
#
# Architecture: see the docstring on `active_inference_actionmodel` and the
# `AifcActionModelsExt` README.

module AifcActionModelsExt

using Aifc
using Aifc: GenerativeModel, Inference, PolicyInference, ActionSelector
using Aifc: state_prior, predict_states
using Aifc: infer_states, free_energy
using Aifc: enumerate_policies, posterior_policies
using Aifc: action_distribution

using ActionModels
using ActionModels: ActionModel, Parameter, State, Observation, Action
using ActionModels: ModelAttributes, AbstractSubmodel, AbstractSubmodelAttributes
using ActionModels: AttributeError
using ActionModels: load_parameters
import ActionModels:
    initialize_attributes,
    reset!,
    get_parameter_types,
    get_state_types,
    get_parameters,
    set_parameters!,
    get_states,
    set_states!

using Distributions: Categorical


# =============================================================================
# Internal submodel types (not exported; users interact via the function)
# =============================================================================

"""
Internal `AbstractSubmodel` wrapping the four pieces of an active-inference
agent. Constructed by `active_inference_actionmodel`; users don't see this
type directly.
"""
struct _ActiveInferenceSubmodel{M<:GenerativeModel,
                                  SI<:Inference,
                                  PI<:PolicyInference,
                                  AS<:ActionSelector} <: AbstractSubmodel
    model::M
    state_inference::SI
    policy_inference::PI
    action_selection::AS
end

mutable struct _ActiveInferenceAttributes{M, SI, PI, AS, B} <: AbstractSubmodelAttributes
    # Live mutable copies of the configurations. The `action_selection` and
    # `policy_inference` get reconstructed each step from the latest parameter
    # samples — see `_aif_action_model_fn`.
    model::M
    state_inference::SI
    policy_inference::PI
    action_selection::AS

    # Last belief — needed for predict_states on the next step
    last_belief::B

    # Last free energy — exposed as the ActionModels state `:free_energy`
    last_F::Float64
end


# =============================================================================
# Required AbstractSubmodel dispatches
# =============================================================================

function initialize_attributes(submodel::_ActiveInferenceSubmodel,
                                 ::Type{TF} = Float64,
                                 ::Type{TI} = Int64) where {TF, TI}
    sp = state_prior(submodel.model)
    return _ActiveInferenceAttributes(
        submodel.model,
        submodel.state_inference,
        submodel.policy_inference,
        submodel.action_selection,
        sp,
        0.0,
    )
end

function reset!(attrs::_ActiveInferenceAttributes)
    attrs.last_belief = state_prior(attrs.model)
    attrs.last_F = 0.0
    return nothing
end

function get_parameter_types(::_ActiveInferenceSubmodel)
    # `action_precision` / `policy_precision` are owned at the top level (so
    # that Turing's parameter sampling and `set_parameters!` flow through a
    # single source of truth). Returning `(;)` here avoids the
    # `merge(top_level, submodel)` conflict that would otherwise let stale
    # submodel state override the top-level Variable.
    return (;)
end

function get_state_types(::_ActiveInferenceSubmodel)
    return (free_energy = Float64,)
end


# Parameter introspection — defer to top level
function get_parameters(::_ActiveInferenceAttributes)
    return (;)
end

function get_parameters(::_ActiveInferenceAttributes, ::Symbol)
    return AttributeError()
end

function get_parameters(::_ActiveInferenceAttributes,
                          ::Tuple{Vararg{Symbol}})
    return AttributeError()
end

# Parameter mutation — defer to top level (signals AttributeError so the
# higher-level `set_parameters!` falls back to top-level Variable update)
function set_parameters!(::_ActiveInferenceAttributes,
                          ::Symbol,
                          ::Union{R,AbstractArray{R}}) where {R<:Real}
    return AttributeError()
end

function set_parameters!(::_ActiveInferenceAttributes,
                          ::Tuple{Vararg{Symbol}},
                          ::Tuple)
    return AttributeError()
end

# State introspection
function get_states(attrs::_ActiveInferenceAttributes)
    return (free_energy = attrs.last_F,)
end

function get_states(attrs::_ActiveInferenceAttributes, name::Symbol)
    if name === :free_energy
        return attrs.last_F
    else
        return AttributeError()
    end
end

function get_states(attrs::_ActiveInferenceAttributes,
                      names::Tuple{Vararg{Symbol}})
    return NamedTuple(name => get_states(attrs, name) for name in names)
end

function set_states!(attrs::_ActiveInferenceAttributes,
                      name::Symbol,
                      value::Union{R,AbstractArray{R}}) where {R<:Real}
    if name === :free_energy
        attrs.last_F = Float64(value)
    else
        return AttributeError()
    end
    return true
end

function set_states!(attrs::_ActiveInferenceAttributes,
                      names::Tuple{Vararg{Symbol}},
                      values::Tuple)
    for (name, value) in zip(names, values)
        out = set_states!(attrs, name, value)
        out isa AttributeError && error("State $name not found.")
    end
    return nothing
end


# =============================================================================
# Per-algorithm parameter accessors
# =============================================================================

# Action precision (α)
_get_action_precision(s::Aifc.Stochastic) = s.α
_get_action_precision(::Aifc.Deterministic) = Inf
_with_action_precision(::Aifc.Stochastic, α::Real) =
    Aifc.Stochastic(α = α)
_with_action_precision(s::Aifc.Deterministic, ::Real) = s   # no-op

# Policy precision (γ)
_get_policy_precision(p::Aifc.EnumerativeEFE)         = p.γ
_get_policy_precision(p::Aifc.SophisticatedInference) = p.γ
_get_policy_precision(p::Aifc.InductiveEFE)           = _get_policy_precision(p.base)

_with_policy_precision(p::Aifc.EnumerativeEFE, γ::Real) =
    Aifc.EnumerativeEFE(γ = γ,
                                        horizon = p.horizon,
                                        use_pragmatic = p.use_pragmatic,
                                        use_epistemic = p.use_epistemic)

_with_policy_precision(p::Aifc.SophisticatedInference, γ::Real) =
    Aifc.SophisticatedInference(γ = γ,
                                                  horizon = p.horizon,
                                                  use_pragmatic = p.use_pragmatic,
                                                  use_epistemic = p.use_epistemic)

function _with_policy_precision(p::Aifc.InductiveEFE, γ::Real)
    new_base = _with_policy_precision(p.base, γ)
    return Aifc.InductiveEFE(new_base; goal_weight = p.goal_weight)
end


# =============================================================================
# Action model function
# =============================================================================

"""
Internal action-model function called by ActionModels' `observe!` each step.
Reads the current parameter sample, runs perception + planning, returns the
action distribution (`Categorical`) for ActionModels to sample / score.
"""
function _aif_action_model_fn(attributes::ModelAttributes, observation)
    aif = attributes.submodel
    params = load_parameters(attributes)

    # 1. Apply parameter samples to the typed configs
    if hasproperty(params, :action_precision)
        aif.action_selection = _with_action_precision(aif.action_selection,
                                                       params.action_precision)
    end
    if hasproperty(params, :policy_precision)
        aif.policy_inference = _with_policy_precision(aif.policy_inference,
                                                        params.policy_precision)
    end

    # 2. Predict prior from previous action stored in attributes (or model
    #    state_prior on step 1)
    prev_action = _previous_action(attributes)
    prior = if ismissing(prev_action)
        state_prior(aif.model)
    else
        predict_states(aif.model, aif.last_belief, prev_action)
    end

    # 3. Perceive
    qs = infer_states(aif.state_inference, aif.model, prior, observation)
    F  = free_energy(aif.state_inference, aif.model, prior, observation, qs)

    # 4. Plan
    policies = collect(enumerate_policies(aif.policy_inference, aif.model, qs))
    q_pi, _  = posterior_policies(aif.policy_inference, aif.model, qs, policies)

    # 5. Cache + expose
    aif.last_belief = qs
    aif.last_F      = Float64(F)
    set_states!(attributes, :free_energy, Float64(F))

    return action_distribution(aif.action_selection, q_pi, policies)
end

# Pull the most-recently-stored action. On step 1 this is `missing` (set
# by `initialize_variables` for actions).
function _previous_action(attributes::ModelAttributes)
    if hasproperty(attributes.actions, :action)
        v = attributes.actions[:action].value
        return ismissing(v) ? missing : Int(v)
    end
    return missing
end


# =============================================================================
# Public entry point — fills in the stub from src/integrations/actionmodels.jl
# =============================================================================

function Aifc.active_inference_actionmodel(;
        model::GenerativeModel,
        state_inference::Inference,
        policy_inference::PolicyInference,
        action_selection::ActionSelector,
        action_precision_prior::Union{Nothing, Parameter} = nothing,
        policy_precision_prior::Union{Nothing, Parameter} = nothing,
        expose::Tuple{Vararg{Symbol}} = (:action_precision, :policy_precision))

    if action_selection isa Aifc.Deterministic
        @warn ("active_inference_actionmodel: `Deterministic` action selection produces a degenerate " *
               "likelihood under Turing fitting. Use `Stochastic(α=...)` for fitting.")
    end

    if model isa Aifc.MultiFactorDiscretePOMDP
        return _build_actionmodel_multi_factor(model, state_inference,
            policy_inference, action_selection,
            action_precision_prior, policy_precision_prior, expose)
    else
        return _build_actionmodel_single_factor(model, state_inference,
            policy_inference, action_selection,
            action_precision_prior, policy_precision_prior, expose)
    end
end

function _build_actionmodel_single_factor(model, state_inference, policy_inference,
                                            action_selection,
                                            action_precision_prior, policy_precision_prior,
                                            expose)
    submodel = _ActiveInferenceSubmodel(
        model, state_inference, policy_inference, action_selection,
    )
    apa = action_precision_prior === nothing ?
        Parameter(_get_action_precision(action_selection)) : action_precision_prior
    ppa = policy_precision_prior === nothing ?
        Parameter(_get_policy_precision(policy_inference)) : policy_precision_prior

    parameter_pairs = Pair{Symbol, Parameter}[]
    :action_precision in expose && push!(parameter_pairs, :action_precision => apa)
    :policy_precision in expose && push!(parameter_pairs, :policy_precision => ppa)
    parameters = NamedTuple(parameter_pairs)

    states       = (free_energy = State(0.0),)
    observations = (observation = Observation(Int64),)
    actions      = (action      = Action(Categorical),)

    return ActionModel(_aif_action_model_fn;
                        parameters    = parameters,
                        states        = states,
                        observations  = observations,
                        actions       = actions,
                        submodel      = submodel,
                        verbose       = false)
end

function _build_actionmodel_multi_factor(model::Aifc.MultiFactorDiscretePOMDP,
                                            state_inference, policy_inference,
                                            action_selection,
                                            action_precision_prior, policy_precision_prior,
                                            expose)
    F = length(model.D)
    M = length(model.A)

    submodel = _ActiveInferenceSubmodel(
        model, state_inference, policy_inference, action_selection,
    )
    apa = action_precision_prior === nothing ?
        Parameter(_get_action_precision(action_selection)) : action_precision_prior
    ppa = policy_precision_prior === nothing ?
        Parameter(_get_policy_precision(policy_inference)) : policy_precision_prior

    parameter_pairs = Pair{Symbol, Parameter}[]
    :action_precision in expose && push!(parameter_pairs, :action_precision => apa)
    :policy_precision in expose && push!(parameter_pairs, :policy_precision => ppa)
    parameters = NamedTuple(parameter_pairs)

    states = (free_energy = State(0.0),)

    # Per-modality observation declarations
    obs_names    = ntuple(m_ -> Symbol(:observation_, m_), M)
    obs_pairs    = [obs_names[m_] => Observation(Int64) for m_ in 1:M]
    observations = NamedTuple(obs_pairs)

    # Per-factor action declarations
    action_names = ntuple(f -> Symbol(:action_, f), F)
    action_pairs = [action_names[f] => Action(Categorical) for f in 1:F]
    actions      = NamedTuple(action_pairs)

    fn = _make_aif_action_model_fn_multi_factor(F, M, action_names)

    return ActionModel(fn;
                        parameters    = parameters,
                        states        = states,
                        observations  = observations,
                        actions       = actions,
                        submodel      = submodel,
                        verbose       = false)
end

# Closure factory: produces the action_model function for a multi-factor model
# with given F and M. Captures F, M, and the per-factor action attribute
# names so that we can read previous actions back out of attributes.actions.
function _make_aif_action_model_fn_multi_factor(F::Int, M::Int,
                                                  action_names::NTuple{F_, Symbol}) where F_
    return function (attributes::ModelAttributes, args...)
        length(args) == M || error(
            "active_inference_actionmodel (multi-factor): expected $M observation arguments, got $(length(args))")
        observation = collect(Int.(args))

        aif = attributes.submodel
        params = load_parameters(attributes)

        if hasproperty(params, :action_precision)
            aif.action_selection = _with_action_precision(aif.action_selection,
                                                           params.action_precision)
        end
        if hasproperty(params, :policy_precision)
            aif.policy_inference = _with_policy_precision(aif.policy_inference,
                                                            params.policy_precision)
        end

        prev_action = _previous_action_multi_factor(attributes, action_names)
        prior = if ismissing(prev_action)
            Aifc.state_prior(aif.model)
        else
            Aifc.predict_states(aif.model, aif.last_belief, prev_action)
        end

        qs = Aifc.infer_states(aif.state_inference, aif.model, prior, observation)
        F_val = Aifc.free_energy(aif.state_inference, aif.model, prior, observation, qs)

        policies = collect(Aifc.enumerate_policies(aif.policy_inference, aif.model, qs))
        q_pi, _  = Aifc.posterior_policies(aif.policy_inference, aif.model, qs, policies)

        aif.last_belief = qs
        aif.last_F      = Float64(F_val)
        ActionModels.set_states!(attributes, :free_energy, Float64(F_val))

        # Per-factor Categoricals (Product → tuple of components)
        dist = Aifc.action_distribution(aif.action_selection, q_pi, policies)
        # `dist` is a Distributions.Product; .v is the vector of marginals
        return Tuple(dist.v)
    end
end

function _previous_action_multi_factor(attributes::ModelAttributes,
                                          action_names::NTuple{F, Symbol}) where F
    actions = Int[]
    for name in action_names
        v = attributes.actions[name].value
        ismissing(v) && return missing
        push!(actions, Int(v))
    end
    return actions
end

end # module
