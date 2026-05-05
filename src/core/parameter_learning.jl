# `ParameterLearning` interface — trajectory-level parameter updates.
#
# Distinct from `Inference` (which does state inference per step). A
# `ParameterLearning` algorithm updates the generative model's parameters
# from a recorded `AgentHistory` — e.g. Dirichlet conjugate updates over
# A/B/D hyperparameters, or future gradient-based ML estimators.
#
# `Agent.step!` calls `infer_parameters(alg, model, history)` after each
# action and rebinds `agent.model = ...`.

"""
    abstract type ParameterLearning end

Trajectory-level parameter learning. Concrete subtypes implement
`infer_parameters(alg, m, history) -> updated_model`.
"""
abstract type ParameterLearning end

"""
    infer_parameters(alg::ParameterLearning, m::GenerativeModel, history::AgentHistory)
        -> updated_model::GenerativeModel

Apply one round of parameter learning. Concrete algorithms specify
whether this is per-step (incremental) or batch (re-fit-from-scratch);
see e.g. `DirichletConjugate`.
"""
function infer_parameters(alg::ParameterLearning, m::GenerativeModel, history)
    error("infer_parameters not implemented for $(typeof(alg))")
end
