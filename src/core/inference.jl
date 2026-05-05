# `Inference` interface — variational inference over present-time latents.
#
# A concrete `Inference` algorithm implements:
#
#   - `infer_states(alg, m, prior, observation)`  —  per-step state inference
#   - `free_energy(alg, m, prior, observation, q)` —  VFE at the posterior
#                                                    (so the conformance
#                                                    suite can verify the
#                                                    monotone-decrease invariant)
#
# Trajectory-level parameter updates live on a separate abstract type
# `ParameterLearning` (see core/parameter_learning.jl) — they're a
# different operation with different inputs and outputs, and conflating
# them under one type just bloats the interface with capability flags.

"""
    abstract type Inference end

Variational state-inference algorithm. Concrete subtypes implement
`infer_states` and `free_energy`.

# Methods

- `infer_states(alg, m, prior, observation)` → posterior `Distribution`

  Single-step filtering: given the predicted prior and a new observation,
  return the posterior over hidden states by minimizing F[q].

- `free_energy(alg, m, prior, observation, q)` → `Real`

  Variational free energy at the given posterior. Conformance tests
  verify F[q] is non-increasing across iterations of any iterative
  algorithm.
"""
abstract type Inference end

# --- Default fallbacks ---

function infer_states(alg::Inference, m::GenerativeModel, prior, observation)
    error("infer_states not implemented for $(typeof(alg))")
end

function free_energy(alg::Inference, m::GenerativeModel, prior, observation, q)
    error("free_energy not implemented for $(typeof(alg))")
end
