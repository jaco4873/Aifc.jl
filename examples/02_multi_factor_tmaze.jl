# Demo 2: T-maze with the multi-factor (location × context) representation.
#
# Run with:
#     julia --project=. examples/02_multi_factor_tmaze.jl
#
# Demonstrates:
#   - The natural multi-factor encoding (4 location × 2 context states)
#   - Per-factor belief tracking — no need for marginalization at the end
#   - End-to-end agent loop with Vector{Int} actions
#
# Note: at present, multi-factor uses EnumerativeEFE (Sophisticated multi-
# factor is a follow-up). At horizon=2 vanilla picks an arm directly rather
# than going to the cue first; that's a known horizon-vs-replanning tradeoff.

using Aifc
using Distributions: probs

println("=" ^ 60)
println("T-maze with multi-factor (location × context) representation")
println("=" ^ 60)

m = tmaze_model_multi_factor(cue_reliability=0.95, preference=4.0)
agent = Agent(
    m,
    FixedPointIteration(num_iter=20, dF_tol=1e-12),
    EnumerativeEFE(γ=64.0, horizon=1),
    Deterministic(),
)

LOC_NAMES = ("C", "Q", "L", "R")
ACTION_NAMES = ("stay", "go-Q", "go-L", "go-R")
OBS_NAMES = ("neutral", "cue→L", "cue→R", "reward", "no reward")

# Drive the agent through a fixed observation sequence to demonstrate the
# multi-factor belief tracking
println()
for (t, obs) in enumerate([1, 2, 4])
    a = step!(agent, [obs])

    qs = current_belief(agent.history)
    loc_marg = probs(marginal(qs, 1))
    ctx_marg = probs(marginal(qs, 2))
    F = agent.history.steps[end].free_energy

    println("Step $t:")
    println("  observation:    $(OBS_NAMES[obs])")
    println("  loc action:     $(ACTION_NAMES[a[1]])")
    println("  loc belief:     ", round.(loc_marg; digits=3), "  ", LOC_NAMES)
    println("  ctx belief:     ", round.(ctx_marg; digits=3), "  (L, R)")
    println("  free energy:    $(round(F; digits=4))")
    println()
end
