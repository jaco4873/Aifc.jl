# Demo 1: classic T-maze with sophisticated inference.
#
# Run with:
#     julia --project=. examples/01_tmaze_sophisticated.jl
#
# What you'll see:
#   - Step 1: agent at C, observes neutral. Sophisticated depth-2 picks go-cue.
#   - Step 2: agent at Q, observes the cue, posterior over context resolves
#             to ~0.95 confidence in one direction.
#   - Step 3: agent commits to the matching arm and gets the reward.

using Aifc
using Distributions: probs

println("=" ^ 60)
println("T-maze with sophisticated inference (depth 2)")
println("=" ^ 60)

# Build the canonical T-maze
m = tmaze_model(cue_reliability=0.95, preference=4.0)
agent = Agent(
    m,
    FixedPointIteration(num_iter=20, dF_tol=1e-12),
    SophisticatedInference(γ=64.0, horizon=2),
    Deterministic(),
)

# Choose the true context — the agent doesn't know this
true_context = :L                     # rewards on the left arm
ctx_idx = true_context == :L ? 1 : 2

# Helper: location indices
LOC_NAMES = ("C", "Q", "L", "R")
ACTION_NAMES = ("stay", "go-Q", "go-L", "go-R")
OBS_NAMES = ("neutral", "cue→L", "cue→R", "reward", "no reward")

# Simulate the environment
function simulate_observation(state_idx::Int, A::Matrix)
    # A is (num_obs × num_states); sample observation from A[:, state_idx]
    p = A[:, state_idx]
    r = rand(); acc = 0.0
    for o in eachindex(p)
        acc += p[o]
        r <= acc && return o
    end
    return length(p)
end

function run_demo(agent, m, true_context, ctx_idx)
    loc = 1                                # 1=C
    true_state = (loc - 1) * 2 + ctx_idx
    println("\nTrue context: $true_context (state $true_state)\n")

    for t in 1:3
        obs = simulate_observation(true_state, m.A)
        a   = step!(agent, obs)

        qs = current_belief(agent.history)
        F  = agent.history.steps[end].free_energy

        p = probs(qs)
        loc_marg = zeros(4); ctx_marg = zeros(2)
        for s in 1:8
            loc_marg[(s - 1) ÷ 2 + 1] += p[s]
            ctx_marg[(s - 1) % 2 + 1] += p[s]
        end

        println("Step $t:")
        println("  observation: $(OBS_NAMES[obs]) (was at $(LOC_NAMES[loc]))")
        println("  action:      $(ACTION_NAMES[a])")
        println("  loc belief:  ", round.(loc_marg; digits=3), "  ", LOC_NAMES)
        println("  ctx belief:  ", round.(ctx_marg; digits=3), "  (L, R)")
        println("  free energy: $(round(F; digits=4))")
        println()

        next_loc_idx = argmax(m.B[:, true_state, a])
        true_state = next_loc_idx
        loc = (next_loc_idx - 1) ÷ 2 + 1
    end
end

run_demo(agent, m, true_context, ctx_idx)
