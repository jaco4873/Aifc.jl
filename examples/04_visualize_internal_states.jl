# Demo 4: Visualize the internal states of a sophisticated active-inference
# agent on the T-maze.
#
# Run with:
#     julia --project=examples examples/04_visualize_internal_states.jl
#
# Produces (under examples/plots/):
#   - 04_belief_evolution.png   posterior over (location, context) per step
#   - 04_free_energy.png         variational free energy over time, per agent
#   - 04_efe_decomposition.png   EFE = -(pragmatic + epistemic) per first action
#   - 04_policy_posterior.png    q(π) per step, sophisticated agent
#   - 04_reward_rate.png         success rate of sophisticated vs vanilla agent
#                                across many independent trials
#
# The interesting comparison is sophisticated (horizon=2) vs vanilla
# (horizon=1) on the same maze: vanilla commits to an arm directly because at
# depth 1 going to the cue has no pragmatic value. Sophisticated, simulating
# its own future posterior, sees that the cue WILL inform a future commit,
# and so chooses the cue first. The plots make that intuition concrete.

using Aifc
using Distributions: probs
using Plots
using StatsPlots: groupedbar
using Random: Xoshiro
using Statistics: mean

const OUT_DIR = joinpath(@__DIR__, "plots")
isdir(OUT_DIR) || mkpath(OUT_DIR)

const LOC_NAMES    = ["C", "Q", "L", "R"]
const CTX_NAMES    = ["L-context", "R-context"]
const ACTION_NAMES = ["stay", "go-Q", "go-L", "go-R"]
const OBS_NAMES    = ["neutral", "cue→L", "cue→R", "reward", "no-reward"]

# ---------------------------------------------------------------------------
# 1. Environment & rollout helpers
# ---------------------------------------------------------------------------

# Sample an observation from the model's likelihood A given the true state.
function sample_observation(rng, A::AbstractMatrix, s::Integer)
    p = @view A[:, s]
    r = rand(rng); acc = 0.0
    for o in eachindex(p)
        acc += p[o]
        r <= acc && return o
    end
    return length(p)
end

# Marginalize a single-factor T-maze posterior into (location, context).
function marginalize_tmaze(qs)
    p = probs(qs)
    loc = zeros(4); ctx = zeros(2)
    for s in 1:8
        loc[(s - 1) ÷ 2 + 1] += p[s]
        ctx[(s - 1) % 2 + 1] += p[s]
    end
    return loc, ctx
end

# Run one episode. Returns the agent's history along with the realized
# observations, the true state path, and the true context.
function rollout!(agent, m; true_context::Symbol, n_steps::Int = 3,
                  rng = Xoshiro(0))
    ctx_idx    = true_context == :L ? 1 : 2
    true_state = (1 - 1) * 2 + ctx_idx          # start at C
    obs_seq    = Int[]
    state_seq  = Int[true_state]

    for _ in 1:n_steps
        o = sample_observation(rng, m.A, true_state)
        push!(obs_seq, o)
        a = step!(agent, o)
        # Apply the (deterministic) transition to the true state.
        true_state = argmax(m.B[:, true_state, a])
        push!(state_seq, true_state)
    end
    return obs_seq, state_seq
end

# Did the agent get the reward? Reward observation is index 4. We count an
# episode as a success if it observed `reward` at any step.
got_reward(obs_seq) = any(==(4), obs_seq)

# ---------------------------------------------------------------------------
# 2. Trial-by-trial reward rate: sophisticated vs vanilla
# ---------------------------------------------------------------------------

function build_agent(; sophisticated::Bool, γ::Float64 = 16.0,
                       rng = Xoshiro(0))
    m = tmaze_model(cue_reliability = 0.95, preference = 4.0)
    planner = sophisticated ?
        SophisticatedInference(γ = γ, horizon = 2) :
        EnumerativeEFE(γ = γ, horizon = 1)
    a = Agent(
        m,
        FixedPointIteration(num_iter = 20, dF_tol = 1e-12),
        planner,
        Stochastic(α = γ);            # let action prob reflect q(π) for plots
        rng = rng,
    )
    return a, m
end

function reward_rate(; sophisticated::Bool, n_trials::Int = 200,
                       seed::Integer = 0xBEEF)
    rng = Xoshiro(seed)
    successes = 0
    for trial in 1:n_trials
        agent, m = build_agent(; sophisticated, rng = Xoshiro(rand(rng, UInt)))
        ctx = rand(rng, Bool) ? :L : :R
        obs, _ = rollout!(agent, m; true_context = ctx, n_steps = 3,
                                     rng = Xoshiro(rand(rng, UInt)))
        successes += got_reward(obs)
    end
    return successes / n_trials
end

# ---------------------------------------------------------------------------
# 3. Per-step belief / FE / EFE / q(π) extraction for a representative trial
# ---------------------------------------------------------------------------

function trace_internal_states(agent, m; true_context::Symbol,
                                n_steps::Int = 3, rng = Xoshiro(42))
    obs_seq, state_seq = rollout!(agent, m;
                                    true_context = true_context,
                                    n_steps = n_steps, rng = rng)

    # Per-step extracts
    loc_beliefs = zeros(4, n_steps)
    ctx_beliefs = zeros(2, n_steps)
    fe          = zeros(n_steps)
    efe         = zeros(length(agent.history.steps[1].expected_free_energies),
                         n_steps)
    q_pi        = zeros(size(efe))

    for (t, step) in enumerate(agent.history.steps)
        loc, ctx = marginalize_tmaze(step.belief)
        loc_beliefs[:, t] .= loc
        ctx_beliefs[:, t] .= ctx
        fe[t]             = step.free_energy
        efe[:, t]        .= step.expected_free_energies
        q_pi[:, t]       .= step.policy_posterior
    end

    actions = [s.action for s in agent.history.steps]
    return (; obs_seq, state_seq, actions,
              loc_beliefs, ctx_beliefs, fe, efe, q_pi)
end

# EFE decomposition for the FIRST step's policies — pragmatic vs epistemic.
function first_step_efe_decomposition(agent, m, qs0)
    policies = collect(enumerate_policies(agent.policy_inference, m, qs0))
    prag = [pragmatic_value(agent.policy_inference, m, qs0, π) for π in policies]
    epi  = [epistemic_value(agent.policy_inference, m, qs0, π) for π in policies]
    G    = -(prag .+ epi)
    return prag, epi, G, policies
end

# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------

function plot_belief_evolution(trace_soph, trace_van, true_context;
                                outpath::AbstractString)
    n_steps = size(trace_soph.loc_beliefs, 2)
    xs = 1:n_steps

    p1 = heatmap(xs, LOC_NAMES, trace_soph.loc_beliefs;
                 c = :viridis, clim = (0, 1),
                 title  = "Sophisticated · q(location)",
                 xlabel = "step", ylabel = "location",
                 colorbar_title = "prob")
    p2 = heatmap(xs, CTX_NAMES, trace_soph.ctx_beliefs;
                 c = :viridis, clim = (0, 1),
                 title  = "Sophisticated · q(context)   [true=$true_context]",
                 xlabel = "step", ylabel = "context",
                 colorbar_title = "prob")
    p3 = heatmap(xs, LOC_NAMES, trace_van.loc_beliefs;
                 c = :viridis, clim = (0, 1),
                 title  = "Vanilla · q(location)",
                 xlabel = "step", ylabel = "location",
                 colorbar_title = "prob")
    p4 = heatmap(xs, CTX_NAMES, trace_van.ctx_beliefs;
                 c = :viridis, clim = (0, 1),
                 title  = "Vanilla · q(context)   [true=$true_context]",
                 xlabel = "step", ylabel = "context",
                 colorbar_title = "prob")

    plt = plot(p1, p2, p3, p4; layout = (2, 2),
               size = (1100, 700), left_margin = 5Plots.mm,
               bottom_margin = 5Plots.mm)
    savefig(plt, outpath)
    return plt
end

function plot_free_energy(trace_soph, trace_van; outpath::AbstractString)
    n = size(trace_soph.loc_beliefs, 2)
    plt = plot(1:n, trace_soph.fe;
               marker = :circle, lw = 2, label = "sophisticated (h=2)",
               title = "Variational free energy F[q(s)] over time",
               xlabel = "step", ylabel = "F  (nats)",
               legend = :topright, size = (800, 500),
               left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
    plot!(plt, 1:n, trace_van.fe; marker = :square, lw = 2, ls = :dash,
          label = "vanilla (h=1)")
    savefig(plt, outpath)
    return plt
end

function plot_efe_decomposition(prag, epi, G, policies; outpath::AbstractString,
                                 title_suffix::String = "(sophisticated, step 1)")
    labels = [ACTION_NAMES[first(π)] for π in policies]
    bar_x  = 1:length(policies)

    # Stacked bar via matrix form: columns = stack layers.
    p1 = groupedbar(
        bar_x, hcat(prag, epi);
        label = ["pragmatic value" "epistemic value"],
        bar_position = :stack,
        xticks = (bar_x, labels),
        xlabel = "first action of policy", ylabel = "value (nats)",
        title  = "EFE decomposition · -G(π) = pragmatic + epistemic\n$title_suffix",
        legend = :topleft,
    )

    p2 = bar(bar_x, G;
             color = :auto,
             xticks = (bar_x, labels),
             xlabel = "first action of policy", ylabel = "G(π)  (nats)",
             title  = "Expected free energy G(π)  (lower is better)",
             legend = false)

    plt = plot(p1, p2; layout = (1, 2), size = (1200, 500),
               left_margin = 6Plots.mm, bottom_margin = 6Plots.mm)
    savefig(plt, outpath)
    return plt
end

function plot_policy_posterior(trace; outpath::AbstractString,
                                title::String = "Sophisticated agent · q(π) per step")
    n_pol, n_steps = size(trace.q_pi)
    labels = reshape([ACTION_NAMES[a] for a in 1:n_pol], 1, n_pol)
    # trace.q_pi' has rows = steps, columns = policies → grouped bars per step
    plt = groupedbar(
        1:n_steps, trace.q_pi';
        bar_position = :dodge,
        label = labels,
        xlabel = "step", ylabel = "q(π)",
        title  = title, ylims = (0, 1),
        size = (900, 500), legend = :topright,
        left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
    )
    savefig(plt, outpath)
    return plt
end

function plot_reward_rate(rates::AbstractDict; outpath::AbstractString)
    names = collect(keys(rates))
    vals  = [rates[k] for k in names]
    plt = bar(1:length(names), vals;
              xticks = (1:length(names), names),
              ylims  = (0, 1.05),
              ylabel = "reward rate",
              title  = "Reward rate over independent trials",
              legend = false, color = [:steelblue, :tomato],
              size = (650, 500), left_margin = 5Plots.mm,
              bottom_margin = 5Plots.mm)
    for (i, v) in enumerate(vals)
        annotate!(plt, i, v + 0.03, text(string(round(v; digits=3)), 10))
    end
    savefig(plt, outpath)
    return plt
end

# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

println("=" ^ 60)
println("Active-inference internals on the T-maze")
println("=" ^ 60)

# --- 5a. Single representative trial for both agents ------------------------

println("\n[1/3] Tracing a single trial for sophisticated and vanilla agents…")
true_context = :L

agent_s, m_s = build_agent(; sophisticated = true,  rng = Xoshiro(7))
agent_v, m_v = build_agent(; sophisticated = false, rng = Xoshiro(7))

trace_soph = trace_internal_states(agent_s, m_s;
                                    true_context = true_context,
                                    rng = Xoshiro(101))
trace_van  = trace_internal_states(agent_v, m_v;
                                    true_context = true_context,
                                    rng = Xoshiro(101))

println("  sophisticated actions: $(join([ACTION_NAMES[a] for a in trace_soph.actions], " → "))")
println("  vanilla       actions: $(join([ACTION_NAMES[a] for a in trace_van.actions],  " → "))")
println("  sophisticated obs:     $(join([OBS_NAMES[o] for o in trace_soph.obs_seq], " → "))")
println("  vanilla       obs:     $(join([OBS_NAMES[o] for o in trace_van.obs_seq],  " → "))")

plot_belief_evolution(trace_soph, trace_van, true_context;
                      outpath = joinpath(OUT_DIR, "04_belief_evolution.png"))
plot_free_energy(trace_soph, trace_van;
                  outpath = joinpath(OUT_DIR, "04_free_energy.png"))
plot_policy_posterior(trace_soph;
                       outpath = joinpath(OUT_DIR, "04_policy_posterior.png"))

# --- 5b. EFE decomposition at step 1 (uniform context prior) ----------------

println("\n[2/3] Computing EFE decomposition for the first decision…")
fresh_agent, fresh_m = build_agent(; sophisticated = true, rng = Xoshiro(0))
qs0 = state_prior(fresh_m)
prag, epi, G, policies = first_step_efe_decomposition(fresh_agent, fresh_m, qs0)
for (π, p, e, g) in zip(policies, prag, epi, G)
    println("  π=[$(ACTION_NAMES[first(π)])]:  prag=$(round(p; digits=3))  ",
            "epi=$(round(e; digits=3))  G=$(round(g; digits=3))")
end
plot_efe_decomposition(prag, epi, G, policies;
                        outpath = joinpath(OUT_DIR, "04_efe_decomposition.png"))

# --- 5c. Reward rate over many independent trials ---------------------------

println("\n[3/3] Estimating reward rate over independent trials…")
n_trials = 200
rate_soph = reward_rate(; sophisticated = true,  n_trials = n_trials,
                          seed = 0xBEEF)
rate_van  = reward_rate(; sophisticated = false, n_trials = n_trials,
                          seed = 0xBEEF)
println("  sophisticated (h=2):  $(round(rate_soph; digits=3))")
println("  vanilla       (h=1):  $(round(rate_van;  digits=3))")

plot_reward_rate(Dict("sophisticated\n(horizon=2)" => rate_soph,
                       "vanilla\n(horizon=1)"      => rate_van);
                  outpath = joinpath(OUT_DIR, "04_reward_rate.png"))

println("\nPlots written to $OUT_DIR")
