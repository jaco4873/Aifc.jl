# Demo 5: Visualize how the Dirichlet conjugate learner reshapes the
# observation likelihood A from a uniform prior toward the true generative
# matrix.
#
# Run with:
#     julia --project=examples examples/05_visualize_learning.jl
#
# Produces (under examples/plots/):
#   - 05_A_snapshots.png       A matrix at t=0/10/100/500
#   - 05_A_convergence.png     L1 error and per-state KL[A_true(:,s) || A_t(:,s)]
#                              vs t (log-x); shows the agent's likelihood
#                              concentrating onto the true conditional.
#   - 05_pA_concentration.png  total Dirichlet pseudocount per (o, s) cell
#   - 05_free_energy.png       per-step variational free energy F over the
#                              learning trajectory, showing surprise driving
#                              parameter updates.
#
# Setup mirrors examples/03_dirichlet_learning.jl: two hidden states, three
# observations, an alternating-state environment driven by the only available
# action so the agent can credit each observation to the right state.

using Aifc
using Distributions: probs
using Plots
using Random: Xoshiro

const OUT_DIR = joinpath(@__DIR__, "plots")
isdir(OUT_DIR) || mkpath(OUT_DIR)

# ---------------------------------------------------------------------------
# 1. The teacher and the student
# ---------------------------------------------------------------------------

# True generative likelihood: column j is P(o | s = j).
const A_TRUE = [0.9 0.1;
                0.1 0.1;
                0.0 0.8]

# Two-state world; the only action deterministically swaps state.
function make_world()
    B = zeros(2, 2, 1)
    B[2, 1, 1] = 1
    B[1, 2, 1] = 1
    return B
end

function make_agent(rng = Xoshiro(0xCAFE))
    A0 = fill(1/3, 3, 2)                  # uninformative
    pA = ones(3, 2)                        # weakly informative Dirichlet prior
    B  = make_world()
    C  = zeros(3)
    D  = [1.0, 0.0]                        # known starting state

    m_agent = DiscretePOMDP(A0, B, C, D; pA = pA, check = false)
    learner = DirichletConjugate(
        lr_pA = 1.0, fr_pA = 1.0,
        learn_pA = true, learn_pB = false, learn_pD = false,
        use_effective_A = false,
    )
    return Agent(
        m_agent,
        FixedPointIteration(),
        EnumerativeEFE(γ = 1.0, horizon = 1),
        Stochastic(α = 1.0);
        parameter_learning = learner,
        rng = rng,
    )
end

# Sample an observation from A_true given the true state.
function sample_observation(rng, s)
    p = @view A_TRUE[:, s]
    r = rand(rng); acc = 0.0
    for o in eachindex(p)
        acc += p[o]
        r <= acc && return o
    end
    return length(p)
end

# ---------------------------------------------------------------------------
# 2. Run the simulation, snapshotting A and pA at each step
# ---------------------------------------------------------------------------

function simulate_with_snapshots!(agent; n_steps::Int = 500,
                                    rng = Xoshiro(0x123))
    A_snapshots  = Vector{Matrix{Float64}}(undef, n_steps + 1)
    pA_snapshots = Vector{Matrix{Float64}}(undef, n_steps + 1)
    A_snapshots[1]  = copy(agent.model.A)
    pA_snapshots[1] = copy(agent.model.pA)

    true_state = 1
    for t in 1:n_steps
        o = sample_observation(rng, true_state)
        step!(agent, o)
        A_snapshots[t + 1]  = copy(agent.model.A)
        pA_snapshots[t + 1] = copy(agent.model.pA)
        true_state = true_state == 1 ? 2 : 1
    end
    return A_snapshots, pA_snapshots
end

# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------

l1_error(A_t::AbstractMatrix, A_true::AbstractMatrix) = sum(abs.(A_t .- A_true))

# Per-state KL divergence: column-wise KL[ A_true(:, s) || A_t(:, s) ].
# (Defined column-wise because each column is a categorical over observations
# given a state; the row-wise mix is not a distribution.)
function per_state_kl(A_t::AbstractMatrix, A_true::AbstractMatrix; ε = 1e-12)
    @assert size(A_t) == size(A_true)
    n_s = size(A_true, 2)
    out = zeros(n_s)
    for s in 1:n_s
        for o in axes(A_true, 1)
            p = A_true[o, s]
            q = A_t[o, s]
            p > 0 && (out[s] += p * log(p / max(q, ε)))
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------

function plot_A_snapshots(A_snapshots; outpath, snapshot_steps = (0, 10, 100, 500))
    plots = []
    for t in snapshot_steps
        idx = t + 1                        # snapshots[1] is t=0
        A   = A_snapshots[idx]
        push!(plots, heatmap(
            ["s=1", "s=2"], ["o=1", "o=2", "o=3"], A;
            c = :viridis, clim = (0, 1),
            title = "A at t=$t",
            xlabel = "state", ylabel = "observation",
            aspect_ratio = :auto,
            colorbar = (t == last(snapshot_steps)),
        ))
    end
    # Bonus: ground truth on the right.
    push!(plots, heatmap(
        ["s=1", "s=2"], ["o=1", "o=2", "o=3"], A_TRUE;
        c = :viridis, clim = (0, 1),
        title = "A_true",
        xlabel = "state", ylabel = "observation",
        aspect_ratio = :auto,
    ))
    plt = plot(plots...; layout = (1, length(plots)),
               size = (1500, 380),
               left_margin = 4Plots.mm, bottom_margin = 5Plots.mm,
               top_margin = 3Plots.mm)
    savefig(plt, outpath)
    return plt
end

function plot_A_convergence(A_snapshots; outpath)
    # snapshots[1] is t=0 (uniform A); log-x can't show that, so plot
    # snapshots[2:end] starting at t=1.
    ts  = 1:(length(A_snapshots) - 1)
    snaps_after_t0 = A_snapshots[2:end]
    l1  = [l1_error(A, A_TRUE) for A in snaps_after_t0]
    kls = reduce(hcat, [per_state_kl(A, A_TRUE) for A in snaps_after_t0])

    p1 = plot(ts, l1;
              xlabel = "Dirichlet update t", ylabel = "Σ |A_t - A_true|",
              title  = "L1 error of A_t vs ground truth",
              lw = 2, legend = false, xscale = :log10)
    p2 = plot(ts, kls';
              xlabel = "Dirichlet update t",
              ylabel = "KL[ A_true(:, s) || A_t(:, s) ]",
              label  = ["state 1" "state 2"],
              title  = "Per-state KL divergence (column-wise)",
              lw = 2, xscale = :log10)
    plt = plot(p1, p2; layout = (1, 2), size = (1100, 480),
               left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
    savefig(plt, outpath)
    return plt
end

function plot_pA_concentration(pA_snapshots; outpath)
    # Normalize each column to a proportion: this is what determines the
    # Dirichlet-mean A and is invariant to the absolute count scale.
    final_pA = pA_snapshots[end]
    proportions = final_pA ./ sum(final_pA; dims = 1)

    p1 = heatmap(
        ["s=1", "s=2"], ["o=1", "o=2", "o=3"], proportions;
        c = :magma, clim = (0, 1),
        title = "Per-column proportions of pA  (after $(length(pA_snapshots) - 1) updates)",
        xlabel = "state", ylabel = "observation",
        colorbar_title = "share of column",
        aspect_ratio = :auto,
    )
    for s in 1:size(proportions, 2), o in 1:size(proportions, 1)
        annotate!(p1, s, o, text(string(round(proportions[o, s]; digits=3)),
                                  10, :white))
    end

    # Show effective concentration over time: total pseudocount per column.
    n_t = length(pA_snapshots)
    col_totals = reduce(hcat,
        [vec(sum(pA; dims = 1)) for pA in pA_snapshots])      # 2 × n_t
    p2 = plot(0:(n_t - 1), col_totals';
              xlabel = "Dirichlet update t", ylabel = "Σ_o pA(o, s)",
              label = ["state 1" "state 2"],
              title = "Effective concentration  (column totals of pA)",
              lw = 2, yscale = :log10,
              left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

    plt = plot(p1, p2; layout = (1, 2), size = (1200, 480),
               left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
    savefig(plt, outpath)
    return plt
end

function plot_free_energy(agent; outpath)
    F = free_energy_history(agent.history)
    plt = plot(1:length(F), F;
               lw = 1.2, alpha = 0.5, label = "F per step",
               xlabel = "step", ylabel = "F  (nats)",
               title  = "Variational free energy as the agent learns A",
               size = (900, 480),
               left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
    # 25-step rolling mean to make the trend visible through stochastic noise.
    w = 25
    rolling = [sum(F[max(1, i - w + 1):i]) / min(i, w) for i in eachindex(F)]
    plot!(plt, 1:length(F), rolling;
          lw = 2.5, color = :black, label = "rolling mean ($w steps)")
    savefig(plt, outpath)
    return plt
end

# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

println("=" ^ 60)
println("Dirichlet learning of A — visualizing convergence")
println("=" ^ 60)

agent = make_agent(Xoshiro(0xCAFE))
println("\nTrue A:")
display(round.(A_TRUE; digits=3))
println("\nInitial agent A (uniform):")
display(round.(agent.model.A; digits=3))

n_steps = 500
println("\nRunning $n_steps steps of online learning…")
A_snaps, pA_snaps = simulate_with_snapshots!(agent;
                                               n_steps = n_steps,
                                               rng     = Xoshiro(0x123))

println("\nFinal agent A:")
display(round.(agent.model.A; digits=3))
println("\nL1 error vs A_true: $(round(l1_error(agent.model.A, A_TRUE); digits=4))")

println("\nSaving plots…")
plot_A_snapshots(A_snaps;
                  outpath = joinpath(OUT_DIR, "05_A_snapshots.png"),
                  snapshot_steps = (0, 10, 100, n_steps))
plot_A_convergence(A_snaps;
                    outpath = joinpath(OUT_DIR, "05_A_convergence.png"))
plot_pA_concentration(pA_snaps;
                       outpath = joinpath(OUT_DIR, "05_pA_concentration.png"))
plot_free_energy(agent;
                  outpath = joinpath(OUT_DIR, "05_free_energy.png"))

println("\nPlots written to $OUT_DIR")
