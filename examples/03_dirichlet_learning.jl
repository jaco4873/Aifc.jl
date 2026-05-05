# Demo 3: Dirichlet learning of A — the agent learns the observation
# likelihood from observations, given that it can identify which state
# it's in (via the deterministic action sequence).
#
# Run with:
#     julia --project=. examples/03_dirichlet_learning.jl
#
# Setup
# -----
# Two hidden states. The TRUE likelihood maps state 1 strongly to obs 1,
# state 2 strongly to obs 3. We force the agent through an alternating
# trajectory by using a `B` that flips the state on action 1 and
# initialising with `pD` = [1, 0] (state 1 with certainty).
#
# Why we need this: with a uniform initial `A`, observations alone don't
# tell the agent anything — so the agent's belief over s_t is determined
# entirely by `B` and the action history. By constructing a deterministic
# trajectory the agent can credit each observation to the right state,
# and pA accumulates correctly.

using Aifc
using Distributions: Categorical
using Random: Xoshiro

println("=" ^ 60)
println("Dirichlet learning of A from observations")
println("=" ^ 60)

# True likelihood: column j is P(o | s = j)
A_true = [0.9 0.1;
          0.1 0.1;
          0.0 0.8]

# B: the only action FLIPS state 1 ↔ state 2 deterministically
B = zeros(2, 2, 1)
B[2, 1, 1] = 1   # state 1 → state 2 under action 1
B[1, 2, 1] = 1   # state 2 → state 1 under action 1

C = [0.0, 0.0, 0.0]
D = [1.0, 0.0]                          # certainty: starts in state 1

# Agent's initial model: uninformative A, weak Dirichlet prior
A0 = fill(1/3, 3, 2)
pA = ones(3, 2)
m_agent = DiscretePOMDP(A0, B, C, D; pA=pA, check=false)

learner = DirichletConjugate(
    lr_pA=1.0, fr_pA=1.0,
    learn_pA=true, learn_pB=false, learn_pD=false,
    use_effective_A=false,
)
agent = Agent(
    m_agent,
    FixedPointIteration(),
    EnumerativeEFE(γ=1.0, horizon=1),
    Stochastic(α=1.0);
    parameter_learning=learner,
    rng=Xoshiro(0xCAFE),
)

# Simulate alternating-state trajectory. The world state alternates
# deterministically; the agent applies action 1 each step (the only
# action available), so its predicted prior also alternates.
function simulate_learning!(agent, A_true, n_steps)
    rng = Xoshiro(0x123)
    true_state = 1
    for t in 1:n_steps
        p = A_true[:, true_state]
        r = rand(rng); acc = 0.0
        obs = 0
        for o in eachindex(p)
            acc += p[o]
            if r <= acc; obs = o; break; end
        end
        obs == 0 && (obs = length(p))

        step!(agent, obs)
        true_state = true_state == 1 ? 2 : 1
    end
end

n_steps = 500
println("\nTrue A (each column is P(o | s)):")
println(round.(A_true; digits=3))
println("\nInitial agent A (uniform):")
println(round.(agent.model.A; digits=3))

simulate_learning!(agent, A_true, n_steps)

println("\nAgent A after $n_steps observations (Dirichlet mean of pA):")
println(round.(agent.model.A; digits=3))

err = sum(abs.(agent.model.A .- A_true))
println("\nL1 error vs true A: $(round(err; digits=4))")
println("(Should be small — under ~0.1 — since each state is observed ~$(n_steps ÷ 2) times.)")
