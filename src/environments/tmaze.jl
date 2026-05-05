# `tmaze_model` — canonical 4-location T-maze (Friston 2015 onward;
# Pezzulo, Rigoli, Friston 2018; Heins et al. 2022 pymdp T-maze tutorial).
#
# Setup:
#   Locations:     C (start/center) · Q (cue) · L (left arm) · R (right arm)
#   Contexts:      reward at L (ctx=L), or reward at R (ctx=R)
#   Hidden state:  (location, context), so |S| = 4 × 2 = 8
#   Observations:  neutral · cue→L · cue→R · reward · no-reward (5)
#   Actions:       stay · go-cue · go-left · go-right (4)
#
# At Q the agent reads a cue with reliability `cue_reliability` (default 0.95).
# The arms are ABSORBING — once the agent commits to L or R it stays there
# regardless of further actions. This is what makes the sophisticated-vs-
# vanilla distinction sharp at depth 2: a vanilla agent commits to an arm
# immediately and gets stuck with whatever it pays out (50-50 prior); a
# depth-2 agent goes to Q first to resolve context, then commits.
#
# State indexing (1-based): s = (loc-1) * 2 + ctx, with loc ∈ {1=C, 2=Q,
# 3=L, 4=R} and ctx ∈ {1=L, 2=R}. So s=1 is C/L-context, s=2 is C/R-context,
# etc. up to s=8 = R/R-context.

"""
    tmaze_model(; cue_reliability=0.95, preference=4.0)

Construct a canonical 4-location T-maze as a `DiscretePOMDP`. With default
parameters the cross-validation tests verify:

- Best vanilla 2-step total = `2 · log(2)` (a sequence of `[go-arm, go-arm]`)
- Sophisticated depth-2 picks `go-cue` first (action 2)
- Sophisticated depth-2 total = `0.495 + 3.798 ≈ 4.293`
- Sophisticated depth-1 picks an arm (no point in cueing if you can't use it)
"""
function tmaze_model(; cue_reliability::Real = 0.95,
                       preference::Real = 4.0)
    num_states = 8
    num_obs = 5
    num_actions = 4

    # A: P(o | s)
    A = zeros(Float64, num_obs, num_states)
    for loc in 0:3, ctx in 0:1
        s = loc * 2 + ctx + 1   # 1-based state index
        if loc == 0          # C (center)
            A[1, s] = 1.0    # neutral observation
        elseif loc == 1      # Q (cue)
            if ctx == 0
                A[2, s] = cue_reliability
                A[3, s] = 1 - cue_reliability
            else
                A[3, s] = cue_reliability
                A[2, s] = 1 - cue_reliability
            end
        elseif loc == 2      # L arm
            if ctx == 0
                A[4, s] = 1.0   # reward
            else
                A[5, s] = 1.0   # no reward
            end
        else                 # R arm
            if ctx == 1
                A[4, s] = 1.0   # reward
            else
                A[5, s] = 1.0   # no reward
            end
        end
    end

    # B: P(s' | s, a). Action labels: 1=stay, 2=go-Q, 3=go-L, 4=go-R.
    # Arms (loc=2, loc=3) are absorbing: stay regardless of action.
    B = zeros(Float64, num_states, num_states, num_actions)
    for a in 1:num_actions, loc in 0:3, ctx in 0:1
        s = loc * 2 + ctx + 1
        on_arm = (loc == 2 || loc == 3)
        next_loc = if on_arm
            loc                                # absorbing
        elseif a == 1
            loc                                # stay
        else
            a - 1                              # go to loc=(a-1): Q=1, L=2, R=3
        end
        sn = next_loc * 2 + ctx + 1
        B[sn, s, a] = 1.0
    end

    # C: log preferences. Reward (obs 4) preferred; no-reward (obs 5) dispreferred.
    C = zeros(Float64, num_obs)
    C[4] =  Float64(preference)
    C[5] = -Float64(preference)

    # D: start at C with uniform belief over context.
    D = zeros(Float64, num_states)
    D[1] = 0.5   # C / ctx-L
    D[2] = 0.5   # C / ctx-R

    return DiscretePOMDP(A, B, C, D; check=true)
end

# Action labels for the T-maze, exposed for readability in tests / demos.
const TMAZE_ACTIONS = (stay = 1, go_cue = 2, go_left = 3, go_right = 4)

# State indices for the T-maze, indexed as `tmaze_state(:Q, :L)` -> 3.
function tmaze_state_index(loc::Symbol, ctx::Symbol)
    loc_idx = loc == :C ? 0 : loc == :Q ? 1 : loc == :L ? 2 : loc == :R ? 3 :
              throw(ArgumentError("loc must be :C, :Q, :L, :R, got $loc"))
    ctx_idx = ctx == :L ? 0 : ctx == :R ? 1 :
              throw(ArgumentError("ctx must be :L, :R, got $ctx"))
    return loc_idx * 2 + ctx_idx + 1
end


"""
    tmaze_model_multi_factor(; cue_reliability=0.95, preference=4.0)

Construct the canonical T-maze as a `MultiFactorDiscretePOMDP` with state
factorized as `(location, context)`. Identical task semantics as
`tmaze_model()`; the natural multi-factor representation tests cleanly
against the 8-state single-factor version (see the cross-validation
testset).

# Factorization

- **State factor 1**: location ∈ {C=1, Q=2, L=3, R=4} (4 states, 4 actions:
  stay/go-Q/go-L/go-R; arms are absorbing)
- **State factor 2**: context ∈ {ctx-L=1, ctx-R=2} (2 states, 1 passive action)
- **Modality 1**: observation ∈ {neutral, cue→L, cue→R, reward, no-reward}

# Action format

`MultiFactorDiscretePOMDP` actions are `Vector{Int}` of length F=2. The
context factor only has 1 action (passive), so all action vectors have the
form `[a_location, 1]`. The constants `TMAZE_MF_ACTIONS` give the location
actions: e.g. `[TMAZE_MF_ACTIONS.go_cue, 1]` is "go-Q with passive context".
"""
function tmaze_model_multi_factor(; cue_reliability::Real = 0.95,
                                     preference::Real = 4.0)
    n_loc = 4
    n_ctx = 2
    n_obs = 5
    n_loc_actions = 4

    # A[1] is (n_obs, n_loc, n_ctx)
    A = zeros(Float64, n_obs, n_loc, n_ctx)
    for loc in 1:n_loc, ctx in 1:n_ctx
        if loc == 1            # C
            A[1, loc, ctx] = 1.0           # neutral
        elseif loc == 2        # Q (cue)
            if ctx == 1
                A[2, loc, ctx] = cue_reliability
                A[3, loc, ctx] = 1 - cue_reliability
            else
                A[3, loc, ctx] = cue_reliability
                A[2, loc, ctx] = 1 - cue_reliability
            end
        elseif loc == 3        # L arm
            A[ctx == 1 ? 4 : 5, loc, ctx] = 1.0
        else                   # R arm
            A[ctx == 2 ? 4 : 5, loc, ctx] = 1.0
        end
    end

    # B[1]: location dynamics (4 actions, arms absorbing)
    B1 = zeros(Float64, n_loc, n_loc, n_loc_actions)
    for a in 1:n_loc_actions, loc in 1:n_loc
        on_arm = (loc == 3 || loc == 4)
        next_loc = on_arm ? loc :
                   (a == 1 ? loc : a)        # actions 2/3/4 → loc 2/3/4
        B1[next_loc, loc, a] = 1.0
    end

    # B[2]: context dynamics — identity, single passive action
    B2 = zeros(Float64, n_ctx, n_ctx, 1)
    for ctx in 1:n_ctx
        B2[ctx, ctx, 1] = 1.0
    end

    C = [zeros(Float64, n_obs)]
    C[1][4] =  Float64(preference)
    C[1][5] = -Float64(preference)

    D = [
        Float64[1.0, 0.0, 0.0, 0.0],     # start at C
        Float64[0.5, 0.5],                # uniform context
    ]

    return MultiFactorDiscretePOMDP([A], [B1, B2], C, D)
end

const TMAZE_MF_ACTIONS = (stay = 1, go_cue = 2, go_left = 3, go_right = 4)
