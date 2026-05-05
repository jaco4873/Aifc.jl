# Type-hierarchy tests for the unified AbstractDiscretePOMDP supertype.
#
# Background: DiscretePOMDP (single-factor) and MultiFactorDiscretePOMDP
# previously had no shared supertype other than GenerativeModel, even
# though they're conceptually the same family — both are tabular A/B/C/D
# active-inference POMDPs with optional Dirichlet hyperparameters and
# goal-state priors. Algorithm methods had to be duplicated for each
# type.
#
# Phase 1 of the unification (this commit): introduce
# `AbstractDiscretePOMDP` as a shared abstract supertype. Users can write
# `function f(m::AbstractDiscretePOMDP)` and get polymorphism across both
# concrete representations.
#
# Phase 2 (deferred): consolidate the duplicated algorithm bodies onto
# the supertype. That's a larger refactor — the multi-factor algorithm
# methods need to handle F=1/M=1 fast paths cleanly before they can
# replace the single-factor methods.

using Aifc: AbstractDiscretePOMDP, DiscretePOMDP, MultiFactorDiscretePOMDP

@testset "AbstractDiscretePOMDP supertype: both concrete types subtype it" begin
    @test DiscretePOMDP            <: AbstractDiscretePOMDP
    @test MultiFactorDiscretePOMDP <: AbstractDiscretePOMDP

    # Both are still GenerativeModels (transitively, via the abstract).
    @test AbstractDiscretePOMDP    <: Aifc.GenerativeModel
    @test DiscretePOMDP            <: Aifc.GenerativeModel
    @test MultiFactorDiscretePOMDP <: Aifc.GenerativeModel
end

@testset "AbstractDiscretePOMDP enables polymorphic functions" begin
    # User-written function dispatching on AbstractDiscretePOMDP works
    # for both concrete subtypes — the value the abstract supertype delivers.
    pomdp_summary(m::AbstractDiscretePOMDP) = (
        n_factors    = length(state_factors(m)),
        n_modalities = length(observation_modalities(m)),
    )

    A = [0.9 0.1; 0.1 0.9]
    B = zeros(2, 2, 1); B[1,1,1] = 1; B[2,2,1] = 1
    C = [0.0, 0.0]; D = [0.5, 0.5]
    sf = DiscretePOMDP(A, B, C, D)
    mf = MultiFactorDiscretePOMDP(sf)

    @test pomdp_summary(sf) == (n_factors=1, n_modalities=1)
    @test pomdp_summary(mf) == (n_factors=1, n_modalities=1)
end
