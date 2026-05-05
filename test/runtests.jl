using Aifc
using Test
using Random
using LinearAlgebra: I
using FiniteDifferences

const TEST_DIR = @__DIR__

# Helper available to all test files: numerically-stable log-sum-exp.
# We don't pull this from LogExpFunctions in tests so the test code doesn't
# accidentally rely on the same implementation it's validating against.
function _logsumexp_test(x::AbstractVector{<:Real})
    m = maximum(x)
    return m + log(sum(exp(xi - m) for xi in x))
end

@testset "Aifc" begin
    @testset "Aqua (code quality)" begin
        include(joinpath(TEST_DIR, "aqua.jl"))
    end

    @testset "math primitives" begin
        @testset "softmax" begin
            include(joinpath(TEST_DIR, "unit", "math", "test_softmax.jl"))
        end
        @testset "information" begin
            include(joinpath(TEST_DIR, "unit", "math", "test_information.jl"))
        end
        @testset "free energy" begin
            include(joinpath(TEST_DIR, "unit", "math", "test_free_energy.jl"))
        end
        @testset "gradients" begin
            include(joinpath(TEST_DIR, "unit", "math", "test_gradients.jl"))
        end
    end

    @testset "core (interface contracts + helpers)" begin
        include(joinpath(TEST_DIR, "unit", "core", "test_interface_contracts.jl"))
        include(joinpath(TEST_DIR, "unit", "core", "test_beliefs_helpers.jl"))
        include(joinpath(TEST_DIR, "unit", "core", "test_history_accessors.jl"))
        include(joinpath(TEST_DIR, "unit", "core", "test_agent_step_eltype.jl"))
    end

    @testset "models" begin
        @testset "DiscretePOMDP" begin
            include(joinpath(TEST_DIR, "unit", "models", "test_discrete_pomdp.jl"))
        end
        @testset "FunctionalModel" begin
            include(joinpath(TEST_DIR, "unit", "models", "test_functional.jl"))
        end
        @testset "Multi-factor DiscretePOMDP" begin
            include(joinpath(TEST_DIR, "unit", "models", "test_multi_factor_pomdp.jl"))
        end
        @testset "Multi-factor POMDP validation" begin
            include(joinpath(TEST_DIR, "unit", "models", "test_multi_factor_pomdp_validation.jl"))
        end
    end

    @testset "inference" begin
        @testset "FixedPointIteration" begin
            include(joinpath(TEST_DIR, "unit", "inference", "test_fixed_point.jl"))
        end
        @testset "FixedPointIteration (multi-factor)" begin
            include(joinpath(TEST_DIR, "unit", "inference", "test_fixed_point_multi_factor.jl"))
        end
    end

    @testset "learning" begin
        @testset "DirichletConjugate" begin
            include(joinpath(TEST_DIR, "unit", "learning", "test_dirichlet_conjugate.jl"))
        end
        @testset "DirichletConjugate (incremental)" begin
            include(joinpath(TEST_DIR, "unit", "learning", "test_dirichlet_conjugate_incremental.jl"))
        end
        @testset "DirichletConjugate (multi-factor)" begin
            include(joinpath(TEST_DIR, "unit", "learning", "test_dirichlet_conjugate_multi_factor.jl"))
        end
    end

    @testset "planning" begin
        @testset "EnumerativeEFE" begin
            include(joinpath(TEST_DIR, "unit", "planning", "test_enumerative.jl"))
        end
        @testset "EnumerativeEFE (multi-factor)" begin
            include(joinpath(TEST_DIR, "unit", "planning", "test_enumerative_multi_factor.jl"))
        end
        @testset "SophisticatedInference" begin
            include(joinpath(TEST_DIR, "unit", "planning", "test_sophisticated.jl"))
        end
        @testset "InductiveEFE" begin
            include(joinpath(TEST_DIR, "unit", "planning", "test_inductive.jl"))
        end
    end

    @testset "action selection" begin
        include(joinpath(TEST_DIR, "unit", "action", "test_action_selectors.jl"))
    end

    @testset "gradient flow through parametric configs" begin
        include(joinpath(TEST_DIR, "unit", "action", "test_gradient_flow.jl"))
    end

    @testset "golden / pymdp cross-validation" begin
        @testset "T-maze" begin
            include(joinpath(TEST_DIR, "golden", "test_pymdp_tmaze.jl"))
        end
        @testset "random POMDP" begin
            include(joinpath(TEST_DIR, "golden", "test_pymdp_random.jl"))
        end
    end

    @testset "property tests on random POMDPs" begin
        include(joinpath(TEST_DIR, "property", "test_random_pomdp_invariants.jl"))
    end

    @testset "SBC / parameter recovery (slow)" begin
        if get(ENV, "AIF_RUN_SBC", "0") == "1"
            include(joinpath(TEST_DIR, "sbc", "test_parameter_recovery.jl"))
        else
            @info "SBC tests skipped (set AIF_RUN_SBC=1 to enable)"
        end
    end

    @testset "integration" begin
        @testset "Agent loop" begin
            include(joinpath(TEST_DIR, "unit", "integration", "test_agent_loop.jl"))
        end
        @testset "T-maze reference cross-validation" begin
            include(joinpath(TEST_DIR, "unit", "integration", "test_tmaze_reference.jl"))
        end
        @testset "T-maze multi-factor equivalence" begin
            include(joinpath(TEST_DIR, "unit", "integration", "test_tmaze_multi_factor.jl"))
        end
        @testset "Agent loop (multi-factor)" begin
            include(joinpath(TEST_DIR, "unit", "integration", "test_agent_loop_multi_factor.jl"))
        end
        @testset "ActionModels.jl extension" begin
            include(joinpath(TEST_DIR, "unit", "integration", "test_actionmodels.jl"))
        end
        @testset "ActionModels.jl extension (multi-factor)" begin
            include(joinpath(TEST_DIR, "unit", "integration", "test_actionmodels_multi_factor.jl"))
        end
    end
end
