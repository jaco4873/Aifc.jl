# Tests for entropy / kl_divergence / cross_entropy / bayesian_surprise.
#
# Coverage:
#   1. Hand-computed special cases (uniform, delta, perfectly informative likelihood)
#   2. Mathematical invariants (non-negativity, Gibbs' inequality, etc.)
#   3. Algebraic identities (cross_entropy = entropy + KL; KL(p,p) = 0)
#   4. Boundary behavior (Inf when KL/cross-entropy diverges; 0·log(0) = 0)
#   5. Bayesian surprise = mutual information (alternative form: expected KL)
#   6. DimensionMismatch on size disagreement

@testset "entropy" begin
    @testset "hand-computed values" begin
        @test entropy([1.0, 0.0]) == 0.0          # delta
        @test entropy([0.0, 1.0]) == 0.0
        @test entropy([0.5, 0.5]) ≈ log(2)        # binary uniform
        @test entropy([1/3, 1/3, 1/3]) ≈ log(3)
        for n in 2:20
            @test entropy(fill(1/n, n)) ≈ log(n) atol=1e-12
        end
    end

    @testset "non-negativity" begin
        rng = Xoshiro(123)
        for _ in 1:100
            n = rand(rng, 1:30)
            p = softmax(randn(rng, n))
            @test entropy(p) >= 0
        end
    end

    @testset "uniform maximizes entropy" begin
        rng = Xoshiro(2024)
        for n in 2:10
            uniform = fill(1/n, n)
            for _ in 1:20
                non_uniform = softmax(randn(rng, n))
                @test entropy(uniform) >= entropy(non_uniform) - 1e-10
            end
        end
    end

    @testset "type promotion" begin
        @test entropy(Float32[0.5, 0.5]) isa Float32
        @test entropy(Float32[0.5, 0.5]) ≈ log(Float32(2)) atol=1f-7
        @test entropy([1, 0]) == 0.0          # Int input
        # Note: integer probability vectors are unusual, but we should not crash.
    end
end

@testset "kl_divergence" begin
    @testset "hand-computed values" begin
        @test kl_divergence([1.0, 0.0], [0.5, 0.5]) ≈ log(2)
        @test kl_divergence([0.5, 0.5], [0.5, 0.5]) == 0.0
        @test kl_divergence([0.0, 1.0], [0.5, 0.5]) ≈ log(2)

        # Asymmetric example
        # KL([0.7, 0.3] || [0.5, 0.5]) = 0.7·log(0.7/0.5) + 0.3·log(0.3/0.5)
        expected = 0.7 * log(0.7/0.5) + 0.3 * log(0.3/0.5)
        @test kl_divergence([0.7, 0.3], [0.5, 0.5]) ≈ expected
    end

    @testset "Inf when q has zero mass on p's support" begin
        @test isinf(kl_divergence([0.5, 0.5], [1.0, 0.0]))
        @test isinf(kl_divergence([1.0, 0.0], [0.0, 1.0]))
        @test isinf(kl_divergence([0.1, 0.9], [0.5, 0.0]))
    end

    @testset "0·log(0/0) = 0 convention" begin
        # When p[i] = 0 the term is 0 regardless of q[i].
        @test kl_divergence([0.0, 1.0], [0.0, 1.0]) == 0.0
        @test kl_divergence([0.0, 1.0], [0.5, 0.5]) ≈ log(2)
    end

    @testset "non-negativity (Gibbs' inequality)" begin
        rng = Xoshiro(456)
        for _ in 1:200
            n = rand(rng, 2:20)
            p = softmax(randn(rng, n))
            q = softmax(randn(rng, n))
            @test kl_divergence(p, q) >= -1e-12   # tolerance for fp roundoff
        end
    end

    @testset "KL(p, p) = 0" begin
        rng = Xoshiro(789)
        for _ in 1:50
            n = rand(rng, 2:20)
            p = softmax(randn(rng, n))
            @test kl_divergence(p, p) ≈ 0 atol=1e-10
        end
    end

    @testset "asymmetry" begin
        # KL is asymmetric in general. Note that for 2-element distributions with
        # `p = [a, 1-a]` and `q = [1-a, a]`, the two KLs *coincidentally* match
        # by permutation symmetry — pick a concretely asymmetric example.
        p = [0.9, 0.1]
        q = [0.5, 0.5]
        @test !isapprox(kl_divergence(p, q), kl_divergence(q, p); atol=1e-3)
    end

    @testset "DimensionMismatch" begin
        @test_throws DimensionMismatch kl_divergence([0.5, 0.5], [0.3, 0.4, 0.3])
    end
end

@testset "cross_entropy" begin
    @testset "identity: H(p,q) = H(p) + KL(p,q)" begin
        rng = Xoshiro(0xCE)
        for _ in 1:100
            n = rand(rng, 2:20)
            p = softmax(randn(rng, n))
            q = softmax(randn(rng, n))
            @test cross_entropy(p, q) ≈ entropy(p) + kl_divergence(p, q) atol=1e-10
        end
    end

    @testset "self-cross-entropy = entropy" begin
        rng = Xoshiro(0xCE2)
        for _ in 1:30
            n = rand(rng, 2:20)
            p = softmax(randn(rng, n))
            @test cross_entropy(p, p) ≈ entropy(p) atol=1e-10
        end
    end

    @testset "hand-computed" begin
        @test cross_entropy([0.5, 0.5], [0.5, 0.5]) ≈ log(2)
        @test cross_entropy([1.0, 0.0], [0.5, 0.5]) ≈ log(2)
        @test isinf(cross_entropy([1.0, 0.0], [0.0, 1.0]))
    end

    @testset "DimensionMismatch" begin
        @test_throws DimensionMismatch cross_entropy([0.5, 0.5], [0.3, 0.4, 0.3])
    end
end

@testset "bayesian_surprise" begin
    @testset "perfectly informative likelihood" begin
        # A = I: observation reveals the state; surprise = H[prior]
        A = [1.0 0.0; 0.0 1.0]
        @test bayesian_surprise(A, [0.5, 0.5]) ≈ log(2) atol=1e-12
        @test bayesian_surprise(A, [1.0, 0.0]) ≈ 0 atol=1e-12   # delta prior — nothing to gain
        @test bayesian_surprise(A, [0.0, 1.0]) ≈ 0 atol=1e-12

        # Wider example: A is the 4×4 identity, uniform prior
        A4 = Matrix{Float64}(I, 4, 4)
        @test bayesian_surprise(A4, fill(1/4, 4)) ≈ log(4) atol=1e-12
    end

    @testset "uninformative likelihood" begin
        # A's columns are identical: observation gives no information
        A = [0.5 0.5; 0.5 0.5]
        for qs in ([0.5, 0.5], [0.7, 0.3], [1.0, 0.0])
            @test abs(bayesian_surprise(A, qs)) < 1e-12
        end
    end

    @testset "asymmetric likelihood, hand-computed" begin
        A = [0.9 0.2; 0.1 0.8]
        qs = [0.5, 0.5]
        qo = A * qs                              # = [0.55, 0.45]
        H_qo = entropy(qo)
        H_qsA = 0.5 * entropy([0.9, 0.1]) + 0.5 * entropy([0.2, 0.8])
        expected = H_qo - H_qsA
        @test bayesian_surprise(A, qs) ≈ expected atol=1e-12
    end

    @testset "non-negativity" begin
        rng = Xoshiro(0xC11)
        for _ in 1:100
            numStates = rand(rng, 2:6)
            numObs = rand(rng, 2:6)
            A = hcat([softmax(randn(rng, numObs)) for _ in 1:numStates]...)
            qs = softmax(randn(rng, numStates))
            @test bayesian_surprise(A, qs) >= -1e-10
        end
    end

    @testset "alternative form: I[s;o] = E_q(o)[KL[q(s|o) || qs]]" begin
        # Bayesian surprise computed two ways:
        #   (i)  H[q(o)] - E_qs[H[A(:|s)]]              (the implementation)
        #   (ii) E_q(o)[KL[q(s|o) || qs]]                (the alternative form)
        # These must agree.
        rng = Xoshiro(0xC11A)
        for _ in 1:50
            numStates = rand(rng, 2:5)
            numObs = rand(rng, 2:5)
            A = hcat([softmax(randn(rng, numObs)) for _ in 1:numStates]...)
            qs = softmax(randn(rng, numStates))

            qo = A * qs
            ig_via_kl = 0.0
            for o in axes(A, 1)
                qo[o] > 0 || continue
                # Posterior over states given observation o
                post = (A[o, :] .* qs) ./ qo[o]
                ig_via_kl += qo[o] * kl_divergence(post, qs)
            end

            @test bayesian_surprise(A, qs) ≈ ig_via_kl atol=1e-10
        end
    end

    @testset "DimensionMismatch" begin
        A = [1.0 0.0; 0.0 1.0]
        @test_throws DimensionMismatch bayesian_surprise(A, [0.5, 0.3, 0.2])
    end
end
