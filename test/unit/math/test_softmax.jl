# Tests for softmax / logsoftmax.
#
# Coverage:
#   1. Basic correctness on hand-computed cases
#   2. Sums-to-1 invariant on random inputs
#   3. Shift invariance (additive constant doesn't affect output)
#   4. Numerical stability under large positive / large negative / extreme inputs
#   5. Tempered softmax: β = 1, β = 0, β → ±∞ limits
#   6. Type promotion (Int → Float64, Float32 → Float32)
#   7. logsoftmax / softmax consistency: exp ∘ logsoftmax ≈ softmax
#   8. Edge cases (empty input, singleton, scalar β)

@testset "basic correctness" begin
    @test softmax([0.0, 0.0, 0.0]) ≈ [1/3, 1/3, 1/3]
    @test softmax([0.0]) == [1.0]
    @test softmax([100.0]) == [1.0]

    # softmax of log probabilities recovers the probabilities
    p = [0.1, 0.2, 0.3, 0.4]
    @test softmax(log.(p)) ≈ p atol=1e-12

    # Manually computed
    @test softmax([log(1.0), log(2.0), log(3.0)]) ≈ [1/6, 2/6, 3/6] atol=1e-12
end

@testset "sums to 1" begin
    rng = Xoshiro(0xC0FFEE)
    for _ in 1:100
        n = rand(rng, 1:50)
        x = randn(rng, n)
        p = softmax(x)
        @test sum(p) ≈ 1 atol=1e-12
        @test all(p .>= 0)
    end
end

@testset "shift invariance" begin
    rng = Xoshiro(42)
    for _ in 1:50
        x = randn(rng, 10)
        c = randn(rng) * 1000
        @test softmax(x) ≈ softmax(x .+ c) atol=1e-10
    end
end

@testset "numerical stability" begin
    # Large positive inputs (would overflow naive exp)
    p = softmax([1000.0, 1001.0])
    @test sum(p) ≈ 1
    @test all(isfinite, p)
    @test p[2] > p[1]

    # Large negative inputs (would underflow)
    p = softmax([-1000.0, -1001.0])
    @test sum(p) ≈ 1
    @test all(isfinite, p)
    @test all(p .>= 0)
    @test p[1] > p[2]

    # Mixed extremes — output is essentially one-hot
    p = softmax([1e10, -1e10])
    @test p[1] ≈ 1.0
    @test p[2] ≈ 0.0 atol=1e-300

    # Tiny differences in large values still resolve
    p = softmax([1000.0, 1000.0 + 1e-3])
    @test p[2] > p[1]
    @test sum(p) ≈ 1
end

@testset "tempered softmax" begin
    x = [1.0, 2.0, 3.0]

    # β = 1 ≡ standard softmax
    @test softmax(x, 1.0) ≈ softmax(x)
    @test softmax(x, 1) ≈ softmax(x)

    # β = 0 → uniform
    p_uniform = softmax(x, 0.0)
    @test all(p_uniform .≈ 1/3)
    @test sum(p_uniform) ≈ 1

    # β → ∞ approaches one-hot at argmax(x)
    p_sharp = softmax(x, 1e6)
    @test p_sharp[3] ≈ 1.0
    @test p_sharp[2] < 1e-100
    @test p_sharp[1] < 1e-100

    # β → -∞ flips the argmax to argmin
    p_inv = softmax(x, -1e6)
    @test p_inv[1] ≈ 1.0
    @test p_inv[3] < 1e-100

    # Algebraic relation: softmax(x, β) = softmax(β·x)
    rng = Xoshiro(7)
    for _ in 1:20
        x = randn(rng, 5)
        β = 2 * randn(rng)
        @test softmax(x, β) ≈ softmax(β .* x) atol=1e-12
    end
end

@testset "type promotion" begin
    # Integer input → Float64 output
    p = softmax([1, 2, 3])
    @test eltype(p) == Float64
    @test sum(p) ≈ 1

    # Float32 input → Float32 output
    p32 = softmax(Float32[1, 2, 3])
    @test eltype(p32) == Float32
    @test sum(p32) ≈ 1f0 atol=1f-6

    # Float64 input → Float64 output
    p64 = softmax([1.0, 2.0, 3.0])
    @test eltype(p64) == Float64

    # Mixed scalar β, Int x — should broadcast / promote correctly
    @test softmax([1, 2], 2) ≈ softmax([2.0, 4.0]) atol=1e-12
end

@testset "logsoftmax" begin
    # exp ∘ logsoftmax ≈ softmax
    rng = Xoshiro(7)
    for _ in 1:50
        x = randn(rng, 10)
        @test exp.(logsoftmax(x)) ≈ softmax(x) atol=1e-10
    end

    # Hand-computed
    @test logsoftmax([0.0, 0.0, 0.0]) ≈ fill(-log(3.0), 3) atol=1e-12

    # Tempered logsoftmax: logsoftmax(x, β) = logsoftmax(β·x)
    @test logsoftmax([1.0, 2.0], 0.5) ≈ logsoftmax([0.5, 1.0]) atol=1e-12

    # Numerical stability for logsoftmax (the original motivation)
    x = [1000.0, 1001.0, 1002.0]
    ls = logsoftmax(x)
    @test all(isfinite, ls)
    @test sum(exp, ls) ≈ 1 atol=1e-12
end

@testset "edge cases" begin
    @test_throws ArgumentError softmax(Float64[])
    @test_throws ArgumentError logsoftmax(Float64[])

    # Singleton input is well-defined: σ([c]) = [1.0] for any c
    @test softmax([0.0]) == [1.0]
    @test softmax([-1e10]) == [1.0]
    @test logsoftmax([0.0]) == [0.0]
end
