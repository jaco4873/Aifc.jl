# Gradient correctness via FiniteDifferences.jl.
#
# Strategy: compare hand-derived analytic gradients against high-order
# finite differences (central-5 stencil + Richardson extrapolation, ~1e-9
# accuracy at well-conditioned points). For functions where we don't have
# a clean analytic gradient (e.g. bayesian_surprise), we just verify the
# gradient is computable and finite — autodiff backend tests come in a
# later PR with `DifferentiationInterface.jl`.
#
# Tolerance is 1e-7: well below FiniteDifferences' 1e-9 reference accuracy
# but above floating-point roundoff at the inputs we test.

const FDM = central_fdm(5, 1)

@testset "softmax — analytic Jacobian" begin
    # ∂σ_i/∂x_j = σ_i (δ_ij - σ_j)  (the softmax Jacobian)
    rng = Xoshiro(0xBEEF)
    for _ in 1:10
        x = randn(rng, 6)
        p = softmax(x)
        for k in eachindex(x)
            fk(x) = softmax(x)[k]
            g_fd = grad(FDM, fk, x)[1]
            g_an = [p[k] * ((i == k) - p[i]) for i in eachindex(x)]
            @test g_fd ≈ g_an atol=1e-7
        end
    end
end

@testset "entropy gradient" begin
    # ∂H/∂p_i = -log p_i - 1   (treating p as a free vector, not a probability)
    rng = Xoshiro(0xE0CE)
    for _ in 1:10
        # Use a strictly positive vector well away from 0 — finite differences
        # diverge near p_i = 0 because of log singularity.
        p = softmax(randn(rng, 6))
        g_fd = grad(FDM, entropy, p)[1]
        g_an = -log.(p) .- 1
        @test g_fd ≈ g_an atol=1e-7
    end
end

@testset "kl_divergence gradient" begin
    # ∂KL/∂p_i = log(p_i/q_i) + 1     (free-variable form)
    # ∂KL/∂q_i = -p_i / q_i
    rng = Xoshiro(0xCAFE)
    for _ in 1:10
        p = softmax(randn(rng, 6))
        q = softmax(randn(rng, 6))

        f_p(p) = kl_divergence(p, q)
        g_p_fd = grad(FDM, f_p, p)[1]
        g_p_an = log.(p) .- log.(q) .+ 1
        @test g_p_fd ≈ g_p_an atol=1e-7

        f_q(q) = kl_divergence(p, q)
        g_q_fd = grad(FDM, f_q, q)[1]
        g_q_an = -p ./ q
        @test g_q_fd ≈ g_q_an atol=1e-7
    end
end

@testset "cross_entropy gradient" begin
    # ∂H(p, q)/∂q_i = -p_i / q_i
    # ∂H(p, q)/∂p_i = -log q_i
    rng = Xoshiro(0xC0E)
    for _ in 1:10
        p = softmax(randn(rng, 6))
        q = softmax(randn(rng, 6))

        f_q(q) = cross_entropy(p, q)
        g_q_fd = grad(FDM, f_q, q)[1]
        g_q_an = -p ./ q
        @test g_q_fd ≈ g_q_an atol=1e-7

        f_p(p) = cross_entropy(p, q)
        g_p_fd = grad(FDM, f_p, p)[1]
        g_p_an = -log.(q)
        @test g_p_fd ≈ g_p_an atol=1e-7
    end
end

@testset "variational_free_energy gradient w.r.t. q" begin
    # F = Σ q_i (log q_i - log P(s_i) - log P(o|s_i))
    # ∂F/∂q_i = log q_i + 1 - log P(s_i) - log P(o|s_i)
    rng = Xoshiro(0xFEED)
    for _ in 1:10
        n = rand(rng, 3:8)
        q = softmax(randn(rng, n))
        log_lik = randn(rng, n)
        log_prior = logsoftmax(randn(rng, n))

        f(q) = variational_free_energy(q, log_lik, log_prior)
        g_fd = grad(FDM, f, q)[1]
        g_an = log.(q) .+ 1 .- log_lik .- log_prior
        @test g_fd ≈ g_an atol=1e-7
    end
end

@testset "accuracy / complexity gradients" begin
    # ∂accuracy/∂q_i = log P(o|s_i)
    # ∂complexity/∂q_i = log q_i + 1 - log P(s_i)
    rng = Xoshiro(0xACC0)
    for _ in 1:10
        n = rand(rng, 3:8)
        q = softmax(randn(rng, n))
        log_lik = randn(rng, n)
        log_prior = logsoftmax(randn(rng, n))

        f_a(q) = accuracy(q, log_lik)
        @test grad(FDM, f_a, q)[1] ≈ log_lik atol=1e-7

        f_c(q) = complexity(q, log_prior)
        g_c_fd = grad(FDM, f_c, q)[1]
        g_c_an = log.(q) .+ 1 .- log_prior
        @test g_c_fd ≈ g_c_an atol=1e-7
    end
end

@testset "bayesian_surprise gradient — finite & well-defined" begin
    rng = Xoshiro(0xB17E)
    for _ in 1:5
        numStates = rand(rng, 2:5)
        numObs = rand(rng, 2:5)
        A = hcat([softmax(randn(rng, numObs)) for _ in 1:numStates]...)
        qs = softmax(randn(rng, numStates))

        f(qs) = bayesian_surprise(A, qs)
        g_fd = grad(FDM, f, qs)[1]
        @test all(isfinite, g_fd)
        @test length(g_fd) == numStates
    end
end
