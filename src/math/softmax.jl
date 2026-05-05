# Numerically stable softmax and log-softmax operations.
#
# Active inference uses softmax constantly: action selection ∝ exp(γ·-G(π));
# state inference ∝ exp(log P(o|s) + log P(s)); preferences encoded as log
# probabilities. The numerical-stability technique of subtracting the max
# before exponentiating is delegated to LogExpFunctions.logsumexp.

using LogExpFunctions: logsumexp

"""
    softmax(x)

Numerically stable softmax of `x`:

    softmax(x)[i] = exp(x[i]) / Σⱼ exp(x[j])

Computed via the shift-invariant identity `softmax(x) = softmax(x - max(x))`
to avoid overflow. Returns a vector that sums to 1 in exact arithmetic; in
floating-point, the sum is exactly 1 by construction.

In active inference, `x` is typically a log-density: log-likelihood + log-prior
for state inference, or `γ·(-G(π))` for policy inference.

# Examples
```jldoctest
julia> softmax([0.0, 0.0, 0.0])
3-element Vector{Float64}:
 0.3333333333333333
 0.3333333333333333
 0.3333333333333333

julia> softmax([1000.0, 1001.0]) ≈ [1/(1+ℯ), ℯ/(1+ℯ)]
true
```
"""
function softmax(x::AbstractVector{T}) where {T<:Real}
    isempty(x) && throw(ArgumentError("softmax: input vector is empty"))
    R = float(T)
    m = maximum(x)
    out = similar(x, R)
    s = zero(R)
    @inbounds for i in eachindex(x)
        out[i] = exp(x[i] - m)
        s += out[i]
    end
    @inbounds for i in eachindex(out)
        out[i] /= s
    end
    return out
end

"""
    softmax(x, β)

Tempered (precision-weighted) softmax: `softmax(β·x)`. For `β = 1` this is
the standard softmax. As `β → ∞` the output approaches a one-hot vector at
`argmax(x)`; as `β → 0` it approaches uniform.

In active inference, `β` is conventionally written as `γ` (precision /
inverse temperature). This is the form used in policy inference:

    q(π) ∝ exp(γ · (-G(π))) = softmax(-G, γ)
"""
function softmax(x::AbstractVector{<:Real}, β::Real)
    return softmax(β .* x)
end

"""
    logsoftmax(x)

Numerically stable log-softmax: `logsoftmax(x)[i] = x[i] - logsumexp(x)`.

Preferred over `log.(softmax(x))` when downstream operations need log
probabilities — avoids underflow when entries of `softmax(x)` are tiny.
"""
function logsoftmax(x::AbstractVector{T}) where {T<:Real}
    isempty(x) && throw(ArgumentError("logsoftmax: input vector is empty"))
    lse = logsumexp(x)
    return x .- lse
end

"""
    logsoftmax(x, β)

Tempered log-softmax: `log.(softmax(β·x))`.
"""
function logsoftmax(x::AbstractVector{<:Real}, β::Real)
    return logsoftmax(β .* x)
end
