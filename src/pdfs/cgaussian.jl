export CGaussian
export AbstractVar, DiagVar, ScalarVar, UnitVar
export mean, variance, mean_var, rand, loglikelihood, kld, xlength, zlength

"""Abstract variance type"""
abstract type AbstractVar end

"""Diagonal variance represented as a vector"""
struct DiagVar <: AbstractVar end

"""Scalar variance represented as a one-element vector"""
struct ScalarVar <: AbstractVar end

"""Unit variance represented by a vector of ones"""
struct UnitVar <: AbstractVar end

"""
    CGaussian{T,AbstractVar}

Conditional Gaussian that maps an input of `zlength` to its mean of `xlength`.
The mapping must output dimensions appropriate for the chosen variance type

# Arguments
- `xlength::Int`: length of mean
- `zlength::Int`: length of condition
- `mapping`: maps condition z to μ=mapping(z) (e.g. a Flux Chain)
- `AbstractVar`: one of the variance types: DiagVar, ScalarVar, UnitVar

# Example
```julia-repl
julia> p = CGaussian{Float64,UnitVar}(3, 2, Dense(2, 3))
CGaussian{Float64,UnitVar}(xlength=3, zlength=2, mapping=Dense(2, 3))

julia> mean_var(p, ones(2))
([-0.339991, -0.061213, -0.769473] (tracked), [1.0, 1.0, 1.0])

julia> rand(p, ones(2))
Tracked 3×1 Array{Float64,2}:
 0.1829532154926673
 0.1235498922955946
 0.0767166501426535
```
"""
struct CGaussian{T,V<:AbstractVar} <: AbstractCPDF{T}
    xlength::Int
    zlength::Int
    mapping

    function CGaussian{T,V}(xlength, zlength, mapping) where T where V

        if T == Float32
            mapping = f32(mapping)
        elseif T == Float64
            mapping = f64(mapping)
        else
            error("Encoder cannot be converted to type $T")
        end

        cg = new(xlength, zlength, mapping)
        ex = mapping(randn(T, zlength))

        if V == UnitVar
            size(ex) == (xlength,) ? cg : error("With UnitVar mapping must return samples of xlength")
        elseif V == ScalarVar
            size(ex) == (xlength+1,) ? cg : error("With ScalarVar mapping must return samples of xlength+1")
        else
            size(ex) == (xlength*2,) ? cg : error("With DiagVar mapping must return samples of xlength*2")
        end
    end
end

Flux.@treelike CGaussian

xlength(p::CGaussian) = p.xlength
zlength(p::CGaussian) = p.zlength

"""
    mean_var(p::CGaussian, z::AbstractArray)

Returns mean and variance of a conditional Gaussian
"""
function mean_var(p::CGaussian{T}, z::AbstractArray) where T
    ex = p.mapping(z)
    return ex[1:p.xlength,:], softplus_safe.(ex[p.xlength+1:end,:])
end

"""
    mean_var(p::CGaussian{T,UnitVar}, z::AbstractArray)

Returns mean and (unit) variance of a conditional Gaussian with unit variance. 
"""
function mean_var(p::CGaussian{T,UnitVar}, z::AbstractArray) where T
    μ = p.mapping(z)
    return μ, ones(T, xlength(p))
end

"""
    mean(p::CGaussian, z::AbstractArray)

Returns mean of a conditional Gaussian.
"""
mean(p::CGaussian, z::AbstractArray) = mean_var(p, z)[1]

"""
    variance(p::CGaussian, z::AbstractArray)

Returns variance of a conditional Gaussian.
"""
variance(p::CGaussian, z::AbstractArray) = mean_var(p, z)[2]


"""
    rand(p::CGaussian, z::AbstractArray; batch=1)

Produce `batch` number of samples from a conditional Gaussian.
"""
function rand(p::CGaussian{T}, z::AbstractArray; batch=1) where T
    (μ, σ2) = mean_var(p, z)
    k = xlength(p)
    μ .+ sqrt.(σ2) .* randn(T, k, batch)
end

"""
    loglikelihood(p::CGaussian, x::AbstractArray, z::AbstractArray)

Computes the log p(x|z).
"""
function loglikelihood(p::CGaussian, x::AbstractArray, z::AbstractArray)
    (μ, σ2) = mean_var(p, z)
    k = xlength(p)
    - (sum((x - μ).^2 ./ σ2, dims=1) .+ sum(log.(σ2)) .+ k*log(2π)) ./ 2
end

"""
    kld(p::CGaussian, q::Gaussian, z::AbstractArray)

Compute Kullback-Leibler divergence between a conditional Gausian `p` given `z`
and a Gaussian `q`.
"""
function kld(p::CGaussian{T}, q::Gaussian{T}, z::AbstractArray) where T
    (μ1, σ1) = mean_var(p, z)
    (μ2, σ2) = mean_var(q)
    k = xlength(p)
    (-k + sum(log.(σ2 ./ σ1)) + sum(σ1 ./ σ2) .+ sum((μ2 .- μ1).^2 ./ σ1, dims=1)) ./ 2
end

function Base.show(io::IO, p::CGaussian{T,V}) where T where V
    e = repr(p.mapping)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    msg = "CGaussian{$T,$V}(xlength=$(p.xlength), zlength=$(p.zlength), mapping=$e)"
    print(io, msg)
end
