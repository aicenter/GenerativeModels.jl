export Gaussian
export mean_var, rand, loglikelihood, kld, length

"""
    Gaussian{T}

Gaussian defined with mean μ and variance σ2 that can be any `AbstractArray`

# Arguments
- `μ::AbstractArray`: mean of Gaussian
- `σ2::AbstractArray`: variance of Gaussian

# Example
```julia-repl
julia> using Flux

julia> p = Gaussian(param(zeros(3)), param(ones(3)))
Gaussian{Float64}(μ=Tracked 3-element Array{Float64,1}, σ2=Tracked 3-element Array{Float64,1})

julia> mean_var(p)
([0.0, 0.0, 0.0] (tracked), [1.0, 1.0, 1.0] (tracked))

julia> rand(p)
Tracked 3×1 Array{Float64,2}:
 -1.8102550562952886
  0.6218903591706907
 -0.8067583329396676
```
"""
struct Gaussian{T} <: AbstractPDF{T}
    μ::AbstractArray{T}
    σ2::AbstractArray{T}
end

Flux.@functor Gaussian
length(p::Gaussian) = size(p.μ, 1)
#mean_var(p::Gaussian{T}) where T = (p.μ, softplus_safe.(p.σ2, T))
mean_var(p::Gaussian) = (p.μ, p.σ2 .* p.σ2)

function rand(p::Gaussian, batchsize::Int=1)
    (μ, σ2) = mean_var(p)
    k = length(p)
    r = randn!(similar(μ, k, batchsize))
    μ .+ sqrt.(σ2) .* r
end

function loglikelihood(p::Gaussian{T}, x::AbstractArray{T}) where T
    - (sum((x .- p.μ).^2 ./ p.σ2, dims=1) .+ sum(log.(p.σ2) .+ T(log(2π)))) ./ 2
end

function kld(p::Gaussian, q::Gaussian)
    (μ1, σ1) = mean_var(p)
    (μ2, σ2) = mean_var(q)
    k = length(p)
    (-k + sum(log.(σ2 ./ σ1)) + sum(σ1 ./ σ2) .+ sum((μ2 - μ1).^2 ./ σ1, dims=1)) ./ 2
end

function Base.show(io::IO, p::Gaussian{T}) where T
    msg = "Gaussian{$T}(μ=$(summary(p.μ)), σ2=$(summary(p.σ2)))"
    print(io, msg)
end
