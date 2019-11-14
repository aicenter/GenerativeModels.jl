export VonMisesFisher

"""
VonMisesFisher{T}

Von Mises-Fisher distribution defined with mean μ and concentration κ that can be any `AbstractArray` and `Real` number respectively

# Arguments
- `μ::AbstractArray`: mean of VMF
- `κ::AbstractArray`: concentration of VMF

# Example
```julia-repl
julia> using Flux

julia> p = VonMisesFisher(zeros(3), 1.0)
VonMisesFisher{Float64}(μ=3-element Array{Float64,1}, κ=[1.0])

julia> mean_conc(p)
([0.0, 0.0, 0.0], [1.0])

julia> rand(p)
3×1 Array{Float64,2}:
 -0.534718473601494 
  0.4131946025140243
  0.7371203256202924
```
"""
struct VonMisesFisher{T} <: AbstractPDF{T}
    μ::AbstractArray{T}
    κ::AbstractArray{T}
    _nograd::Dict{Symbol,Bool}
end

#! Watch out, there is no check for μ actually being on a sphere, even though all the methods count with that!
VonMisesFisher(μ::AbstractMatrix{T}, κ::Union{T, AbstractArray{T}}) where {T} = VonMisesFisher(vec(μ), κ)
VonMisesFisher(μ::AbstractVector{T}, κ::T) where {T} = VonMisesFisher(μ, NoGradArray([κ]))
function VonMisesFisher(μ::AbstractVector{T}, κ::AbstractArray{T}) where {T}
    _nograd = Dict(
        :μ => μ isa NoGradArray,
        :κ => κ isa NoGradArray)
    μ = _nograd[:μ] ? μ.data : μ
    κ = _nograd[:κ] ? κ.data : κ
    VonMisesFisher(μ, κ, _nograd)
end

Flux.@functor VonMisesFisher

function Flux.trainable(p::VonMisesFisher)
    ps = (;(k=>getfield(p,k) for k in keys(p._nograd) if !p._nograd[k])...)
end

length(p::VonMisesFisher) = size(p.μ, 1)
mean_conc(p::VonMisesFisher) = (p.μ, p.κ)
mean(p::VonMisesFisher) = p.μ
concentration(p::VonMisesFisher) = p.κ

function rand(p::VonMisesFisher, batchsize::Int=1)
    (μ, κ) = mean_conc(p)
    μ = μ .* ones(size(μ, 1), batchsize)
    κ = κ .* ones(1, batchsize)
    sample_vmf(μ, κ)
end

loglikelihood(p::VonMisesFisher{T}, x::AbstractVector{T}) where T = loglikelihood(p, x * ones(1, 1))
function loglikelihood(p::VonMisesFisher{T}, x::AbstractMatrix{T}) where T
    (μ, κ) = mean_conc(p)
    μ = μ * ones(1, size(x, 2))
    log_vmf(x, μ, κ[1])
end

"""
kld(p::AbstractCVMF, q::HypersphericalUniform, z::AbstractArray)

Compute Kullback-Leibler divergence between a conditional Von Mises-Fisher distribution `p` given `z`
and a hyperspherical uniform distribution `q` with the same dimensionality.
"""
function kld(p::VonMisesFisher{T}, q::HypersphericalUniform{T}) where {T}
    if length(p.μ) != q.dims
        error("Cannot compute KLD between VMF and HSU with different dimensionality")
    end
    .- vmfentropy(q.dims, concentration(p)[1]) .+ huentropy(q.dims)
end

function Base.show(io::IO, p::VonMisesFisher{T}) where T
    msg = "VonMisesFisher{$T}(μ=$(summary(mean(p))), κ=$(concentration(p)))"
    print(io, msg)
end
