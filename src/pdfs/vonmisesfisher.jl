export VonMisesFisher

"""
VonMisesFisher{T}

Von Mises-Fisher distribution defined with mean μ and concentration κ that can be any `AbstractArray` and `Real` number respectively

# Arguments
- `μ::AbstractArray`: mean of Gaussian
- `σ2::AbstractArray`: variance of Gaussian

# Example
```julia-repl
julia> using Flux

julia> p = Gaussian(zeros(3), ones(3))
Gaussian{Float64}(μ=3-element Array{Float64,1}, σ2=3-element Array{Float64,1})

julia> mean_var(p)
([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))

julia> rand(p)
Tracked 3×1 Array{Float64,2}:
 -1.8102550562952886
  0.6218903591706907
 -0.8067583329396676
```
"""
struct VonMisesFisher{T} <: AbstractPDF{T}
    μ::AbstractArray{T}
    κ::AbstractArray{T}
    _nograd::Dict{Symbol,Bool}
end

VonMisesFisher(μ::AbstractArray{T}, κ::T) where {T} = VonMisesFisher(μ, NoGradArray[κ])
function VonMisesFisher(μ::AbstractArray{T}, κ::AbstractArray{T}) where {T}
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

function loglikelihood(p::VonMisesFisher{T}, x::AbstractArray{T}) where T
    (μ, κ) = mean_conc(p)
    log_vmf(μ, κ)
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
    .- vmfentropy(q.dims, concentration(p)) .+ huentropy(q.dims)
end

function Base.show(io::IO, p::VonMisesFisher{T}) where T
    msg = "VonMisesFisher{$T}(μ=$(summary(mean(p))), κ=$(concentration(p)))"
    print(io, msg)
end
