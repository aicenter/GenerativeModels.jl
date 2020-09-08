"""
	VAMP{K<:Int,P<:AbstractArray}(X::P)
	VAMP{K<:Int,P<:AbstractArray}(K::Int, xdim::Union{Tuple, Int})
	
Vamp prior with `K` pseudoinputs of type `P`. If only the pseudoinput `X` is given, `K` is the size of 
the last dimension of the input array. 

# Arguments
* `K`: Number of pseudoinputs
* `xdim`: Size of an individual pseudoinput
* `X`: Pseudoinput array

# Example
Initialize a VAMP with K pseudoinputs of size xdim:
```julia-repl
julia> v = VAMP(2, 2)
VAMP{Int64,Array{Float32,2}}(
K: 2
pseudoinputs: Float32[0.876433 0.56233144; 0.24732344 0.44447201]
)

julia> v = VAMP(2, (3, 2))
VAMP{Int64,Array{Float32,3}}(
K: 2
pseudoinputs: Float32[0.21458945 0.8710681; 0.2402707 -0.20706704; 0.9916858 0.6571086]

Float32[0.6324719 0.27955383; -0.39046887 0.18451884; -0.18170312 -1.3017029]
)

julia> v = VAMP(zeros(3,2))
VAMP{Int64,Array{Float64,2}}(
K: 2
pseudoinputs: [0.0 0.0; 0.0 0.0; 0.0 0.0]
)
```
"""
struct VAMP{K<:Int,P<:AbstractArray} <: ContinuousMultivariateDistribution # add type here
	K::K
	pseudoinputs::P
end
VAMP(K::Int, xdim::Union{Tuple, Int}) = VAMP(K, randn(Float32, xdim..., K))
VAMP(X::AbstractArray) = VAMP(size(X)[end], X)

Flux.@functor VAMP

function Flux.trainable(m::VAE{<:VAMP})
    (prior=m.prior, encoder=m.encoder, decoder=m.decoder)
end

_nograd_rand(a,b) = rand(a,b)
_nograd_repeat(a,b) = repeat(a,b)
Flux.Zygote.@nograd _nograd_rand
Flux.Zygote.@nograd _nograd_repeat 

"""
	init_vamp_mean(K::Int, X::AbstractArray[, σ=1f0])

Initialize a VAMP whose pseudoinputs are `K` slightly perturbed means of `X`. Currently only works
for X on cpu.
"""
init_vamp_mean(K::Int, X::AbstractArray, σ=1f0) = VAMP(K, σ*randn(eltype(X), size(X)[1:end-1]..., K) .+ mean(X, dims=ndims(X)))

"""
	init_vamp_sample(K::Int, X::AbstractArray)

Initialize a VAMP whose pseudoinputs are `K` random samples of `X`.
"""
init_vamp_sample(K::Int, X::AbstractArray) = VAMP(K,  X[repeat([:], ndims(X)-1)..., rand(1:size(X)[end], K)])

"""
	rand(p::VAMP[, batchsize, component_id])

Sample batchsize pseudoinputs of VAMP.
"""
function Random.rand(p::VAMP, batchsize::Int=1)
	ids = _nograd_rand(1:p.K, batchsize)
	_sampleVamp(p, ids)
end
function Random.rand(p::VAMP, batchsize::Int, k::Int)
	k > p.K ? error("requested samples from pseudoinput $k, only $(p.K) available") : 
		nothing
	ids = _nograd_repeat([k], batchsize)
	_sampleVamp(p, ids)
end
_sampleVamp(p::VAMP, ids) = 
	p.pseudoinputs[_nograd_repeat([:], ndims(p.pseudoinputs)-1)..., ids]

"""
    mmd_mean_vamp(m::AbstractVAE{<:VAMP}, x::AbstractArray, k[; distance])

MMD distance between a VAE VAMP prior and the encoding of X. This version uses mean 
of encoder.
"""
function mmd_mean(m::VAE{<:VAMP}, x::AbstractArray, k::AbstractKernel;
                  distance = IPMeasures.pairwisel2)
    mmd(k, mean(m.encoder, x), mean(m.encoder, rand(m.prior, size(x, 2))), distance)
end

mmd_mean_vamp(m,x,k,distance=IPMeasures.pairwisel2) = mmd_mean(m,x,k,distance)
