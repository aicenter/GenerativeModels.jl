import ConditionalDists: AbstractCGaussian
export VAMP, mmd_mean_vamp
"""
	VAMP{K,P}

Vamp prior with K pseudoinputs P.
"""
struct VAMP{K<:Int,P<:AbstractArray} <: AbstractPDF # add type here
	K::K
	pseudoinputs::P

#	VAMP(K::Int, xdim::Union{Tuple, Int}) = new(K, Flux.param(randn(Float32, xdim..., K)))
#	VAMP(K::Int, X::AbstractArray) = new(K, Flux.Tracker.istracked(X) ? X : Flux.param(X))
end
# TODO - better pseudoinputs initializers
# TODO - check gpu compatibility

Flux.@functor VAMP

_nograd_rand(a,b) = rand(a,b)
_nograd_repeat(a,b) = repeat(a,b)
Flux.Zygote.@nograd _nograd_rand
Flux.Zygote.@nograd _nograd_repeat 

"""
	rand(p::VAMP[, batchsize, component_id])

Sample batchsize pseudoinputs of VAMP.
"""
function rand(p::VAMP, batchsize::Int=1)
	ids = _nograd_rand(1:p.K, batchsize)
	_sampleVamp(p, ids)
end
function rand(p::VAMP, batchsize::Int, k::Int)
	k > p.K ? error("requested samples from pseudoinput $k, only $(p.K) available") : 
		nothing
	ids = _nograd_repeat([k], batchsize)
	_sampleVamp(p, ids)
end
_sampleVamp(p::VAMP, ids) = 
	p.pseudoinputs[_nograd_repeat([:], ndims(p.pseudoinputs)-1)..., ids]

"""
	mmd_mean_vamp(m::AbstractVAE, x::AbstractArray, k[; distance])

MMD distance between a VAE VAMP prior and the encoding of X. This version uses mean 
of encoder.
"""
mmd_mean_vamp(m::AbstractVAE, x::AbstractArray, k; 
		distance = IPMeasures.pairwisel2) = 
    	mmd(k, mean(m.encoder, x), mean(m.encoder, rand(m.prior, size(x, 2))), distance)
	