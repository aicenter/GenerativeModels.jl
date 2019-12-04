export CMeanGaussian
export mean_var, variance

"""
    CMeanGaussian{T,AbstractVar}(mapping, σ2, xlength)

Conditional Gaussian that maps an input z to a mean μx. The variance σ2 is
shared for all datapoints.  The mapping must output dimensions appropriate for
the chosen variance type:

# Arguments
- `mapping`: maps condition z to μ=mapping(z) (e.g. a Flux Chain)
- `σ2`: shared variance for all datapoints
- `xlength`: length of mean/variance vectors. Only needed for `ScalarVar`
- `AbstractVar`: one of the variance types: DiagVar, ScalarVar
- `T`: expected eltype. E.g. `rand` will try to sample arrays of this eltype.
  If the mapping returns a different eltype the output of `mean`,`variance`,
  and `rand` is not necessarily of eltype T.

# Example
```julia-repl
julia> p = CMeanGaussian{Float32,DiagVar}(Dense(2,3),ones(Float32,3))
CMeanGaussian{Float32}(mapping=Dense(2, 3), σ2=3-element Array{Float32,1}

julia> mean_var(p,ones(2))
(Float32[1.8698889, -0.24418116, 0.76614076], Float32[1.0, 1.0, 1.0])

julia> rand(p, ones(2))
3-element Array{Float32,2}:
 0.1829532154926673
 0.1235498922955946
 0.0767166501426535
```
"""
struct CMeanGaussian{T,V<:AbstractVar} <: AbstractCGaussian{T}
    mapping
    σ::AbstractArray{T}
    xlength::Int
    _nograd::Dict{Symbol,Bool}
end

function CMeanGaussian{T,V}(m, σ::AbstractArray, xlength::Int) where {T,V}
    _nograd = Dict(:σ => σ isa NoGradArray)
    σ = _nograd[:σ] ? σ.data : σ
    CMeanGaussian{T,V}(m, σ, xlength, _nograd)
end

CMeanGaussian{T,DiagVar}(m, σ) where T = CMeanGaussian{T,DiagVar}(m, σ, size(σ,1))

mean(p::CMeanGaussian, z::AbstractArray) = p.mapping(z)
# TODO: use softplus_safe
variance(p::CMeanGaussian{T,DiagVar}) where T = p.σ .* p.σ
variance(p::CMeanGaussian{T,ScalarVar}) where T =
    p.σ .* p.σ .* fill!(similar(p.σ, p.xlength), 1)
variance(p::CMeanGaussian, z::AbstractArray) = variance(p)
mean_var(p::CMeanGaussian, z::AbstractArray) = (mean(p, z), variance(p))

# make sure that parameteric constructor is called...
function Flux.functor(p::CMeanGaussian{T,V}) where {T,V}
    fs = fieldnames(typeof(p))
    nt = (; (name=>getfield(p, name) for name in fs)...)
    nt, y -> CMeanGaussian{T,V}(y...)
end

function Flux.trainable(p::CMeanGaussian)
    ps = [getfield(p,k) for k in keys(p._nograd) if !p._nograd[k]]
    (p.mapping, ps...)
end

function Base.show(io::IO, p::CMeanGaussian{T}) where T
    e = repr(p.mapping)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    m = "CMeanGaussian{$T}(mapping=$e, σ2=$(summary(variance(p)))"
    print(io, m)
end
