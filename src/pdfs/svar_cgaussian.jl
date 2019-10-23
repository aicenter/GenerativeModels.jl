export CMeanGaussian
export mean_var, variance

"""
    CMeanGaussian{T}

Conditional Gaussian that maps an input of `zlength` to its mean of `xlength`.
The mapping must output dimensions appropriate for the chosen variance type
The variance is the same for all data-points, but can still be represented by
an optimized `TrackedArray`.

# Arguments
- `xlength::Int`: length of mean
- `zlength::Int`: length of condition
- `mapping`: maps condition z to μ=mapping(z) (e.g. a Flux Chain)
- `σ2`: shared variance for all data-points

# Example
```julia-repl
julia> p = CMeanGaussian(Dense(2, 3), param(ones(Float32, 3)))
CMeanGaussian{Float32}(mapping=Dense(2, 3), σ2=...)

julia> rand(p, ones(2))
Tracked 3×1 Array{Float32,2}:
 0.1829532154926673
 0.1235498922955946
 0.0767166501426535
```
"""
struct CMeanGaussian{T,V<:AbstractVar} <: AbstractCGaussian{T}
    mapping
    σ::AbstractArray{T}
    xlength::Int
end

CMeanGaussian{T,DiagVar}(m, σ) where T = CMeanGaussian{T,DiagVar}(m, σ, size(σ,1))

mean(p::CMeanGaussian, z::AbstractArray) = p.mapping(z)
# TODO: dispatch on variance type
# TODO: use softplus_safe
variance(p::CMeanGaussian{T,DiagVar}) where T = p.σ .* p.σ
variance(p::CMeanGaussian{T,ScalarVar}) where T = p.σ .* p.σ .* fill!(similar(p.σ, p.xlength), 1)
variance(p::CMeanGaussian, z::AbstractArray) = variance(p)
mean_var(p::CMeanGaussian, z::AbstractArray) = (mean(p, z), variance(p))

# make sure that parameteric constructor is called...
function Flux.functor(p::CMeanGaussian{T,V}) where {T,V}
    fs = fieldnames(typeof(p))
    nt = (; (name=>getfield(p, name) for name in fs)...)
    nt, y -> CMeanGaussian{T,V}(y...)
end

function Base.show(io::IO, p::CMeanGaussian{T}) where T
    e = repr(p.mapping)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    m = "CMeanGaussian{$T}(mapping=$e, σ2=$(summary(variance(p)))"
    print(io, m)
end
