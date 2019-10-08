export SharedVarCGaussian
export mean_var, variance

"""
    SharedVarCGaussian{T}

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
julia> p = SharedVarCGaussian(3, 2, Dense(2, 3), param(ones(Float32, 3)))
SharedVarCGaussian{Float32}(xlength=3, zlength=2, mapping=Dense(2, 3), σ2=...)

julia> rand(p, ones(2))
Tracked 3×1 Array{Float32,2}:
 0.1829532154926673
 0.1235498922955946
 0.0767166501426535
```
"""
struct SharedVarCGaussian{T} <: AbstractCGaussian{T}
    xlength::Int
    zlength::Int
    mapping
    σ2::AbstractArray{T}
end

function SharedVarCGaussian(xlength::Int, zlength::Int, mapping::Function, σ2::AbstractArray{T}) where T
    V = detect_mapping_variant(mapping, T, xlength, zlength)
    SharedVarCGaussian{T}(xlength, zlength, mapping, σ2)
end

function SharedVarCGaussian(xlength::Int, zlength::Int, mapping, σ2::AbstractArray{T}) where T
    @assert eltype(first(params(mapping)).data) == T
    V = detect_mapping_variant(mapping, xlength, zlength)
    SharedVarCGaussian{T}(xlength, zlength, mapping, σ2)
end

Flux.@treelike SharedVarCGaussian

variance(p::SharedVarCGaussian) = p.σ2.^2
mean_var(p::SharedVarCGaussian, z::AbstractArray) = (p.mapping(z), variance(p))
#function mean_var(p::SharedVarCGaussian{T}, z::AbstractArray{T}) where T
#    (p.mapping(z), softplus_safe.(p.σ2, T))
#end
#variance(p::SharedVarCGaussian{T}) where T = softplus_safe.(p.σ2, T)

function Base.show(io::IO, p::SharedVarCGaussian{T}) where T
    e = repr(p.mapping)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    xl = p.xlength
    zl = p.zlength
    m = "SharedVarCGaussian{$T}(xlength=$xl, zlength=$zl, mapping=$e, σ2=$(summary(p.σ2))"
    print(io, m)
end
