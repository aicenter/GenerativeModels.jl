export CMeanVarGaussian
export mean_var

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
julia> p = CGaussian(3, 2, Dense(2, 3))
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
struct CMeanVarGaussian{T,V<:AbstractVar} <: AbstractCGaussian{T}
    mapping
end

function mean_var(p::CMeanVarGaussian{T,DiagVar}, z::AbstractArray) where T
    ex = p.mapping(z)
    @assert eltype(ex) == T

    xlen = Int(size(ex, 1) / 2)
    μ = ex[1:xlen,:]
    σ = ex[xlen+1:end,:]

    return μ, σ .* σ
end

function mean_var(p::CMeanVarGaussian{T,ScalarVar}, z::AbstractArray) where T
    ex = p.mapping(z)
    @assert eltype(ex) == T

    μ = ex[1:end-1,:]
    σ = ex[end:end,:] .* fill!(similar(μ, 1, size(μ,2)), 1)

    return μ, σ .* σ
end

# make sure that parameteric constructor is called...
function Flux.functor(p::CMeanVarGaussian{T,V}) where {T,V}
    fs = fieldnames(typeof(p))
    nt = (; (name=>getfield(p, name) for name in fs)...)
    nt, y -> CMeanVarGaussian{T,V}(y...)
end

function Base.show(io::IO, p::CMeanVarGaussian{T,V}) where T where V
    e = repr(p.mapping)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    msg = "CMeanVarGaussian{$T,$V}(mapping=$e)"
    print(io, msg)
end
