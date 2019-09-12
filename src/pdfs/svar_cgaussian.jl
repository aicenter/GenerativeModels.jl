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
julia> p = SharedVarCGaussian{Float64}(3, 2, Dense(2, 3), param(ones(3)))
SharedVarCGaussian{Float64}(xlength=3, zlength=2, mapping=Dense(2, 3), σ2=...)

julia> rand(p, ones(2))
Tracked 3×1 Array{Float64,2}:
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

    function SharedVarCGaussian{T}(xlength, zlength, mapping, σ2) where T

        if T == Float32
            mapping = f32(mapping)
        elseif T == Float64
            mapping = f64(mapping)
        else
            error("Encoder cannot be converted to type $T")
        end

        cg = new(xlength, zlength, mapping, σ2)
        ex = mapping(randn(T, zlength))
        size(ex) == (xlength,) ? cg : error("Mapping must return samples of xlength")
    end

end

Flux.@treelike SharedVarCGaussian

mean_var(p::SharedVarCGaussian, z::AbstractArray) = (p.mapping(z), p.σ2)
variance(p::SharedVarCGaussian) = p.σ2

function Base.show(io::IO, p::SharedVarCGaussian{T}) where T
    e = repr(p.mapping)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    xl = p.xlength
    zl = p.zlength
    m = "SharedVarCGaussian{$T}(xlength=$xl, zlength=$zl, mapping=$e, σ2=$(summary(p.σ2))"
    print(io, m)
end
