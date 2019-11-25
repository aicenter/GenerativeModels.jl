export CMeanConcVMF

"""
CMeanConcVMF(mapping, xlength::Int)

Conditional Von Mises-Fisher that maps an input z to a mean μx and a concentration κ.
The mapping should end by the last hidden layer because the constructor will add
transformations for μ and κ. 
    ```julia-repl
    μ_from_hidden = Chain(Dense(hidden_dim, xlength), x -> normalizecolumns(x))
    ```

    ```julia-repl
    κ_from_hidden = Dense(hidden_dim, 1, x -> σ.(x) .* 100)
    ```
# Arguments
- `mapping`: maps condition z to mean and concentration (e.g. a Flux Chain)
- `T`: expected eltype. E.g. `rand` will try to sample arrays of this eltype.
  If the mapping returns a different eltype the output of `mean`,`concentration`,
  and `rand` is not necessarily of eltype T.

# Example
```julia-repl
julia> p = CMeanConcVMF{Float32}(Dense(2, 3), 3)
CMeanConcVMF{Float32}(mapping=Dense(2, 3), μ_from_hidden=Chain(Dense(3, 3), #45), κ_from_hidden=Dense(3, 1, #46))

julia> mean_conc(p, ones(2, 1))
(Float32[-0.1507113; -0.9488135; 0.27755922], Float32[85.390144])

julia> rand(p, ones(2,1))
3×1 Array{Float32,2}:
  0.22024345
 -0.9597406 
 -0.1743287
```
"""
struct CMeanConcVMF{T} <: AbstractCVMF{T}
    mapping
    μ_from_hidden
    κ_from_hidden
end

#! Watch out, kappa is capped between 0 and 100 because it was exploding before. You might want to change this to softmax for kappa but in practice it did not behave well
CMeanConcVMF{T}(mapping, hidden_dim::Int, xlength::Int) where {T} = CMeanConcVMF{T}(mapping, Chain(Dense(hidden_dim, xlength), x -> normalizecolumns(x)), Dense(hidden_dim, 1, x -> σ.(x) .* 100))
CMeanConcVMF{T}(mapping::Chain{C}, xlength::Int) where {C, T} = CMeanConcVMF{T}(mapping, size(mapping[length(mapping)].W, 1), xlength)
CMeanConcVMF{T}(mapping::Dense{D}, xlength::Int) where {D, T} = CMeanConcVMF{T}(mapping, size(mapping.W, 1), xlength)

function mean_conc(p::CMeanConcVMF{T}, z::AbstractArray) where {T}
    ex = p.mapping(z)
    if eltype(ex) != T
        error("Mapping should return eltype $T. Found: $(eltype(ex))")
    end

    return p.μ_from_hidden(ex), p.κ_from_hidden(ex)
end

# make sure that parameteric constructor is called...
function Flux.functor(p::CMeanConcVMF{T}) where {T}
    fs = fieldnames(typeof(p))
    nt = (; (name=>getfield(p, name) for name in fs)...)
    nt, y -> CMeanConcVMF{T}(y...)
end

function Base.show(io::IO, p::CMeanConcVMF{T}) where {T}
    msg = "CMeanConcVMF{$T}(mapping=$(short_repr(p.mapping)), μ_from_hidden=$(short_repr(p.μ_from_hidden)), κ_from_hidden=$(short_repr(p.κ_from_hidden)))"
    print(io, msg)
end