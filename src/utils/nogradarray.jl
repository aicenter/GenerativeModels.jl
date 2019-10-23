export NoGradArray

struct NoGradArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

Base.size(A::NoGradArray) = size(A.data)
Base.IndexStyle(::Type{<:NoGradArray}) = IndexLinear()
Base.getindex(A::NoGradArray, i::Int) = A.data[i]
