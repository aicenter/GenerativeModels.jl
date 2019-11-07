export NoGradArray

"""
    NoGradArray(A::Array)

Wraps an array. Only used to filter out arrays that should stay constant
during training.
"""
struct NoGradArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

Base.size(A::NoGradArray) = size(A.data)
Base.IndexStyle(::Type{<:NoGradArray}) = IndexLinear()
Base.getindex(A::NoGradArray, idx) = getindex(A.data, idx)
