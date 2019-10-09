module GenerativeModels

    using Reexport
    using Requires
    using Statistics
    using Random

    using BSON
    using DrWatson
    using ValueHistories
    using Flux
    @reexport using LinearAlgebra

    abstract type AbstractGM end
    abstract type AbstractVAE{T<:Real} <: AbstractGM end
    abstract type AbstractGAN{T<:Real} <: AbstractGM end

    import Base.length
    import Random.rand
    import Statistics.mean

    include(joinpath("utils", "saveload.jl"))
    include(joinpath("utils", "utils.jl"))

    # optional dependencies
    function __init__()
        @require DiffEqFlux="aae7a2af-3d4f-5e19-a356-7da93b79d9d0" using DiffEqBase
        @require DiffEqFlux="aae7a2af-3d4f-5e19-a356-7da93b79d9d0" using OrdinaryDiffEq
        @require DiffEqFlux="aae7a2af-3d4f-5e19-a356-7da93b79d9d0" include(joinpath("models", "rodent.jl"))
    end

    include(joinpath("pdfs", "abstract_pdfs.jl"))
    include(joinpath("pdfs", "gaussian.jl"))
    include(joinpath("pdfs", "abstract_cgaussian.jl"))
    include(joinpath("pdfs", "cgaussian.jl"))
    include(joinpath("pdfs", "svar_cgaussian.jl"))

    include(joinpath("models", "vae.jl"))
    include(joinpath("models", "gan.jl"))

end # module
