module GenerativeModels

    using Reexport
    using Random
    using BSON
    using DrWatson
    using ValueHistories
    using Flux
    using ForwardDiff
    @reexport using ConditionalDists
    @reexport using IPMeasures


    using Flux: @adjoint
    using DiffEqBase: ODEProblem, solve
    using OrdinaryDiffEq: Tsit5
    using ConditionalDists: AbstractPDF, AbstractCPDF

    abstract type AbstractGM end
    abstract type AbstractVAE <: AbstractGM end
    abstract type AbstractGAN <: AbstractGM end

    # functions that are overloaded by this module
    import Base.length
    import Random.rand
    import Statistics.mean

    include(joinpath("utils", "flux_ode_decoder.jl"))
    include(joinpath("utils", "saveload.jl"))
    include(joinpath("utils", "utils.jl"))

    include(joinpath("models", "vae.jl"))
    include(joinpath("models", "rodent.jl"))
    include(joinpath("models", "gan.jl"))
    include(joinpath("models", "vamp.jl"))

end # module
