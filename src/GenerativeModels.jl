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


    using Distributions: ContinuousMultivariateDistribution
    using ConditionalDists: AbstractConditionalDistribution
    const CMD = ContinuousMultivariateDistribution
    const ACD = AbstractConditionalDistribution

    using Flux: @adjoint
    using DiffEqBase: ODEProblem, solve
    using OrdinaryDiffEq: Tsit5

    abstract type AbstractGM end
    abstract type AbstractVAE <: AbstractGM end
    abstract type AbstractGAN <: AbstractGM end

    # include(joinpath("utils", "flux_ode_decoder.jl"))
    # include(joinpath("utils", "saveload.jl"))
    include(joinpath("utils", "utils.jl"))

    include(joinpath("models", "vae.jl"))
    # include(joinpath("models", "rodent.jl"))
    # include(joinpath("models", "gan.jl"))

end # module
