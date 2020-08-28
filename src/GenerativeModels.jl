module GenerativeModels

    using Reexport
    using Random
    using Flux
    # using ForwardDiff

    using Distributions
    @reexport using ConditionalDists
    #@reexport using IPMeasures


    using ConditionalDists: AbstractConditionalDistribution
    const ACD = AbstractConditionalDistribution

    #using Flux: @adjoint
    #using DiffEqBase: ODEProblem, solve
    #using OrdinaryDiffEq: Tsit5

    abstract type AbstractGM end
    abstract type AbstractVAE <: AbstractGM end
    abstract type AbstractGAN <: AbstractGM end

    #include(joinpath("utils", "flux_ode_decoder.jl"))
    #include(joinpath("utils", "saveload.jl"))
    include(joinpath("utils", "utils.jl"))

    include(joinpath("models", "vae.jl"))
    #include(joinpath("models", "rodent.jl"))
    #include(joinpath("models", "gan.jl"))
    #include(joinpath("models", "vamp.jl"))

end # module
