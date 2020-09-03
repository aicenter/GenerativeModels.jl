module GenerativeModels

    using Reexport
    using Random
    using Flux
    # using ForwardDiff

    using Distributions
    using DistributionsAD
    using Distances
    using KernelFunctions
    @reexport using ConditionalDists
    #@reexport using IPMeasures


    using ConditionalDists: AbstractConditionalDistribution
    const ACD = AbstractConditionalDistribution
    const TuMvNormal = Union{DistributionsAD.TuringDenseMvNormal,
                             DistributionsAD.TuringDiagMvNormal,
                             DistributionsAD.TuringScalMvNormal}

    #using Flux: @adjoint
    #using DiffEqBase: ODEProblem, solve
    #using OrdinaryDiffEq: Tsit5

    abstract type AbstractGM end
    abstract type AbstractVAE <: AbstractGM end
    abstract type AbstractGAN <: AbstractGM end

    #include(joinpath("utils", "flux_ode_decoder.jl"))
    #include(joinpath("utils", "saveload.jl"))
    include(joinpath("utils", "kld.jl"))
    include(joinpath("utils", "mmd.jl"))
    include(joinpath("utils", "utils.jl"))

    include(joinpath("models", "vae.jl"))
    #include(joinpath("models", "rodent.jl"))
    #include(joinpath("models", "gan.jl"))
    #include(joinpath("models", "vamp.jl"))

end # module
