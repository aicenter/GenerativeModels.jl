module GenerativeModels

    using Reexport
    using Random
    using Flux
    # using ForwardDiff

    using Distributions
    using DistributionsAD
    using Distances
    @reexport using ConditionalDists
    @reexport using IPMeasures


    using IPMeasures: AbstractKernel
    using ConditionalDists: AbstractConditionalDistribution
    const ACD = AbstractConditionalDistribution
    const TuMvNormal = Union{DistributionsAD.TuringDenseMvNormal,
                             DistributionsAD.TuringDiagMvNormal,
                             DistributionsAD.TuringScalMvNormal}

    export VAE
    export elbo, mmd_mean, mmd_rand
    export train!, softplus_safe


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
