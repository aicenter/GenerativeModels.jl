module GenerativeModels

    using Reexport
    using Random
    using Flux

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

    export VAE, GAN
    export elbo, mmd_mean, mmd_rand, generator_loss, discriminator_loss
    export train!, softplus_safe, save_checkpoint, load_checkpoint


    abstract type AbstractGM end
    abstract type AbstractVAE <: AbstractGM end
    abstract type AbstractGAN <: AbstractGM end

    # include(joinpath("utils", "saveload.jl"))
    include(joinpath("utils", "utils.jl"))

    # to make Flux.gpu work on VAE/GAN/etc priors we need:
    Flux.@functor DistributionsAD.TuringScalMvNormal
    Flux.@functor DistributionsAD.TuringDiagMvNormal
    Flux.@functor DistributionsAD.TuringDenseMvNormal

    include(joinpath("models", "vae.jl"))
    include(joinpath("models", "gan.jl"))
    #include(joinpath("models", "vamp.jl"))

    # TODO: wrap in requires
    #using ForwardDiff
    #using Flux: @adjoint
    #using DiffEqBase: ODEProblem, solve
    #using OrdinaryDiffEq: Tsit5
    #include(joinpath("models", "rodent.jl"))
    #include(joinpath("utils", "flux_ode_decoder.jl"))

end # module
