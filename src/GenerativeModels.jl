module GenerativeModels

    using Reexport
    using Random
    using Flux

    using Distributions
    using DistributionsAD
    @reexport using ConditionalDists
    @reexport using IPMeasures

    using IPMeasures: AbstractKernel
    using ConditionalDists: AbstractConditionalDistribution
    const ACD = AbstractConditionalDistribution
    const TuMvNormal = Union{DistributionsAD.TuringDenseMvNormal,
                             DistributionsAD.TuringDiagMvNormal,
                             DistributionsAD.TuringScalMvNormal}

    export VAE, GAN, VAMP
    export elbo, mmd_mean, mmd_rand, generator_loss, discriminator_loss
    export train!, softplus_safe, save_checkpoint, load_checkpoint
    export mmd_mean_vamp, init_vamp_mean, init_vamp_sample

    abstract type AbstractGM end
    abstract type AbstractVAE <: AbstractGM end
    abstract type AbstractGAN <: AbstractGM end


    # to make Flux.gpu work on VAE/GAN/etc priors we need:
    Flux.@functor DistributionsAD.TuringScalMvNormal
    Flux.@functor DistributionsAD.TuringDiagMvNormal
    Flux.@functor DistributionsAD.TuringDenseMvNormal

    include("utils.jl")
    include("vae.jl")
    include("gan.jl")
    include("vamp.jl")

end # module
