module GenerativeModels

    using Random
    using BSON, DrWatson, ValueHistories
    using Flux, ForwardDiff
    using Zygote: @nograd, @adjoint
    using DiffEqBase: ODEProblem, solve
    using OrdinaryDiffEq: Tsit5
    using SpecialFunctions
    using Distributions
    using Adapt

    abstract type AbstractGM end
    abstract type AbstractVAE{T<:Real} <: AbstractGM end
    abstract type AbstractGAN{T<:Real} <: AbstractGM end
    abstract type AbstractSVAE{T<:Real} <: AbstractGM end

    # functions that are overloaded by this module
    import Base.length
    import Random.rand
    import Statistics.mean
    import SpecialFunctions: besselix, logabsgamma

    # needed to make e.g. sampling work
    @nograd similar, randn!, fill!

    include(joinpath("pdfs", "abstract_pdfs.jl"))

    include(joinpath("utils", "nogradarray.jl"))
    include(joinpath("utils", "saveload.jl"))
    include(joinpath("utils", "utils.jl"))
    include(joinpath("utils", "vmf.jl"))
    include(joinpath("utils", "flux_ode_decoder.jl"))

    include(joinpath("pdfs", "gaussian.jl"))
    include(joinpath("pdfs", "hs_uniform.jl"))
    include(joinpath("pdfs", "vonmisesfisher.jl"))
    include(joinpath("pdfs", "abstract_cgaussian.jl"))
    include(joinpath("pdfs", "cmean_gaussian.jl"))
    include(joinpath("pdfs", "cmeanvar_gaussian.jl"))
    include(joinpath("pdfs", "abstract_cvmf.jl"))
    include(joinpath("pdfs", "cmeanconc_vmf.jl"))

    include(joinpath("models", "vae.jl"))
    include(joinpath("models", "rodent.jl"))
    include(joinpath("models", "gan.jl"))
    include(joinpath("models", "svae.jl"))

end # module
