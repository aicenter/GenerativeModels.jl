module GenerativeModels

    using Reexport
    using Requires
    using Statistics

    @reexport using BSON
    @reexport using DrWatson
    @reexport using ValueHistories

    @reexport using Flux
    @reexport using LinearAlgebra

    abstract type AbstractGM end

    include(joinpath("utils", "misc.jl"))
    include(joinpath("utils", "utils.jl"))


    # optional dependencies
    function __init__()
        @require PyPlot="d330b81b-6aea-500a-939a-2ce795aea3ee" include(joinpath("utils", "visualize.jl"))
        @require DiffEqFlux="aae7a2af-3d4f-5e19-a356-7da93b79d9d0" @reexport using DifferentialEquations
        @require DiffEqFlux="aae7a2af-3d4f-5e19-a356-7da93b79d9d0" include(joinpath("utils", "ode.jl"))
    end

    include(joinpath("models", "abstract_vae.jl"))
    include(joinpath("models", "vae.jl"))
    include(joinpath("models", "ard_vae.jl"))

end # module
