module GenerativeModels

    using Reexport

    @reexport using BSON
    @reexport using DrWatson
    @reexport using ValueHistories
    @reexport using PyPlot

    @reexport using Flux
    @reexport using DifferentialEquations
    @reexport using DiffEqFlux
    @reexport using LinearAlgebra

    abstract type AbstractAutoEncoder end

    include(joinpath("utils", "ode.jl"))
    include(joinpath("utils", "misc.jl"))
    include(joinpath("utils", "visualize.jl"))

    include(joinpath("models", "ard_autoencoder.jl"))

end # module
