module GenerativeModels

using DrWatson
using ValueHistories

using Flux
using DifferentialEquations
using DiffEqFlux

abstract type AbstractAutoEncoder end

include("models/ard_autoencoder.jl")

include("utils/ode.jl")

end # module
