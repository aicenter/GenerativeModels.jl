module GenerativeModels

using BSON
using DrWatson
using ValueHistories

using Flux
using DifferentialEquations
using DiffEqFlux

abstract type AbstractAutoEncoder end

include(joinpath("utils", "ode.jl"))
include(joinpath("utils", "misc.jl"))

include(joinpath("models", "ard_autoencoder.jl"))

end # module
