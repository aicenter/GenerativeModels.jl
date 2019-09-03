using Test
using Logging

using DifferentialEquations
using DiffEqFlux

using Revise
using GenerativeModels

@testset "GenerativeModels.jl" begin
    
include(joinpath("models", "vae.jl"))
include(joinpath("models", "ard_vae.jl"))

include(joinpath("utils", "misc.jl"))

end
