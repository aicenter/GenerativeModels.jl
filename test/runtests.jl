using Test
using Logging

using DifferentialEquations
using DiffEqFlux

using Revise
using GenerativeModels

@testset "GenerativeModels.jl" begin

include(joinpath("pdfs", "gaussian.jl"))
include(joinpath("pdfs", "cgaussian.jl"))
    
include(joinpath("models", "vae.jl"))
# include(joinpath("models", "ard_vae.jl"))

# include(joinpath("utils", "misc.jl"))

end
