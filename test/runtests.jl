using Test
using Logging

using DifferentialEquations
using DiffEqFlux

using Revise
using GenerativeModels

# for testing of parameter change in training
get_params(model) =  map(x->copy(Flux.Tracker.data(x)), collect(params(model)))
param_change(frozen_params, model) = 
	map(x-> x[1] != x[2], zip(frozen_params, collect(params(model))))

@testset "GenerativeModels.jl" begin

    include(joinpath("pdfs", "gaussian.jl"))
    include(joinpath("pdfs", "cgaussian.jl"))
    include(joinpath("pdfs", "svar_cgaussian.jl"))
        
    include(joinpath("models", "vae.jl"))
    include(joinpath("models", "rodent.jl"))
    
    # include(joinpath("utils", "misc.jl"))

end
