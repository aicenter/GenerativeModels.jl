using Test
using Logging
using Random

using Flux
using Distributions
using ConditionalDists
using IPMeasures
using GenerativeModels

using ConditionalDists: SplitLayer

if Flux.use_cuda[]
    using CUDA
    CUDA.allowscalar(false)
end

# set logging to debug to get more test output
logger = ConsoleLogger(stdout, Logging.Info)
global_logger(logger)

# for testing of parameter change in training
get_params(model) =  map(copy, collect(Flux.params(model)))
param_change(frozen_params, model) = 
	map(x-> x[1] != x[2], zip(frozen_params, collect(Flux.params(model))))

include("utils.jl")
include("vae.jl")
include("gan.jl")
include("vamp.jl")
include("statistician.jl")
