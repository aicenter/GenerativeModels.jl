using Test
using Suppressor
using Logging
using Parameters
using Random
using BSON
using DrWatson
using ValueHistories

using Flux
using ForwardDiff
using ConditionalDists

using Revise
using GenerativeModels

if Flux.use_cuda[] using CuArrays end

# set logging to debug to get more test output
logger = ConsoleLogger(stdout, Logging.Info)
global_logger(logger)

# for testing of parameter change in training
get_params(model) =  map(copy, collect(params(model)))
param_change(frozen_params, model) = 
	map(x-> x[1] != x[2], zip(frozen_params, collect(params(model))))

# include(joinpath("models", "vae.jl"))
include(joinpath("models", "gan.jl"))
include(joinpath("models", "rodent.jl"))

include(joinpath("utils", "flux_ode_decoder.jl"))
include(joinpath("utils", "saveload.jl"))
include(joinpath("utils", "utils.jl"))
