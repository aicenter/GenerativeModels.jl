using Test
using Logging

using Flux
using ValueHistories

using Revise
using GenerativeModels

@testset "GenerativeModels.jl" begin
    
include(joinpath("models", "ard_autoencoder.jl"))

include(joinpath("utils", "misc.jl"))

end
