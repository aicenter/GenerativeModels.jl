module GenerativeModels

using Flux

abstract type AbstractAutoEncoder end

include("models/ard_autoencoder.jl")

end # module
