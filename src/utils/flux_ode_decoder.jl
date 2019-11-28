export FluxODEDecoder

"""
    FluxODEDecoder(slength::Int, tlength::Int, timesteps::Vector, model)

Can use any Flux model as neural ODE. The adjoint is computed via ForwardDiff.
(dec::FluxODEDecoder)(z) assumes that all parameters to the model
and the initial conditions to the neural ODE are passed in as one long vector
z = vcat(destructure(dec.model), u0).

# Arguments
* `slength::Int`: length of the ODE state
* `timesteps::Vector`: timesteps at which ODE solution is checkpointed
* `model`: Flux model
* `observe`: Observations operator. Function that receives ODESolution and
  outputs the observation. Default: observe(sol) = reshape(hcat(sol.u...),:)
"""
mutable struct FluxODEDecoder
    slength::Int
    timesteps::Vector
    model
    observe::Function
    _zlength::Int
end

function FluxODEDecoder(slength::Int, tlength::Int, tspan::Tuple, model, observe::Function)
    timesteps = range(tspan[1], stop=tspan[2], length=tlength)
    _zlength = length(destructure(model)) + slength
    FluxODEDecoder(slength, timesteps, model, observe, _zlength)
end

function FluxODEDecoder(slength::Int, tlength::Int, tspan::Tuple, model)
    observe(sol) = reshape(hcat(sol.u...), :)
    FluxODEDecoder(slength, tlength, tspan, model, observe)
end

function (dec::FluxODEDecoder)(z::AbstractVector, observe::Function)
    @assert length(z) == dec._zlength
    ps = z[1:end-dec.slength]
    u0 = z[end-dec.slength+1:end]
    dec.model = restructure(dec.model, ps)
    z = vcat(destructure(dec.model), u0)
    tspan = (dec.timesteps[1], dec.timesteps[end])
    dudt_(u::AbstractVector, ps, t) = dec.model(u)
    prob = ODEProblem(dudt_, u0, tspan, ps)
    sol = solve(prob, Tsit5(), saveat=dec.timesteps)
    observe(sol)
end

# by default call with stored observe function
(dec::FluxODEDecoder)(z::AbstractVector) = dec(z, dec.observe)

# Use loop to get batched reconstructions so that jacobian and @adjoint work...
(dec::FluxODEDecoder)(Z::AbstractMatrix) = hcat([dec(Z[:,ii]) for ii in 1:size(Z,2)]...)

ddec(dec::FluxODEDecoder, z::AbstractVector) = ForwardDiff.jacobian(dec, z)

@adjoint function (dec::FluxODEDecoder)(z::AbstractVector)
    (dec(z), Δ -> (J=(Δ'*ddec(dec, z))'; (nothing,J)))
end
