export FluxODEDecoder

"""
    FluxODEDecoder{M}(slength::Int, tlength::Int, tspan::Tuple,
                      model::M, observe::Function)

Uses a Flux `model` as ODE and solves it for the given time span.
The solver can be conveniently called by (dec::FluxODEDecoder)(z), which
assumes that all parameters of the neural ODE and its initial conditions are
passed in as one long vector i.e.: z = vcat(destructure(dec.model), u0).
Can use any Flux model as neural ODE. The adjoint is computed via ForwardDiff.

# Arguments
* `slength`: length of the ODE state
* `tlength`: number of ODE solution samples
* `tspan`: time span for which ODE is solved
* `model`: Flux model
* `observe`: Observation operator. Function that receives ODESolution and
  outputs the observation. Default: observe(sol) = reshape(hcat(sol.u...),:)
"""
mutable struct FluxODEDecoder
    slength::Int
    timesteps::Vector
    model
    observe::Function
    _zlength::Int
end

# TODO: FluxODEDecoder{M} fails during training because forward diff wants to
#       stick dual number in there...
function FluxODEDecoder(slength::Int, tlength::Int, dt::T,
                        model, observe::Function) where T
    timesteps = range(T(0), step=dt, length=tlength)
    _zlength = length(destructure(model)) + slength
    FluxODEDecoder(slength, timesteps, model, observe, _zlength)
end

function FluxODEDecoder(slength::Int, tlength::Int, dt::Real, model)
    observe(sol) = reshape(hcat(sol.u...), :)
    FluxODEDecoder(slength, tlength, dt, model, observe)
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
