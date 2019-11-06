export FluxODEDecoder

mutable struct FluxODEDecoder
    slength::Int
    tlength::Int
    timesteps::Vector
    model
end

function FluxODEDecoder(slength::Int, tlength::Int, tspan::Tuple, model)
    timesteps = range(tspan[1], stop=tspan[2], length=tlength)
    FluxODEDecoder(slength, tlength, timesteps, model)
end

function (dec::FluxODEDecoder)(z::AbstractVector)
    ps = z[1:end-dec.slength]
    u0 = z[end-dec.slength+1:end]
    dec.model = restructure(dec.model, ps)
    z = vcat(destructure(dec.model), u0)
    tspan = (dec.timesteps[1], dec.timesteps[end])
    dudt_(u::AbstractVector, ps, t) = dec.model(u)
    prob = ODEProblem(dudt_, u0, tspan, ps)
    sol = solve(prob, Tsit5(), saveat=dec.timesteps)
    reshape(hcat(sol.u...), :)
end

# Use loop to get batched reconstructions so that jacobian and @adjoint work...
(dec::FluxODEDecoder)(Z::AbstractMatrix) = hcat([dec(Z[:,ii]) for ii in 1:size(Z,2)]...)

ddec(dec::FluxODEDecoder, z::AbstractVector) = ForwardDiff.jacobian(dec, z)

@adjoint function (dec::FluxODEDecoder)(z::AbstractVector)
    (dec(z), Δ -> (J=(Δ'*ddec(dec, z))'; (nothing,J)))
end
