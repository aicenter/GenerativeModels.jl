export ODEDecoder

"""
    ODEDecoder(slength::Int, tlength::Int, tspan::Tuple)

ODE decoder that reconstructs only first element of the internal state (see
function: (dec::ODEDecoder)(A,b,u0))
"""
struct ODEDecoder
    slength::Int
    tlength::Int
    timesteps::Vector
end

function ODEDecoder(slength::Int, tlength::Int, tspan::Tuple)
    timesteps = range(tspan[1], stop=tspan[2], length=tlength)
    ODEDecoder(slength, tlength, timesteps)
end

function ode(u, p, t)
    (A, b, _) = p
    du = A*u + b
end

ode_params_length(slength::Int) = slength^2 + slength*2

function ode_latent_split(z::AbstractVector, slength::Int)
    @assert length(z) == ode_params_length(slength)
    A = reshape(z[1:slength^2], slength, slength)
    b = z[slength^2+1:slength^2+slength]
    u = z[end-slength+1:end]
    return (A,b,u)
end

function (dec::ODEDecoder)(A::AbstractMatrix, b::AbstractVector, u0::AbstractVector)
    p = (A,b,u0)
    #_prob = remake(prob; u0=convert.(eltype(A),u0), p=z) # TODO: use remake instead?
    tspan = (dec.timesteps[1], dec.timesteps[end])
    prob = ODEProblem(ode, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=dec.timesteps) #TODO: maybe use Vern9 ????
    res = reshape(hcat(sol.u...), :)
end

function (dec::ODEDecoder)(z::AbstractVector)
    (A,b,u0) = ode_latent_split(z, dec.slength)
    dec(A,b,u0)
end

# Use loop to get batched reconstructions so that jacobian and @adjoint work...
(dec::ODEDecoder)(Z::AbstractMatrix) = hcat([dec(Z[:,ii]) for ii in 1:size(Z,2)]...)

ddec(dec::ODEDecoder, z::AbstractVector) = ForwardDiff.jacobian(dec, z)

@adjoint function (dec::ODEDecoder)(z::AbstractVector)
    (dec(z), Δ -> (J=Δ'*ddec(dec, z); (nothing,J')))
    # TODO: why is the nothing needed???
end
