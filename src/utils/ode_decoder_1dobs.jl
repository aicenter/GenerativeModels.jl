export ODEDecoder

"""
    ODEDecoder

exemplary implementation of ODE decoder that reconstructs only first element
of the internal state (see function: (dec::ODEDecoder)(A,b,u0))
"""
struct ODEDecoder
    order::Int
    xlength::Int
    timesteps::Vector
end

function ODEDecoder(order::Int, xlength::Int, tspan::Tuple)
    timesteps = range(tspan[1], stop=tspan[2], length=xlength)
    ODEDecoder(order, xlength, timesteps)
end

function ode(u, p, t)
    (A, b, _) = p
    du = A*u + b
end

function split_latent(z::AbstractVector, order::Int)
    A = reshape(z[1:order^2], order, order)
    b = z[order^2+1:order^2+order]
    u = z[end-order+1:end]
    return (A,b,u)
end

function (dec::ODEDecoder)(A::AbstractMatrix, b::AbstractVector, u0::AbstractVector)
    p = (A,b,u0)
    #_prob = remake(prob; u0=convert.(eltype(A),u0), p=z) # TODO: use remake instead?
    tspan = (dec.timesteps[1], dec.timesteps[end])
    prob = ODEProblem(ode, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=dec.timesteps) #TODO: maybe use Vern9 ????

    # return time series of only first element of the state
    res = hcat(sol.u...)[1,:]
    # TODO: extend this to full state by reshaping
end

function (dec::ODEDecoder)(z::AbstractVector)
    (A,b,u0) = split_latent(z, dec.order)
    dec(A,b,u0)
end

# Use loop to get batched reconstructions so that jacobian and @adjoint work...
(dec::ODEDecoder)(Z::AbstractMatrix) = hcat([dec(Z[:,ii]) for ii in 1:size(Z,2)]...)

ddec(dec::ODEDecoder, z::AbstractVector) = ForwardDiff.jacobian(dec, z)

@adjoint function (dec::ODEDecoder)(z::AbstractVector)
    (dec(z), Δ -> (J=Δ'*ddec(dec, z); (nothing,J')))
    # TODO: why is the nothing needed???
end
