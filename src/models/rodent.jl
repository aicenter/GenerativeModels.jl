export Rodent
export make_ode_decoder

"""
    Rodent{T}(p::Gaussian{T}, e::SharedVarCGaussian{T}, d::SharedVarCGaussian{T})

Variational Auto-Encoder with shared variances.
Provides a constructor that creates a VAE with ARD prior and an ODE decoder.

# Example
With a 2nd order ODE decoder you can solve a harmonic ODE with ξ̇=Wξ+b. 
Setting `W = [0 1; -1 0]`; `b = [0,0]`; `ξ₀=[0,1]` will create a sine wave.
All ODE params are collected in z = {W,b,ξ₀}.

```julia-repl
julia> xlen = 5; tspan = (0f0, Float32(2π)); order = 2;

julia> z = Float32.([0, 1, -1, 0, 0, 0, 0, 1]);

julia> encoder = Dense(xlen, length(z))

julia> rodent = Rodent(xlen, encoder, tspan, order)

julia> mean(rodent.decoder, z)
Tracked 5-element Array{Float32,1}:
  0.0f0
  1.0000039f0
 -7.0631504f-6
 -1.0000119f0
  8.059293f-5
```
"""
struct Rodent{T} <: AbstractVAE{T}
    prior::Gaussian
    encoder::SharedVarCGaussian
    decoder::SharedVarCGaussian

    function Rodent{T}(p::Gaussian{T}, e::SharedVarCGaussian{T}, d::SharedVarCGaussian{T}) where T
        if xlength(e) == zlength(d)
            new(p, e, d)
        else
            error("Encoder and decoder must have same zlength.")
        end
    end
end

Flux.@functor Rodent

Rodent(p::Gaussian{T}, e::SharedVarCGaussian{T}, d::SharedVarCGaussian{T}) where T = Rodent{T}(p, e, d)

function Rodent(xlen::Int, zlen::Int, encoder, decoder, T=Float32)
    λ2z = ones(T, zlen)
    μpz = SVector{zlen}(zeros(T, zlen))
    prior = Gaussian(μpz, λ2z)

    σ2z = ones(T, zlen)
    enc_dist = SharedVarCGaussian{T}(zlen, xlen, encoder, σ2z)

    σ2x = ones(T, 1)
    dec_dist = SharedVarCGaussian{T}(xlen, zlen, decoder, σ2x)

    Rodent{T}(prior, enc_dist, dec_dist)
end

ode_params_length(order) = order^2 + order*2

# """
#     make_ode_decoder(xlength::Int, tspan::Tuple{T,T}, order::Int)
# 
# Creates an ODE solver function that solves an ODE of order N.
# The last parameters are assumed to be the initial conditions to the ODE.
# 
# Returns the ODE solver function and a named tuple that contains the ODE problem
# setup.
# """
# function make_ode_decoder(xlength::Int, tspan::Tuple{T,T}, order::Int) where T
#     timesteps = range(tspan[1], stop=tspan[2], length=xlength)
# 
#     function ode(u, p, t)
#         (A, b, _) = p
#         du = A*u + b
#     end
# 
#     function decode(A::AbstractMatrix, b::AbstractVector, u0::AbstractVector)
#         p = (A,b,u0)
#         #_prob = remake(prob; u0=convert.(eltype(A),u0), p=z) # TODO: use remake instead?
#         prob = ODEProblem(ode, u0, tspan, p)
#         sol = solve(prob, Tsit5(), saveat=1) #TODO: maybe use Vern9 ????
#         res = hcat(sol.u...)[1,:]
#     end
#     
#     function split_latent(z::AbstractVector)
#         A = reshape(z[1:order^2], order, order)
#         b = z[order^2+1:order^2+order]
#         u = z[end-order+1:end]
#         return (A,b,u)
#     end
#     
#     decode(z::AbstractVector) = decode(split_latent(z)...)
#     decode(Z::AbstractMatrix) = hcat([decode(Z[:,ii]) for ii in 1:size(Z,2)]...)
# 
#     ddec(z::AbstractVector) = ForwardDiff.jacobian(decode, z)
#     @adjoint decode(z::AbstractVector) = (decode(z), Δ -> (J = Δ' * ddec(z); (J',)))
# 
#     return decode
# end
