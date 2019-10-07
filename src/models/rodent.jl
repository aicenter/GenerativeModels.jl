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

Flux.@treelike Rodent

Rodent(p::Gaussian{T}, e::SharedVarCGaussian{T}, d::SharedVarCGaussian{T}) where T = Rodent{T}(p, e, d)

ode_params_length(order) = order^2 + order*2

"""
    make_ode_decoder(xlength::Int, tspan::Tuple{T,T}, order::Int)

Creates an ODE solver function that solves an ODE of order N.
The last parameters are assumed to be the initial conditions to the ODE.

Returns the ODE solver function and a named tuple that contains the ODE problem
setup.
"""
function make_ode_decoder(xlength::Int, tspan::Tuple{T,T}, order::Int) where T
    ODEProblem = DifferentialEquations.ODEProblem
    diffeq_rd = DiffEqFlux.diffeq_rd
    Tsit5 = DifferentialEquations.Tsit5
    timesteps = range(tspan[1], stop=tspan[2], length=xlength)

    function ode(u, p, t)
        (A, b, _) = p
        du = A*u + b
    end

    function decode(A::AbstractMatrix, b::AbstractVector, u::AbstractVector)
        z = (A,b,u)
        ode_prob = ODEProblem(ode, u, tspan, z)
        sol = diffeq_rd(z, ode_prob, Tsit5(), saveat=timesteps)
        res = hcat(sol.u...)[1,:]
    end

    function decode(A::AbstractArray, b::AbstractMatrix, u::AbstractMatrix)
        X = [decode(A[:,:,ii], b[:,ii], u[:,ii]) for ii in 1:size(A, 3)]
        hcat(X...)
    end

    function decode(z::AbstractVector)
        A = reshape(z[1:order^2], order, order)
        b = z[order^2+1:order^2+order]
        u = z[end-order+1:end]
        decode(A, b, u)
    end

    function decode(Z::AbstractMatrix)
        A = reshape(Z[1:order^2, :], order, order, :)
        b = Z[order^2+1:order^2+order, :]
        u = Z[end-order+1:end, :]
        decode(A, b, u)
    end

    decode
end

function Rodent(xlen::Int, encoder, tspan::Tuple{T,T}, order::Int) where T
    zlen = ode_params_length(order)

    λ2z = param(ones(T, zlen))
    prior = Gaussian(zeros(T, zlen), λ2z)

    σ2z = param(ones(T, zlen))
    enc_dist = SharedVarCGaussian{T}(zlen, xlen, encoder, σ2z)

    μx  = make_ode_decoder(xlen, tspan, order)
    σ2x = param(ones(T, 1))
    dec_dist = SharedVarCGaussian{T}(xlen, zlen, μx, σ2x)

    Rodent{T}(prior, enc_dist, dec_dist)
end
