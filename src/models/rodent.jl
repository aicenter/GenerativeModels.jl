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

julia> rodent = Rodent(xlen, tspan, order)

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

function order2ode(du, u, p, t)
    du[1] = p[1]*u[1] + p[2]*u[2] + p[5]
    du[2] = p[3]*u[1] + p[4]*u[2] + p[6]
end

function order3ode(du, u, p, t)
    du[1] = p[1]*u[1] + p[2]*u[2] + p[3]*u[3] + p[10]
    du[2] = p[4]*u[1] + p[5]*u[2] + p[6]*u[3] + p[11]
    du[3] = p[7]*u[1] + p[8]*u[2] + p[9]*u[3] + p[12]
end

get_nr_ode_params(order) = order^2 + order*2

_ODE = Dict(
    2 => order2ode,
    3 => order3ode
)

"""
    make_ode_decoder(xlength::Int, tspan::Tuple{T,T}, order::Int)

Creates an ODE solver function that solves an ODE of order N.
The last parameters are assumed to be the initial conditions to the ODE.

Returns the ODE solver function and a named tuple that contains the ODE problem
setup.
"""
function make_ode_decoder(xlength::Int, tspan::Tuple{T,T}, order::Int) where T
    global _ODE
    ODEProblem = DifferentialEquations.ODEProblem
    Tsit5 = DifferentialEquations.Tsit5
    diffeq_rd = DiffEqFlux.diffeq_rd
    nr_ode_ps = get_nr_ode_params(order)

    ode_ps = rand(T, nr_ode_ps)
    u0_func(ode_ps, t0) = [p for p in ode_ps[end-order+1:end]]
    ode_prob = ODEProblem(_ODE[order], u0_func, tspan, ode_ps)
    timesteps = range(tspan[1], stop=tspan[2], length=xlength)

    function decode(ode_ps)
        sol = diffeq_rd(ode_ps, ode_prob, Tsit5())
        res = Tracker.collect(sol(timesteps)[1,:])
    end

    function decoder(Z)
        @assert size(Z, 1) == nr_ode_ps

        if Base.length(size(Z)) == 1
            decode(Z)
        elseif Base.length(size(Z)) == 2
            U = [decode(Z[:, ii]) for ii in 1:size(Z, 2)]
            hcat(U...)
        else
            error("Latent input must be either vector or matrix!")
        end
    end

    decoder, (ode_ps=ode_ps, u0_func=u0_func, ode_prob=ode_prob, timesteps=timesteps)
end

function Rodent(xlen::Int, tspan::Tuple{T,T}, order::Int) where T
    zlen = get_nr_ode_params(order)

    λ2z = param(ones(T, zlen))
    prior = Gaussian(zeros(T, zlen), λ2z)

    σ2z = param(ones(T, zlen))
    μz  = Dense(xlen, zlen)
    encoder = SharedVarCGaussian{T}(zlen, xlen, μz, σ2z)

    (μx, _) = make_ode_decoder(xlen, tspan, order)
    σ2x = param(ones(T, xlen))
    decoder = SharedVarCGaussian{T}(xlen, zlen, μx, σ2x)

    Rodent{T}(prior, encoder, decoder)
end
