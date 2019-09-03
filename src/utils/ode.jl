export order2ode
export make_ode_decoder


function order2ode(du, u, p, t)
    du[1] = p[1]*u[1] + p[2]*u[2] + p[5]
    du[2] = p[3]*u[1] + p[4]*u[2] + p[6]
end

function order3ode(du, u, p, t)
    du[1] = p[1]*u[1] + p[2]*u[2] + p[3]*u[3] + p[10]
    du[2] = p[4]*u[1] + p[5]*u[2] + p[6]*u[3] + p[11]
    du[3] = p[7]*u[1] + p[8]*u[2] + p[9]*u[3] + p[12]
end

_ODE = Dict(
    2 => order2ode,
    3 => order3ode
)

"""`make_ode_decoder(xsize::Int, tspan::Tuple{T,T})

Creates an ODE solver function that solves an ODE of 2nd or 3rd order.
The last parameters are assumed to be the initial conditions to the ODE.

Returns the ODE solver function and a named tuple that contains the ODE problem
setup.
"""
function make_ode_decoder(xsize::Int, tspan::Tuple{T,T}, order::Int) where T
    global _ODE
    ODEProblem = DifferentialEquations.ODEProblem
    Tsit5 = DifferentialEquations.Tsit5
    diffeq_rd = DiffEqFlux.diffeq_rd
    nr_ode_ps = order^2 + order*2

    ode_ps = rand(T, nr_ode_ps)
    u0_func(ode_ps, t0) = [p for p in ode_ps[end-order+1:end]]
    ode_prob = ODEProblem(_ODE[order], u0_func, tspan, ode_ps)
    timesteps = range(tspan[1], stop=tspan[2], length=xsize)

    function decode(ode_ps)
        sol = diffeq_rd(ode_ps, ode_prob, Tsit5())
        res = Tracker.collect(sol(timesteps)[1,:])
    end

    function decoder(Z)
        # TODO: this can be done with mapslices with Zygote -> parallelize?!
        @assert size(Z, 1) == nr_ode_ps

        if length(size(Z)) == 1
            decode(Z)
        elseif length(size(Z)) == 2
            U = [decode(Z[:, ii]) for ii in 1:size(Z, 2)]
            hcat(U...)
        else
            error("Latent input must be either vector or matrix!")
        end
    end

    decoder, (ode_ps=ode_ps, u0_func=u0_func, ode_prob=ode_prob, timesteps=timesteps)
end
