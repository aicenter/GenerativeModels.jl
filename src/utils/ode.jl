export order2ode
export make_order2ode_decoder


function order2ode(du, u, p, t)
    du[1] = p[1]*u[1] + p[2]*u[2] + p[5]
    du[2] = p[3]*u[1] + p[4]*u[2] + p[6]
end


"""`make_sine_ode_decoder(xsize::Int, tspan::Tuple{T,T})

Creates an ODE solver function that solves an ODE of 2nd order with six free
model parameters + two free inital condition parameters.
The last two parameters are assumed to be the initial conditions to the ODE.

Returns the ODE solver function and a named tuple that contains the ODE problem
setup.
"""
function make_order2ode_decoder(xsize::Int, tspan::Tuple{T,T}) where T
    NR_ODE_PS = 8

    ode_ps = rand(T, NR_ODE_PS)
    u0_func(ode_ps, t0) = [ode_ps[end-1], ode_ps[end]]
    ode_prob = ODEProblem(order2ode, u0_func, tspan, ode_ps)
    timesteps = range(tspan[1], stop=tspan[2], length=xsize)

    function decode(ode_ps)
        sol = diffeq_rd(ode_ps, ode_prob, Tsit5())
        res = Tracker.collect(sol(timesteps)[1,:])
    end

    function decoder(Z)
        # TODO: this can be done with mapslices with Zygote -> parallelize?!
        @assert size(Z, 1) == NR_ODE_PS

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


function save_checkpoint(filename::String, model::AbstractAutoEncoder,
                         history::MVHistory; keep=100)
    d = DrWatson.@dict(model, history)
    # if save_svi
    #     svi_stats = model.svi_stats
    #     d = DrWatson.@dict(model, svi_stats, history)
    # end

    @info "Saving checkpoint at $filename"
    DrWatson.tagsave(filename, d, true)

    (name, ext) = splitext(filename)
    oldest_ckpt = "$(name)_#$(keep)$(ext)"
    if isfile(oldest_ckpt)
        rm(oldest_ckpt)
        @info "Removed oldest checkpoint: $oldest_ckpt"
    end
end

function load_checkpoint!(filename::String)
    @info "Loading checkpoint from $filename"
    BSON.load(filename)
end

function push_ntuple!(history::MVHistory, ntuple::NamedTuple; idx=nothing)
    if idx == nothing
        _keys = keys(history)
        if length(_keys) > 0
            idx = length(history, first(_keys)) + 1
        else
            idx = 1
        end
    end

    for (name, value) in pairs(ntuple)
        if isa(value, TrackedArray) || isa(value, Tracker.TrackedReal)
            value = value.data
        end
        push!(history, name, idx, deepcopy(value))
    end
end


function mvhistory_callback(history::MVHistory)
    function cb(ntuple)
        push_ntuple!(history, ntuple)
    end
end
