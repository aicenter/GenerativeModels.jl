export save_checkpoint
export load_checkpoint


function save_checkpoint(filename::String, model::AbstractAutoEncoder,
                         history::MVHistory; keep=100)
    d = DrWatson.@dict(model, history)

    @info "Saving checkpoint at $filename"
    DrWatson.tagsave(filename, d, true)

    (name, ext) = splitext(filename)
    oldest_ckpt = "$(name)_#$(keep)$(ext)"
    if isfile(oldest_ckpt)
        rm(oldest_ckpt)
        @info "Removed oldest checkpoint: $oldest_ckpt"
    end
end


function load_checkpoint(filename::String)
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


function mvhistory_callback(h::MVHistory, m::AbstractAutoEncoder, lossf::Function, test_data::AbstractArray)
    function callback()
        (μz, σz, γ) = encoder_params(m, test_data)
        σe = m.σe[1]
        xrec = decode(m, μz .* γ)
        loss = lossf(test_data)
        ntuple = DrWatson.@ntuple μz σz γ xrec loss σe
        GenerativeModels.push_ntuple!(h, ntuple)
    end
end


"""
Generate pairs of sine waves and their generating frequency
"""
function generate_sine_data(nr_examples; steps=30, dt=pi/10, freq_range=[1.0, 1.2])
    U = []
    T = []
    Omega = []
    for ii in 1:nr_examples
        omega = freq_range[1] + rand() * sum(freq_range[2] - freq_range[1])
        start = rand() * 2pi

        t = range(start, length=steps, step=dt)
        u = sin.(omega * t)
        push!(U, Float32.(u))
        push!(T, Float32.(t))
        push!(Omega, Float32.(omega))
    end

    U = hcat(U...)
    T = hcat(T...)

    U, T, Omega
end
