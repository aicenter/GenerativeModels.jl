export save_checkpoint
export load_checkpoint


function save_checkpoint(filename::String, model::AbstractGM,
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
