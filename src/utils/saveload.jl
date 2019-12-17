export save_checkpoint
export load_checkpoint


"""
    save_checkpoint(filename, model, history; keep=100)

Save a model and training history at a given filename.
If the checkpoint already exists it is moved to filename_#1.bson.
`keep` specifies the number of checkpoints that are kept.
"""
function save_checkpoint(filename::String, model::AbstractGM,
                         history::MVHistory; keep=100)
    model = model |> cpu
    d = @dict model history

    @info "Saving checkpoint at $filename"
    DrWatson.tagsave(filename, d, safe=true)

    (name, ext) = splitext(filename)
    oldest_ckpt = "$(name)_#$(keep)$(ext)"
    if isfile(oldest_ckpt)
        rm(oldest_ckpt)
        @info "Removed oldest checkpoint: $oldest_ckpt"
    end
end


"""
    load_checkpoint(filename::String)::Tuple{AbstractGM,MVHistory}

Loads a checkpoint saved with `save_checkpoint` and returns the model and its
training history
"""
function load_checkpoint(filename::String)::Tuple{AbstractGM,MVHistory}
    @info "Loading checkpoint from $filename"
    res = BSON.load(filename)
    res[:model], res[:history]
end
