@testset "utils/misc.jl" begin
    @info "Testing utils/misc.jl"

    xsize = 10
    zsize = 8
    (decoder, _) = make_ode_decoder(xsize, (0f0,1f0), 2)
    encoder = Dense(xsize, zsize)
    model = VAE{Float32}(xsize, zsize, encoder, decoder)

    warn_logger = SimpleLogger(stdout, Logging.Warn)
    model_dir   = mktempdir()
    @debug "  model_dir: $model_dir"


    @debug "  Testing `save_checkpoint`"
    model_ckpt = joinpath(model_dir, "ckpt.bson")
    history = MVHistory()
    push!(history, :loss, 1, 1)
    with_logger(warn_logger) do
        save_checkpoint(model_ckpt, model, history)
    end

    @test isfile(model_ckpt)


    @debug "  Testing `load_checkpoint`"
    ckpt = with_logger(warn_logger) do 
        load_checkpoint(model_ckpt)
    end
    loaded_model = ckpt[:model]
    loaded_history = ckpt[:history]
    @test model.encoder.W == loaded_model.encoder.W

    z = randn(Float32, zsize)
    (x, _) = decoder_mean_var(model, z)
    @test size(x) == (xsize,)
end
