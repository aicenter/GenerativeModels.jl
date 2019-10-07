@testset "utils/saveload.jl" begin
    @info "Testing save/load of model checkpoints"

    xlength = 10
    keep = 2
    decoder = make_ode_decoder(xlength, (0f0,1f0), 2)
    encoder = Dense(xlength, 8)
    model = Rodent(xlength, encoder, (0f0, 1f0), 2)

    nowarn_logger = SimpleLogger(stdout, Logging.Error)
    model_dir   = mktempdir()
    @debug "  model_dir: $model_dir"


    @debug "  Testing `save_checkpoint`"
    model_ckpt = joinpath(model_dir, "ckpt.bson")
    history = MVHistory()
    push!(history, :loss, 1, 1)
    with_logger(nowarn_logger) do
        save_checkpoint(model_ckpt, model, history, keep=keep)
        save_checkpoint(model_ckpt, model, history, keep=keep)
        save_checkpoint(model_ckpt, model, history, keep=keep)
    end

    @test isfile(model_ckpt)
    @test length(readdir(dirname(model_ckpt))) == keep


    @debug "  Testing `load_checkpoint`"
    loaded_model, history = with_logger(nowarn_logger) do 
        load_checkpoint(model_ckpt)
    end
    @test model.encoder.mapping.W == loaded_model.encoder.mapping.W

    opt = ADAM()
    lossf(x) = elbo(model, x, β=1e-3)
    data = [(randn(Float32, xlength),)]
    Flux.train!(lossf, params(model), data, opt)
    params_trained = get_params(model)

    with_logger(nowarn_logger) do
        save_checkpoint(model_ckpt, model, history, keep=keep)
    end

    loaded_model, history = with_logger(nowarn_logger) do 
        load_checkpoint(model_ckpt)
    end

    @test !any(param_change(params_trained, loaded_model)) # did the params change?
    @test size(mean(loaded_model.encoder, randn(Float32, xlength))) == (8,)
end
