@testset "Neural Statistician" begin
    Random.seed!(0)
    # create dummy data
    N = 10
    data = [randn(3,rand(Poisson(30))) .+ 5 for i in 1:N]

    # number of neurons
    xdim = 3
    hdim = 8
    vdim = 4
    cdim = 4
    zdim = 2

    # model components
    instance_enc = Chain(Dense(xdim, hdim, swish), Dense(hdim, hdim, swish), Dense(hdim, vdim))

    enc_c = Chain(Dense(vdim,hdim,swish),Dense(hdim,hdim,swish), SplitLayer(hdim, [cdim,cdim], [identity,softplus]))
    enc_c_dist = ConditionalMvNormal(enc_c)

    cond_z = Chain(Dense(cdim,hdim,swish),Dense(hdim,hdim,swish),SplitLayer(hdim, [zdim,zdim], [identity,softplus]))
    cond_z_dist = ConditionalMvNormal(cond_z)

    enc_z = Chain(Dense(cdim+vdim,hdim,swish),Dense(hdim,hdim,swish), SplitLayer(hdim, [zdim,zdim], [identity,softplus]))
    enc_z_dist = ConditionalMvNormal(enc_z)

    dec = Chain(Dense(zdim,hdim,swish),Dense(hdim,hdim,swish), SplitLayer(hdim, [xdim,1], [identity,softplus]))
    dec_dist = ConditionalMvNormal(dec)

    # model initialization
    model = NeuralStatistician(instance_enc, cdim, enc_c_dist, cond_z_dist, enc_z_dist, dec_dist)

    # training init
    params_init = get_params(model)
    ps = Flux.params(model)
    @test length(ps) > 0

    # loss
    # extend kl_divergence from IPMeasures
    using IPMeasures: _kld_gaussian
    (m::KLDivergence)(p::ConditionalDists.BMN, q::ConditionalDists.BMN) = _kld_gaussian(p,q)
    loss(x) = -elbo(model,x)

    ls = mean(loss.(data))
    @test isa(ls, Real)
    
    # train the model
    opt = ADAM()
    Flux.train!(loss,ps,data,opt)

    # test change of parameters
    @test all(param_change(params_init, model))

    # test loss reduction
    new_ls = mean(loss.(data))
    @test new_ls < ls
end