@testset "src/models/rodent.jl" begin

    Random.seed!(1)

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

    function generate(ω::T, batch::Int; ω0=0.5, noise=0.01, dt=0.1, steps=20) where T
        U, Times, Omega = generate_sine_data(
            batch, steps=steps, dt=T(dt), freq_range=[T(ω0), ω])
        U .+= randn(T, size(U)) * T(noise)
        return U, Times, Omega
    end

    function construct_rodent(setup)
        @unpack batch, xlen, order, dt, noise, dtype = setup
        zlen = order^2 + order*2
        tspan = (0f0, dtype.(xlen*dt))

        init(s...) = randn(dtype, s...)/10000
        act = σ
        encoder  = Chain(
            Dense(xlen, 100, act, initW=init, initb=init),
            Dense(100, zlen, initW=init, initb=init))

        λ2z = -ones(dtype, zlen)
        prior = Gaussian(zeros(dtype, zlen), λ2z)

        σ2z = -ones(dtype, zlen) ./ 100
        enc_dist = SharedVarCGaussian{dtype}(zlen, xlen, encoder, σ2z)

        dec = make_ode_decoder(xlen, tspan, order)
        μx(z) = dec(z)[1,:,:]
        σ2x = -ones(dtype, 1) ./ 10
        dec_dist = SharedVarCGaussian{dtype}(xlen, zlen, μx, σ2x)

        Rodent(prior, enc_dist, dec_dist)
    end

    @info "Testing Rodent"

    xlen = 20
    order = 2
    batch = 20
    dt = 0.3
    dtype = Float32
    noise = 0.01f0
    setup = @dict xlen order batch dt dtype noise

    rodent = construct_rodent(setup) # |> gpu
    test_data = generate(0.5, batch, dt=dt, steps=xlen)[1] # |> gpu
    #test_data = test_data[:,1]

    ls = elbo(rodent, test_data)
    ps = params(rodent)
    @test length(ps) > 0
    @test isa(ls, dtype)

    function loss(x)
        N = size(x, 2)
        λz = variance(rodent.prior)
        (μz, σz) = mean_var(rodent.encoder, x)
        z = rand(rodent.encoder, x)

        llh = -mean(loglikelihood(rodent.decoder, x, z))
        KLz = (sum(log.(λz ./ σz)) + sum(σz ./ λz) + sum(μz.^2 ./ λz)) / N

        σe = variance(rodent.decoder, z)[1]
        llh + KLz + log(σe)*rodent.decoder.zlength
    end

    cb = Flux.throttle(()->(
      @debug "$(loss(test_data)) noise: $(sqrt.(variance(rodent.decoder)))"), 1)

    η = 0.001
    ω = 0.5
    opt = RMSProp(η)
    data = [(test_data,) for _ in 1:200]
    Flux.train!(loss, ps, data, opt, cb=cb)
    reconstruct(m, x) = mean(m.decoder, mean(m.encoder, x))
    rec_err = mean((test_data .- reconstruct(rodent, test_data)).^2)
    @debug "Rec. Error: $rec_err"
    @assert rec_err < 0.05

    Random.seed!()  # reset the seed
end
