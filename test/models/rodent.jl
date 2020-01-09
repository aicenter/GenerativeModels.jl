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

    tlen = 20
    slen = 2
    batch = 10
    dt = 0.3f0
    zlen = 8
    dtype = Float32
    noise = 0.01f0
    H(sol) = hcat(sol.u...)[1,:]
    init(s...) = randn(dtype, s...)/10000
    enc  = Chain(
        Dense(tlen, 100, σ, initW=init, initb=init),
        Dense(100, zlen, initW=init, initb=init))

    rodent = Rodent(slen, tlen, dt, enc, observe=H)
    test_data = generate(0.5, batch, dt=dt, steps=tlen)[1] # |> gpu

    ls = -elbo(rodent, test_data)
    ps = params(rodent)
    @test length(ps) > 0
    @test isa(ls, dtype)

    loss(x) = -elbo(rodent, x)

    cb = Flux.throttle(()->(
      @debug "$(loss(test_data)) noise: $(sqrt.(variance(rodent.decoder)))"), 0.1)

    η = 0.001
    ω = 0.5
    opt = RMSProp(η)
    data = [(test_data,) for _ in 1:400]
    Flux.train!(loss, ps, data, opt, cb=cb)
    reconstruct(m, x) = mean(m.decoder, mean(m.encoder, x))
    rec_err = mean((test_data .- reconstruct(rodent, test_data)).^2)
    @debug "Rec. Error: $rec_err"
    @test rec_err < 0.05

    # p = plot(test_data, color="gray")
    # plot!(p, reconstruct(rodent, test_data), ls=:dash)
    # display(p)

    Random.seed!()  # reset the seed
end
