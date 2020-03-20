@testset "src/models/rodent.jl" begin

    Random.seed!(1)

    """
    Generate pairs of sine waves and their generating frequency
    """
    function generate_sines(ω, batch; ω0=0.5, noise=0.01, dt=0.1, steps=20, T=Float32)
        U = Array{T,2}(undef, steps, batch)
        t = T.(range(0, length=steps, step=dt))
        for ii in 1:batch
            ω = T(ω0 + rand() * (ω - ω0))
            ϕ = rand(T) * 2pi
            u = sin.(ω * t .+ ϕ)
            U[:,ii] .= T.(u)
        end
        return U
    end
    
    tlen  = 20
    slen  = 2
    batch = 10
    dt    = 0.3f0
    zlen  = 8
    dtype = Float32
    noise = 0.01f0
    H(sol) = hcat(sol.u...)[1,:]
    init(s...) = randn(dtype, s...)/10000
    enc  = Chain(
        Dense(tlen, 100, σ, initW=init, initb=init),
        Dense(100, zlen, initW=init, initb=init))

    rodent = Rodent(slen, tlen, dt, enc, observe=H, olen=tlen)
    test_data = generate_sines(0.5, batch, dt=dt, steps=tlen) # |> gpu

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

    @test occursin("Rodent", sprint(show, rodent))

    Random.seed!()  # reset the seed
end
