@testset "src/models/rodent.jl" begin
    
    @info "Testing Rodent"

    tspan = (0f0, 1f0)
    xlen = 30
    order = 2
    batch = 20

    rodent = Rodent(xlen, tspan, order)
    test_data = randn(Float32, xlen, batch)

    loss = elbo(rodent, test_data)
    ps = params(rodent)
    @test length(ps) > 0
    @test isa(loss, Tracker.TrackedReal{Float32})
    
end
