@testset "utils/nalu_ode_decoder.jl" begin

    order = 2
    xlength = 10
    tspan = (0f0, 1f0)

    @testset "NAC" begin
        nac = NAC(xlength, xlength-1)
        z = rand(xlength)
        @test length(nac(z)) == xlength-1
    end

    @testset "NALU" begin
        nalu = NALU(xlength, xlength-1)
        z = rand(xlength)
        @test length(nalu(z)) == xlength-1
    end

    @testset "NaluODEDecoder" begin
        ndec = NaluODEDecoder(order, xlength, tspan)

        z = ones(order^2*3 + order*2)
        x = ndec(z)
        @test length(x) == xlength*order

        loss(z) = sum(ndec(z))
        gs = Zygote.gradient(loss, z)

        @test length(gs) == 1
        @test length(gs[1]) == (order^2*3 + order*2)
    end

end
