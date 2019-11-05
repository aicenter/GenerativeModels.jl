@testset "utils/nalu_ode_decoder.jl" begin

    slength = 2
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
        ndec = NaluODEDecoder(slength, xlength, tspan)

        z = ones(slength^2*3 + slength)
        x = ndec(z)
        @test length(x) == xlength*slength

        loss(z) = sum(ndec(z))
        gs = Zygote.gradient(loss, z)

        @test length(gs) == 1
        @test length(gs[1]) == (slength^2*3 + slength)
    end

end
