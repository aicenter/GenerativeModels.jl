@testset "utils/nogradarray.jl" begin

    @testset "Params on CPU/GPU" begin
        x = NoGradArray(ones(3))
        y = ones(3)

        p = Gaussian(x, y)
        @test length(params(p)) == 1

        if Flux.use_cuda[]
            g = gpu(p)
            @test length(params(g)) == 1
            @test rand(g) isa CuArray

            c = cpu(g)
            @test length(params(c)) == 1
            @test rand(c) isa Array
        end
    end

    @testset "Gradients" begin
        enc = CMeanVarGaussian{Float32,DiagVar}(Dense(5,4))
        dec = CMeanVarGaussian{Float32,ScalarVar}(Dense(2,6))
        pri = Gaussian(NoGradArray(ones(Float32, 2)), ones(Float32, 2))
        
        vae = VAE(pri, enc, dec)
        x = ones(5)
        
        vae = gpu(vae)
        x = gpu(x)
        
        loss() = elbo(vae, x)
        ps = params(vae)
        @test length(ps) == 5
        
        gs = Flux.gradient(loss, ps)
        for p in ps
            @test haskey(gs, p)
        end
    end

end
