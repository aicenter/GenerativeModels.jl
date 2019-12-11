@testset "utils/utils.jl" begin

    # softplus
    s = (3,4,2)
    x = GenerativeModels.softplus_safe.(randn(Float32, s))
    @test size(x) == s
    @test all(x .> 0)
    @test eltype(x) == Float32
    @test eltype(softplus_safe.(x, Float64)) == Float64

    # stack layers
    s = (8, 6)
    x = randn(Float32, s)
    zdim = 2
    lsize = (s[1], 4, zdim)
    m = GenerativeModels.stack_layers(lsize, relu)
    y = m(x)

    @test length(m.layers) == 2
    @test size(y) == (zdim, s[2])
    @test eltype(y) == Float32
    @test size(m.layers[1].W) == (lsize[2], lsize[1])

    m = GenerativeModels.stack_layers(lsize, relu, last = sigmoid)
    y = m(x)
    @test size(y) == (2, 6)

    # convolutional constructors
    s = (8, 6, 2, 6)
    x = randn(Float32, s)
    xsize = s[1:3]
    ldim = 2
    kernelsizes = (3, 5)
    nchannels = (4, 8)
    scalings = (2, 1)
    densedims = [128]
    
    # encoder
    encoder = GenerativeModels.conv_encoder(xsize, ldim, kernelsizes, nchannels, scalings)
    y = encoder(x)
    @test length(encoder.layers) == 2*length(kernelsizes) + 1 + 1 # 2*(conv + maxpool) + reshape + dense
    @test size(y) == (ldim, s[end])
    @test eltype(y) == Float32

    encoder = GenerativeModels.conv_encoder(xsize, ldim, kernelsizes, nchannels, scalings; 
        densedims = densedims)
    @test length(encoder.layers) == 2*length(kernelsizes) + 1 + 2 # 2*(conv + maxpool) + reshape + 2*dense
    
    # decoder
    decoder = GenerativeModels.conv_decoder(xsize, ldim, reverse(kernelsizes), 
        reverse(nchannels), reverse(scalings))
    z = decoder(y)
    @test length(decoder.layers) == 1 + 1 + 2*length(kernelsizes) # dense + reshape + 2*(conv + maxpool)
    @test size(z) == size(x)
    @test eltype(z) == Float32

    decoder = GenerativeModels.conv_decoder(xsize, ldim, reverse(kernelsizes), 
        reverse(nchannels), reverse(scalings); densedims = densedims)
    @test length(decoder.layers) == 2 + 1 + 2*length(kernelsizes) # dense + reshape + 2*(conv + maxpool)
    
    # also test trainability
    enc_params = get_params(encoder)
    dec_params = get_params(decoder)    
    loss(x) = Flux.mse(decoder(encoder(x)), x)
    opt = ADAM()
    data = (x,)
    GenerativeModels.update_params!(encoder, data, loss, opt)
    @test all(param_change(enc_params, encoder))
    GenerativeModels.update_params!(decoder, data, loss, opt)
    @test all(param_change(dec_params, decoder))
    
    @testset "destructure/restructure" begin
        m = Dense(3,3)
        σ = NoGradArray(ones(3))
        g = CMeanGaussian{Float32,DiagVar}(m, σ)

        z = GenerativeModels.destructure(g)
        @test z isa Vector
        @test length(z) == 3*3 + 3 + 3

        _z = rand(3*3 + 3 + 3)
        _g = GenerativeModels.restructure(g, _z)

        @test _g.mapping.W[1] != g.mapping.W[1]
        @test _g.mapping.W[1] == _z[1]
    end
end
