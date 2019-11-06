export NALU, NAC, FauxNALU
export NaluODEDecoder

using Flux: glorot_uniform


#################### NEURAL ACCUMULATOR ########################################

"""
    NAC

Neural Accumulator. Special case of affine layer in which the parameters
are encouraged to be close to {-1, 0, 1}.
"""
struct NAC
    W::AbstractMatrix
    M::AbstractMatrix
end

function NAC(in::Int, out::Int;
             initW=glorot_uniform, initM=glorot_uniform)
    # TODO: use kaiming uniform here???
    # TODO: include bias???
    W = initW(out, in)
    M = initM(out, in)
    NAC(W, M)
end

function (nac::NAC)(x)
    _W = tanh.(nac.W) .* σ.(nac.M)
    _W*x
end

function Base.show(io::IO, l::NAC)
    in = size(l.W, 2)
    out = size(l.W, 1)
    print(io, "NAC(in=$in, out=$out)")
end

Flux.@functor NAC

#################### NEURAL ARITHMETIC LOGIC UNIT ##############################

"""
    NALU

Neural Arithmetic Logic Unit. Layer that is capable of learing mulitplication,
division, power functions, addition, and subtraction.
"""
struct NALU
    nac::NAC
    G::AbstractMatrix
    ϵ::Real
end

NALU(nac::NAC, G::AbstractMatrix) = NALU(nac, G, 1e-10)

function NALU(in::Int, out::Int; initNAC=glorot_uniform, initG=glorot_uniform)
    nac = NAC(in, out, initW=initNAC, initM=initNAC)
    G = initG(out, in)
    NALU(nac, G)
end

function (nalu::NALU)(x)
    nac, G, ϵ = nalu.nac, nalu.G, nalu.ϵ
    a = nac(x)
    log_inp = log.(abs.(x) .+ ϵ)
    m = exp.(nac(log_inp))
    g = σ.(G*x)
    g .* a .+ (1.0 .- g) .* m
end

function Base.show(io::IO, l::NALU)
    in = size(l.G, 2)
    out = size(l.G, 1)
    print(io, "NALU(in=$in, out=$out)")
end

Flux.@functor NALU

#################### NEURAL ARITHMETIC LOGIC UNIT WITH BIAS ####################

struct FauxNALU
    nac::NAC
    G::AbstractMatrix
    b::AbstractVector
    ϵ::Real
end

FauxNALU(nac::NAC, G::AbstractMatrix, b::AbstractVector) = FauxNALU(nac, G, b, 1e-10)

function FauxNALU(in::Int, out::Int; initNAC=glorot_uniform, initG=glorot_uniform)
    nac = NAC(in, out, initW=initNAC, initM=initNAC)
    G = initG(out, in)
    b = initG(out)
    FauxNALU(nac, G, b)
end

function (nalu::FauxNALU)(x)
    nac, G, ϵ = nalu.nac, nalu.G, nalu.ϵ
    a = nac(x) + nalu.b
    log_inp = log.(abs.(x) .+ ϵ)
    m = exp.(nac(log_inp))
    g = σ.(G*x)
    g .* a .+ (1.0 .- g) .* m
end

function Base.show(io::IO, l::FauxNALU)
    in = size(l.G, 2)
    out = size(l.G, 1)
    print(io, "FauxNALU(in=$in, out=$out)")
end

Flux.@functor FauxNALU



####################  NALU ODE Decoder  ########################################

"""
    NaluODEDecoder(slength::Int, tlength::Int, timesteps::Vector, nalu::NALU)

Decoder with a NALU as a neural ODE.
Calling `decoder(nalu, u0)` will return a vector of length slength*tlength
which is the reshaped ODE solution (`reshape(hcat(sol.u...), :)`).

# Arguments
* `slength::Int`: length of the internal state of the ODE
* `tlength::Int`: length of the decoded time series
* `timesteps::Vector`: timesteps of the decoded time series
* `nalu::NALU`: the NALU that is used as neural ODE.
"""
struct NaluODEDecoder
    slength::Int
    tlength::Int
    timesteps::Vector
    nalu::NALU
end

function NaluODEDecoder(slength::Int, tlength::Int, tspan::Tuple)
    timesteps = range(tspan[1], stop=tspan[2], length=tlength)
    nalu = NALU(slength, slength)
    NaluODEDecoder(slength, tlength, timesteps, nalu)
end

nalu_ode_params_length(slength::Int) = (slength^2)*3 + slength

function nalu_latent_split(z::AbstractVector, slength::Int)
    @assert length(z) == nalu_ode_params_length(slength)
    o2 = slength^2
    W = reshape(z[1:o2], slength, slength)
    M = reshape(z[o2+1:2o2], slength, slength)
    G = reshape(z[2o2+1:3o2], slength, slength)
    u0 = z[3o2+1:3o2+slength]
    (W,M,G), u0
end

function (dec::NaluODEDecoder)(nalu::NALU, u0::AbstractVector, ps)
    tspan = (dec.timesteps[1], dec.timesteps[end])
    dudt_(u::AbstractVector, ps, t) = nalu(u)
    prob = ODEProblem(dudt_, u0, tspan, ps)
    sol = solve(prob, Tsit5(), saveat=dec.timesteps)
    reshape(hcat(sol.u...), :)
end

function (dec::NaluODEDecoder)(z::AbstractVector)
    (ps, u0) = nalu_latent_split(z, dec.slength)
    (W, M, G) = ps
    nac = NAC(W, M)
    nalu = NALU(nac, G, 1e-10)
    dec(nalu, u0, ps)
end

# Use loop to get batched reconstructions so that jacobian and @adjoint work...
(dec::NaluODEDecoder)(Z::AbstractMatrix) = hcat([dec(Z[:,ii]) for ii in 1:size(Z,2)]...)

ddec(dec::NaluODEDecoder, z::AbstractVector) = ForwardDiff.jacobian(dec, z)

@adjoint function (dec::NaluODEDecoder)(z::AbstractVector)
    (dec(z), Δ -> (J=Δ'*ddec(dec, z); (nothing,J')))
end
