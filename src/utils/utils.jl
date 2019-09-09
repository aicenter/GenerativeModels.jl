# for over/underflow in logs
"""
	softplus_safe(x,T=Float32)

A softplus with small additive constant for safe operations.
"""
softplus_safe(x,T=Float32) = softplus(x) .+ T(1e-6)

