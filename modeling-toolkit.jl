using ModelingToolkit
using Graphs
using Symbolics: scalarize

N = 5
g = watts_strogatz(N,4,0.3)
L = laplacian_matrix(g)

@parameters D t
@variables u(t)[1:N]
Dt = Differential(t)

eqs = [
    Dt.(u) ~ -D * L * u
]

scalarize(eqs)

@named sys = ODESystem(eqs)