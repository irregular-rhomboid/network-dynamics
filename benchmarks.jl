using LinearAlgebra
using Graphs
#using DifferentialEquations
using ModelingToolkit
using NetworkDynamics
using Random
using Statistics
using BenchmarkTools
using StatsPlots

Random.seed!(2023)

# Handy functions for the base Brusselator equations
brusselator_x(x,y,a,b) = a + x^2 * y - b*x -x
brusselator_y(x,y,a,b) = b*x - x^2 * y

# in-place base Brusselator dynamics
function brusselator!(dx, x, p, t)
    a, b = p
    dx[1] = brusselator_x(x[1], x[2], a, b)
    dx[2] = brusselator_y(x[1], x[2], a, b)
end

# Reaction-Diffusion Brusselator dynamics over a network
function brusselator_rd!(du, u, p, t)
    p_b, L, D = p
    @views for i in axes(du, 1)
        brusselator!(du[i,:], u[i,:], p_b, t)
    end
    @views begin
        mul!(du[:,1], L, u[:,1], -D[1], 1.0) # du[:,1] .-= D[1]*L*u[:,1]
        mul!(du[:,2], L, u[:,2], -D[2], 1.0)
    end
end

# Construct Reaction-Diffusion Brusselator system for a generic network with N nodes. Return in-place ODE function
function brusselator_rd_mt(N)
    @parameters t a b L[1:N,1:N] D_u D_v
    @variables u(t)[1:N] v(t)[1:N]
    D = Differential(t)

    dudt = brusselator_x.(u,v,a,b) - D_u * (L * u)
    dvdt = brusselator_y.(u,v,a,b) - D_v * (L * v)

    eqs = [
        D.(u) ~ dudt;
        D.(v) ~ dvdt
    ]

    @named sys_rd = ODESystem(eqs)
    return ODEFunction{true}(sys_rd)
end

# Edge dynamics for NetworkDynamics
function rd_edge!(e, v_s, v_d, p, t)
    e .= v_s .- v_d
    nothing
end


# Vertex Dynamics
function rd_vertex!(dv, v, edges, p, t)
    p_b, D = p
    brusselator!(dv, v, p_b, t)
    for e in edges
        dv .+= D .* e
    end
    nothing
end

nd_rd_vertex = ODEVertex(; f=rd_vertex!, dim=2)
nd_rd_edge = StaticEdge(; f=rd_edge!, dim=2, coupling=:antisymmetric)

# Construct an in-place function for the network g
function brusselator_rd_nd(g)
    return network_dynamics(nd_rd_vertex, nd_rd_edge, g)
end

# Edge dynamics for NetworkDynamics
Base.@propagate_inbounds function rd_edge_ib!(e, v_s, v_d, p, t)
    e .= v_s .- v_d
    nothing
end


# Vertex Dynamics
Base.@propagate_inbounds function rd_vertex_ib!(dv, v, edges, p, t)
    p_b, D = p
    brusselator!(dv, v, p_b, t)
    for e in edges
        dv .+= D .* e
    end
    nothing
end

nd_rd_vertex_ib = ODEVertex(; f=rd_vertex_ib!, dim=2)
nd_rd_edge_ib = StaticEdge(; f=rd_edge_ib!, dim=2, coupling=:antisymmetric)

# Construct an in-place function for the network g
function brusselator_rd_nd_ib(g)
    return network_dynamics(nd_rd_vertex_ib, nd_rd_edge_ib, g)
end

function benchmark(n, seed=2023)
    rng = Xoshiro(seed)
    g = watts_strogatz(n, 4, 0.3, rng=rng)
    nd_rd! = brusselator_rd_nd(g)
    nd_rd_ib! = brusselator_rd_nd_ib(g)
    # define params
    L_ = float.(laplacian_matrix(g)) # this is a minor optimization
    p_sparse = ([1.0,3.0], L_, [0.5,0.1])
    p_dense = ([1.0,3.0], Array(L_), [0.5,0.1])
    p_nd = (([1.0,3.0], [0.5,0.1]), nothing)
    # init vars 
    u = rand(rng,n,2)
    du = similar(u)
    u_vec = vec(u)
    du_vec = vec(du)
    # run once to compile
    @time brusselator_rd!(du, u, p_sparse, 0.0)
    @time brusselator_rd!(du, u, p_dense, 0.0)
    @time nd_rd!(du_vec, u_vec, p_nd, 0.0)
    @time nd_rd_ib!(du_vec, u_vec, p_nd, 0.0)
    
    # run the benchmarks
    b_sparse = @benchmark brusselator_rd!($du, $u, $p_sparse, 0.0)
    b_dense = @benchmark brusselator_rd!($du, $u, $p_dense, 0.0)
    b_nd = @benchmark $(nd_rd!)($du_vec, $u_vec, $p_nd, 0.0)
    b_nd_ib = @benchmark $(nd_rd_ib!)($du_vec, $u_vec, $p_nd, 0.0)

    if n <= 100
        mt_rd! = brusselator_rd_mt(n)
        @parameters t a b L[1:n,1:n] D_u D_v
        p_mt = [
            a => 1.0,
            b => 3.0,
            collect(L .=> L_)...,
            #L => L_,
            D_u => 0.5,
            D_v => 0.1
        ]
        p_vec = [q.second for q in p_mt]
        @time mt_rd!(du_vec, u_vec, p_vec, 0.0)
        b_mt = @benchmark $(mt_rd!)($du_vec, $u_vec, $p_vec, 0.0)
    else
        b_mt = nothing
    end

    return b_sparse, b_dense, b_nd, b_nd_ib, b_mt
end

begin
    ns = [5,10,20,30,50,100,200, 500, 1000]
    quantiles = zeros(3,5,length(ns))
    for j in eachindex(ns)
        println("n = $(ns[j])")
        btimes = benchmark(ns[j])
        for i in eachindex(btimes)
            if !isnothing(btimes[i])
                quantiles[:,i,j] .= quantile(btimes[i].times, [0.25,0.5,0.75])
            end
        end
    end
end


begin
    labels = [
        "Hand-Crafted (Sparse Matmul)",
        "Hand-Crafted (Dense Matmul)",
        "NetworkDynamics",
        "NetworkDynamics (@propagate_inbounds)",
        "ModelingToolkit",
    ]
    p = plot(
        title="Median time for one ODE function evaluation",
        ylabel="Time (ns)",
        xlabel="#nodes",
        yscale= :log10,
        xscale= :log10,
        yticks = 10 .^(1:8),
        legend=:topleft
        )
    for i in 1:4
        plot!(p, ns, quantiles[2,i,:], 
        #yerror=(quantiles[1,i,:], quantiles[3,i,:]),
        label = labels[i],
        marker = :cirle,
        markersize = 3
        )
    end
    plot!(p, ns[1:5], quantiles[2,5,1:5], 
    label = labels[5],
    markersize = 3,
    marker = :circle,
    linestyle = :dash
    )
    p
end