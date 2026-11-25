"""
Shared model and exact-arithmetic utilities for the direct CPC benchmark.

The physical generator preserves its original arithmetic and draw order: even unused
legacy gain draws advance the RNG and therefore affect subsequent plants and
line conductances. Changing those draws would invalidate published witnesses.
The active controller search ignores the generated legacy feedback gains.
"""
module BuckConverterCPC

using JuMP
using Clarabel
using MosekTools
using LinearAlgebra
using Random

const MOI = JuMP.MOI
const Exact = Rational{BigInt}

"Construct an exact rational without passing through floating-point arithmetic."
rat(n::Integer, d::Integer = 1) = BigInt(n) // BigInt(d)
rows(matrix) = [collect(row) for row in eachrow(matrix)]
fraction(value::Exact) = string(numerator(value), "/", denominator(value))

"""
    generate_instance(N=2, seed=20261122)

Generate the exact physical plant and signal selectors used by both methods.
Parameter, edge, and variable order are part of the saved-witness contract.
The returned legacy F, A and lambda metadata are retained for compatibility;
the direct search rebuilds its own candidate dynamics and free SOS multipliers.
"""
function generate_instance(N::Int = 2, seed::Int = 20261122)
    N >= 2 || throw(ArgumentError("N must be at least 2"))
    rng = MersenneTwister(seed)
    dt, current_scale, voltage_scale = rat(1, 20_000), rat(1, 10), rat(1)
    voltage_reference = rat(24)
    radius, epsilon_local, alpha = rat(7, 20), rat(1, 1000), rat(1, 100)
    lambda = epsilon_local / radius ^ 2
    modules = NamedTuple[]
    for n in 1:N
        L = rat(rand(rng, 80:120), 100_000) # H
        C = rat(rand(rng, 80:120), 1_000_000) # F
        r = rat(rand(rng, 8:16), 100) # Ohm
        G = rat(rand(rng, 16:24), 100) # Siemens
        source = rat(rand(rng, 46:54)) # V
        ki = rat(rand(rng, 350:450), 100) # Ohm (supplied feedback gain)
        kv = rat(rand(rng, 85:95), 100)
        current_reference = G * voltage_reference
        duty_reference = (voltage_reference + r * current_reference) / source
        push!(modules, (; n, L, C, r, G, source, ki, kv, current_reference, duty_reference))
    end

    # Simple undirected cycle: N=2 has ONE physical line, not two copies of it.
    edges = sort!(collect(Set(minmax(n, mod1(n + 1, N)) for n in 1:N)))
    conductance = zeros(Exact, N, N)
    for (i, j) in edges
        conductance[i, j] = conductance[j, i] = rat(rand(rng, 5:15), 1000)
    end
    M = zeros(Exact, N, 2N)
    F, T = Matrix{Exact}[], Matrix{Exact}[]
    for (n, p) in enumerate(modules)
        degree = sum(conductance[n, :])
        degree > 0 || error("isolated module")
        # w_n is a conductance-weighted average of neighbors' normalized
        # voltage errors, so W_n=[-1,1]. The full shifted output is z_n in R^2.
        for j in 1:N
            M[n, 2j] = conductance[n, j] / degree
        end
        A = Exact[
            1 - dt * (p.r + p.ki) / p.L dt * (p.kv - 1) * voltage_scale / (p.L * current_scale);
            dt * current_scale / (p.C * voltage_scale) 1 - dt * (p.G + degree) / p.C
        ]
        b = Exact[0, dt * degree / p.C]
        push!(F, hcat(b, A)) # local signal order: [w_n,z_{n,1},z_{n,2}]
        Tn = zeros(Exact, 3, 2N)
        Tn[1, :] = M[n, :]
        Tn[2, 2n - 1], Tn[3, 2n] = 1, 1
        push!(T, Tn)
    end
    A = reduce(vcat, (F[n] * T[n] for n in 1:N))
    initial_lower = repeat(Exact[rat(-9, 10), rat(3, 5)], N)
    initial_upper = repeat(Exact[rat(-7, 10), rat(4, 5)], N)
    return (;
        N,
        seed,
        dt,
        current_scale,
        voltage_scale,
        voltage_reference,
        radius,
        epsilon_local,
        alpha,
        lambda,
        modules,
        conductance,
        M,
        F,
        T,
        A,
        initial_lower,
        initial_upper,
    )
end

function make_model(solver::Symbol; tolerance = 1.0e-9, solver_chordal = true, verbose = false)
    if solver == :clarabel
        model = Model(Clarabel.Optimizer)
        set_optimizer_attribute(model, "tol_feas", tolerance)
        set_optimizer_attribute(model, "tol_gap_abs", tolerance)
        set_optimizer_attribute(model, "tol_gap_rel", tolerance)
        set_optimizer_attribute(model, "max_iter", 250)
        set_optimizer_attribute(model, "time_limit", 120.0)
        set_optimizer_attribute(model, "max_threads", 1)
        set_optimizer_attribute(model, "chordal_decomposition_enable", solver_chordal)
    elseif solver == :mosek
        model = Model(MosekTools.Optimizer)
        set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)
        set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", 120.0)
        for key in ("MSK_DPAR_INTPNT_CO_TOL_PFEAS", "MSK_DPAR_INTPNT_CO_TOL_DFEAS")
            set_optimizer_attribute(model, key, tolerance)
        end
    else
        throw(ArgumentError("solver must be clarabel or mosek"))
    end
    verbose ? unset_silent(model) : set_silent(model)
    return model
end

"Chordal extension obtained by elimination, followed by maximal-clique extraction."
function composition_cliques(instance)
    m = 2instance.N
    adjacency = [Set{Int}() for _ in 1:m]
    for Tn in instance.T
        active = findall(j -> any(!iszero, Tn[:, j]), 1:m)
        for i in active, j in active
            i != j && push!(adjacency[i], j)
        end
    end
    remaining = trues(m)
    candidates = Vector{Int}[]
    # Current coordinates have three voltage neighbors. Eliminating currents
    # first, then voltages around the ring, gives cliques of order at most five.
    order = vcat(collect(1:2:m), collect(2:2:m))
    for v in order
        neighbors = sort!([j for j in adjacency[v] if remaining[j]])
        push!(candidates, sort!(vcat(v, neighbors)))
        for i in neighbors, j in neighbors
            i != j && push!(adjacency[i], j)
        end
        remaining[v] = false
    end
    cliques = [
        C for (i, C) in enumerate(candidates) if !any(
            j != i && length(D) > length(C) && all(v -> v in D, C) for
            (j, D) in enumerate(candidates)
        )
    ]
    return (; cliques, order)
end

function exact_psd(A::Matrix{Exact})
    A == A' || return (; pass = false, reason = "not exactly symmetric", pivots = Exact[])
    n = size(A, 1)
    L = Matrix{Exact}(I, n, n)
    pivots = Exact[]
    for k in 1:n
        d = A[k, k] - sum((L[k, j] ^ 2 * pivots[j] for j in 1:(k - 1)); init = rat(0))
        push!(pivots, d)
        d < 0 && return (; pass = false, reason = "negative rational pivot $(k)", pivots)
        for i in (k + 1):n
            residual =
                A[i, k] - sum((L[i, j] * L[k, j] * pivots[j] for j in 1:(k - 1)); init = rat(0))
            if iszero(d)
                iszero(residual) ||
                    return (; pass = false, reason = "nonzero column at zero pivot $(k)", pivots)
            else
                L[i, k] = residual / d
            end
        end
    end
    return (; pass = true, reason = "exact rational LDL PSD", pivots)
end

end # module BuckConverterCPC
