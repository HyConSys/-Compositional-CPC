#!/usr/bin/env julia
#=
Direct polynomial B,V/controller search, degree two, shared two-priority DPA.
Standalone Julia interface:
  julia --project=. Synthesis_Direct_CPC.jl \
    [N=2] [seed=20261122] [candidate_budget=7] [normalized|physical]
  julia --project=. Synthesis_Direct_CPC.jl verify results.toml

Controller candidates are computed by admissibility/domain-invariance LPs;
acceptance requires a genuine B,V SOS feasibility solve and exact proof.
No Lyapunov decay, positive-definite certificate coefficient matrix, B=0,
Upsilon=0 or fixed relation between ranks is imposed. Each polynomial at each
automaton state has independent free coefficients. This incomplete outer
candidate search is not CEGIS and is not a global nonconvex optimizer.

The default CLI uses dimensionless closed-loop controller coordinates
(A11,A12,drift), mapped exactly back to the physical offset and gains.
The physical-coordinate candidate mode is retained for reproduction. Both
parameterizations have the same admissible controller polytope. Certificate
and SOS variable units are unchanged, as supported by the normalization study.

Numerical conditioning uses coordinates centered at the COMPUTED candidate's
fixed point, with all domains and target labels translated consistently. This
does not require stability of that point. Certificates are also exported in
the original physical coordinates. At an interior accepting stationary point,
the accepting self-transition residuals necessarily vanish. Their SOS faces
are reduced accordingly; this is an implied equality, not a stability test.

All affine SOS coefficient identities are repaired exactly over rationals;
every repaired Gram matrix, multiplier and original aggregate is then checked.
Raw solver coefficients and the correction magnitude are retained.
=#
include(joinpath(@__DIR__, "Synthesis_Buck_CPC.jl"))
using .BuckConverterCPC
using JuMP, LinearAlgebra, TOML, Dates, SHA
const CPC = BuckConverterCPC
const Rat = CPC.Exact

"""
An affine expression with exact rational coefficients and stable variable IDs.
JuMP receives floating-point coefficients; this parallel representation keeps
the original identities available for rational reconstruction and verification.
"""
struct ExactAffine
    constant::Rat
    coefficients::Dict{Int, Rat}
end
ExactAffine(x::Real) = ExactAffine(Rat(x), Dict{Int, Rat}())
Base.zero(::Type{ExactAffine}) = ExactAffine(0)
Base.zero(::ExactAffine) = ExactAffine(0)
Base.iszero(x::ExactAffine) = iszero(x.constant) && isempty(x.coefficients)
function Base.:+(x::ExactAffine, y::ExactAffine)
    a = copy(x.coefficients)
    for (i, v) in y.coefficients
        a[i] = get(a, i, CPC.rat(0)) + v
        iszero(a[i]) && delete!(a, i)
    end
    ExactAffine(x.constant + y.constant, a)
end
Base.:-(x::ExactAffine) = ExactAffine(-x.constant, Dict(i => -v for (i, v) in x.coefficients))
Base.:-(x::ExactAffine, y::ExactAffine) = x + (-y)
Base.:+(x::ExactAffine, y::Real) = x + ExactAffine(y)
Base.:+(x::Real, y::ExactAffine) = ExactAffine(x) + y
Base.:-(x::ExactAffine, y::Real) = x - ExactAffine(y)
Base.:-(x::Real, y::ExactAffine) = ExactAffine(x) - y
Base.:*(s::Real, x::ExactAffine) =
    iszero(s) ? ExactAffine(0) :
    ExactAffine(Rat(s) * x.constant, Dict(i => Rat(s) * v for (i, v) in x.coefficients))
Base.:*(x::ExactAffine, s::Real) = s * x
Base.:/(x::ExactAffine, s::Real) = inv(Rat(s)) * x
evaluate_exact(x::ExactAffine, u) =
    x.constant + sum((v * u[i] for (i, v) in x.coefficients); init = CPC.rat(0))

"The SDP together with the exact algebra needed to verify a saved witness."
mutable struct CertificateProblem
    model::Model
    variables::Vector{VariableRef}
    variable_ids::Dict{VariableRef, Int}
    variable_names::Vector{String}
    nonnegative_ids::Vector{Int}
    coefficient_identities::Vector{ExactAffine}
    psd_blocks::Vector{NamedTuple}
    sos_constraints::Vector{NamedTuple}
    polynomials::Dict{String, Matrix{ExactAffine}}
    supplies::Dict{String, Matrix{ExactAffine}}
end
CertificateProblem() = CertificateProblem(
    CPC.make_model(:mosek),
    VariableRef[],
    Dict{VariableRef, Int}(),
    String[],
    Int[],
    ExactAffine[],
    NamedTuple[],
    NamedTuple[],
    Dict(),
    Dict(),
)
function register_variable!(problem, v)
    i = get!(problem.variable_ids, v) do
        push!(problem.variables, v)
        push!(problem.variable_names, name(v))
        length(problem.variables)
    end
    ExactAffine(CPC.rat(0), Dict(i => CPC.rat(1)))
end
function add_scalar!(problem, name; nonnegative = false)
    v=@variable(problem.model, base_name = name)
    nonnegative && set_lower_bound(v, 0.0)
    l = register_variable!(problem, v)
    nonnegative && push!(problem.nonnegative_ids, only(keys(l.coefficients)))
    l
end
function add_psd_matrix!(problem, k, name)
    Q=@variable(problem.model, [1:k, 1:k], PSD, base_name = name)
    L = [register_variable!(problem, Q[i, j]) for i in 1:k, j in 1:k]
    ids = [only(keys(L[i, j].coefficients)) for i in 1:k, j in 1:k]
    push!(problem.psd_blocks, (; name, ids))
    L
end
function as_jump_expression(problem, l)
    a = AffExpr(Float64(l.constant))
    for (i, v) in l.coefficients
        add_to_expression!(a, Float64(v), problem.variables[i])
    end
    a
end
function add_identity!(problem, l)
    iszero(l) && return
    @constraint(problem.model, as_jump_expression(problem, l) == 0)
    push!(problem.coefficient_identities, l)
end
zero_affine_matrix(k) = [ExactAffine(0) for _ in 1:k, _ in 1:k]
"Use free polynomial coefficients: PSD is imposed on residual Grams, not on B,V."
function add_free_quadratic!(problem, k, name)
    H = zero_affine_matrix(k + 1)
    for j in 1:(k + 1), i in 1:j
        v = add_scalar!(problem, "$(name)_coef_$(i)_$(j)")
        H[i, j] = H[j, i] = i == j ? v : v / 2
    end
    problem.polynomials[name] = H
    H
end
function add_supply_matrix!(problem, k, name)
    H = zero_affine_matrix(k)
    for j in 1:k, i in 1:j
        H[i, j] = H[j, i] = add_scalar!(problem, "$(name)_$(i)_$(j)")
    end
    problem.supplies[name] = H
    H
end
function pullback(H, R)
    n = size(R, 2)
    Q = zero_affine_matrix(n)
    for j in 1:n, i in 1:j
        z = ExactAffine(0)
        for b in axes(R, 1), a in axes(R, 1)
            (iszero(R[a, i]) || iszero(R[b, j]) || iszero(H[a, b])) && continue
            z = z + (R[a, i] * R[b, j]) * H[a, b]
        end
        Q[i, j] = Q[j, i] = z
    end
    Q
end
function evaluate_matrix(H, u)
    [evaluate_exact(H[i, j], u) for i in axes(H, 1), j in axes(H, 2)]
end
function box_polynomial(k, i, lo, hi)
    Q = zeros(Rat, k + 1, k + 1)
    Q[1, 1] = -lo * hi
    Q[1, i + 1] = Q[i + 1, 1] = (lo + hi) / 2
    Q[i + 1, i + 1] = -1
    Q
end
function target_polynomial(k, indices, center, radius; good = true)
    Q = zeros(Rat, k + 1, k + 1)
    Q[1, 1] = radius ^ 2 - sum(abs2, center)
    for (i, e) in zip(indices, center)
        Q[1, i + 1] = Q[i + 1, 1] = -e
        Q[i + 1, i + 1] = -1
    end
    good ? Q : -Q
end
"""
Represent residual = Gram + sum(multiplier * domain polynomial).
At an interior stationary point a zero residual forces its constant Gram row
and positive-domain multipliers to vanish. The reduced face encodes only those
implied equalities; it does not add stability or a positive robustness margin.
"""
function add_sos_constraint!(problem, residual, generators, name; stationary_face = false)
    k = size(residual, 1)
    gram = if stationary_face
        @assert all(G[1, 1] > 0 for G in generators)
        Q = zero_affine_matrix(k)
        Q[2:end, 2:end] = add_psd_matrix!(problem, k - 1, name * "_face_Gram")
        Q
    else
        add_psd_matrix!(problem, k, name * "_Gram")
    end
    lambdas = ExactAffine[]
    remainder = copy(residual)
    if !stationary_face
        for (j, G) in enumerate(generators)
            l = add_scalar!(problem, "$(name)_multiplier_$(j)"; nonnegative = true)
            push!(lambdas, l)
            remainder = [remainder[a, b] - G[a, b] * l for a in 1:k, b in 1:k]
        end
    end
    for j in 1:k, i in 1:j
        add_identity!(problem, remainder[i, j] - gram[i, j])
    end
    push!(problem.sos_constraints, (; name, residual, generators, lambdas, gram, stationary_face))
end

"Controller candidate LP: ONLY input bounds and robust state-box preservation."
function solve_physical_controller_lp(base, attempt)
    model = CPC.make_model(:mosek)
    N = base.N
    a=@variable(model, [1:N], lower_bound = 0, upper_bound = 1, base_name = "duty_offset")
    ki=@variable(model, [1:N], base_name = "ki")
    kv=@variable(model, [1:N], base_name = "kv")
    for (n, converter) in enumerate(base.modules)
        dt, L, C, E, r, G = Float64.((
            base.dt,
            converter.L,
            converter.C,
            converter.source,
            converter.r,
            converter.G,
        ))
        si, sv = Float64.((base.current_scale, base.voltage_scale))
        degree = Float64(sum(base.conductance[n, :]))
        gainabs=@variable(model, [1:2], lower_bound = 0, base_name = "input_abs_$(n)")
        for (j, g) in enumerate((-si * ki[n] / E, sv * kv[n] / E))
            @constraint(model, gainabs[j] >= g)
            @constraint(model, gainabs[j] >= -g)
        end
        @constraint(model, a[n] >= sum(gainabs))
        @constraint(model, a[n] + sum(gainabs) <= 1)
        F = [
            0.0 1 - dt * (r + ki[n]) / L dt * (kv[n] - 1) * sv / (L * si);
            dt * degree / C dt * si / (C * sv) 1 - dt * (G + degree) / C
        ]
        c =
            dt * (
                E * a[n] -
                Float64(converter.r * converter.current_reference + base.voltage_reference)
            ) / (L * si)
        absolute=@variable(model, [1:2, 1:4], lower_bound = 0, base_name = "next_abs_$(n)")
        for i in 1:2, j in 1:4
            z = j == 1 ? (i == 1 ? c : 0.0) : F[i, j - 1]
            @constraint(model, absolute[i, j] >= z)
            @constraint(model, absolute[i, j] >= -z)
        end
        for i in 1:2
            @constraint(model, sum(absolute[i, j] for j in 1:4) <= 1)
        end
    end
    # Different linear objectives explore the admissible controller polytope.
    if attempt == 1
        @objective(model, Min, 0.0)
    else
        coordinates = vcat(a, ki, kv)
        j = mod1((attempt - 2)÷2 + 1, length(coordinates))
        direction = iseven(attempt) ? 1.0 : -1.0
        @objective(model, Min, direction * coordinates[j])
    end
    optimize!(model)
    termination_status(model) in (CPC.MOI.OPTIMAL, CPC.MOI.ALMOST_OPTIMAL) ||
        error("controller LP: $(termination_status(model))")
    (;
        a = Rat.(value.(a)),
        ki = Rat.(value.(ki)),
        kv = Rat.(value.(kv)),
        status = string(termination_status(model)),
    )
end

function solve_normalized_controller_lp(base, attempt)
    model = CPC.make_model(:mosek)
    N = base.N
    t=@variable(model, [1:N], base_name = "closed_loop_A11")
    s=@variable(model, [1:N], base_name = "closed_loop_A12")
    drift=@variable(model, [1:N], base_name = "closed_loop_drift")
    a = Any[]
    ki = Any[]
    kv = Any[]
    for (n, converter) in enumerate(base.modules)
        reference =
            (converter.r * converter.current_reference + base.voltage_reference) / converter.source
        push!(
            a,
            Float64(reference) +
            Float64(converter.L * base.current_scale / (base.dt * converter.source)) * drift[n],
        )
        push!(ki, Float64(converter.L / base.dt) * (1 - t[n]) - Float64(converter.r))
        push!(
            kv,
            1 + Float64(converter.L * base.current_scale / (base.dt * base.voltage_scale)) * s[n],
        )
        @constraint(model, a[n] >= 0)
        @constraint(model, a[n] <= 1)
    end
    for (n, converter) in enumerate(base.modules)
        dt, L, C, E, G =
            Float64.((base.dt, converter.L, converter.C, converter.source, converter.G))
        si, sv = Float64.((base.current_scale, base.voltage_scale))
        degree = Float64(sum(base.conductance[n, :]))
        gainabs=@variable(model, [1:2], lower_bound = 0, base_name = "input_abs_$(n)")
        for (j, g) in enumerate((-si * ki[n] / E, sv * kv[n] / E))
            @constraint(model, gainabs[j] >= g)
            @constraint(model, gainabs[j] >= -g)
        end
        @constraint(model, a[n] >= sum(gainabs))
        @constraint(model, a[n] + sum(gainabs) <= 1)
        F = [0.0 t[n] s[n]; dt * degree / C dt * si / (C * sv) 1 - dt * (G + degree) / C]
        absolute=@variable(model, [1:2, 1:4], lower_bound = 0, base_name = "next_abs_$(n)")
        for i in 1:2, j in 1:4
            z = j == 1 ? (i == 1 ? drift[n] : 0.0) : F[i, j - 1]
            @constraint(model, absolute[i, j] >= z)
            @constraint(model, absolute[i, j] >= -z)
        end
        for i in 1:2
            @constraint(model, sum(absolute[i, j] for j in 1:4) <= 1)
        end
    end
    if attempt == 1
        @objective(model, Min, 0.0)
    else
        physical = vcat(a, ki, kv)
        j = mod1((attempt - 2)÷2 + 1, length(physical))
        direction = iseven(attempt) ? 1.0 : -1.0
        @objective(model, Min, direction * physical[j])
    end
    optimize!(model)
    termination_status(model) in (CPC.MOI.OPTIMAL, CPC.MOI.ALMOST_OPTIMAL) ||
        error("controller LP: $(termination_status(model))")
    av = Rat[]
    kiv = Rat[]
    kvv = Rat[]
    for (n, converter) in enumerate(base.modules)
        reference =
            (converter.r * converter.current_reference + base.voltage_reference) / converter.source
        push!(
            av,
            reference +
            converter.L * base.current_scale / (base.dt * converter.source) * Rat(value(drift[n])),
        )
        push!(kiv, converter.L / base.dt * (1 - Rat(value(t[n]))) - converter.r)
        push!(
            kvv,
            1 +
            converter.L * base.current_scale / (base.dt * base.voltage_scale) * Rat(value(s[n])),
        )
    end
    (; a = av, ki = kiv, kv = kvv, status = string(termination_status(model)))
end

"""
Average admissible support-LP solutions, optionally reweighting one solution.
All support LPs are rebuilt for each attempt to preserve the benchmark's
candidate rule and accounting; their solutions are not cached across attempts.
Only the subsequent CPC oracle can accept this admissible candidate.
"""
function generate_controller_candidate(base, attempt; coordinates = :physical)
    coordinates in (:physical, :normalized) ||
        error("controller coordinates must be physical or normalized")
    # Averaging admissible polytope vertices avoids crossover boundary errors.
    # This is a candidate-generation rule, not a stability/CPC restriction.
    vertices = [
        coordinates == :physical ? solve_physical_controller_lp(base, k) :
        solve_normalized_controller_lp(base, k) for k in 1:(1 + 6base.N)
    ]
    weights = fill(CPC.rat(1), length(vertices))
    attempt > 1 && (weights[mod1(attempt - 1, length(vertices))] += attempt - 1)
    total = sum(weights)
    a = [sum(weights[j] * vertices[j].a[n] for j in eachindex(vertices)) / total for n in 1:base.N]
    ki =
        [sum(weights[j] * vertices[j].ki[n] for j in eachindex(vertices)) / total for n in 1:base.N]
    kv =
        [sum(weights[j] * vertices[j].kv[n] for j in eachindex(vertices)) / total for n in 1:base.N]
    (; a, ki, kv, status = "computed polytope barycenter", LP_count = length(vertices))
end

"""
Reconstruct the exact affine closed loop for a candidate and translate its ports.
The interior fixed point is a conditioning choice, not a stability constraint.
All domain and target polynomials are pulled back through the same translation.
"""
function reconstruct_closed_loop(base, theta)
    N = base.N
    F = Matrix{Rat}[]
    c = Rat[]
    for (n, converter) in enumerate(base.modules)
        dt = base.dt
        degree = sum(base.conductance[n, :])
        f = Rat[
            0 1 - dt * (converter.r + theta.ki[n]) / converter.L dt * (theta.kv[n] - 1) * base.voltage_scale / (converter.L * base.current_scale);
            dt * degree / converter.C dt * base.current_scale / (converter.C * base.voltage_scale) 1 - dt * (converter.G + degree) / converter.C
        ]
        push!(F, f)
        append!(
            c,
            [
                dt * (
                    converter.source * theta.a[n] - converter.r * converter.current_reference -
                    base.voltage_reference
                ) / (converter.L * base.current_scale),
                CPC.rat(0),
            ],
        )
        variation =
            (abs(theta.ki[n]) * base.current_scale + abs(theta.kv[n]) * base.voltage_scale) /
            converter.source
        @assert theta.a[n] - variation >= 0 && theta.a[n] + variation <= 1 "controller input bound"
        for i in 1:2
            @assert sum(abs, f[i, :]) + abs(c[2n - 2 + i]) <= 1 "controller state-box preservation"
        end
    end
    A = reduce(vcat, (F[n] * base.T[n] for n in 1:N))
    equilibrium = (Matrix{Rat}(I, 2N, 2N) - A)\c
    @assert A * equilibrium + c == equilibrium
    @assert all(abs.(equilibrium) .< 1)
    we = base.M * equilibrium
    verify_physical_dynamics(base, theta, A, c)
    (; base, theta, F, c, A, equilibrium, we)
end

"""
Check the affine realization against the original physical equations on an
exact affine basis. This catches a shifted-port or offset mistake that could
otherwise pass verification against the same incorrectly transformed model.
"""
function verify_physical_dynamics(base, theta, A, c)
    N = base.N
    # Independently evaluate the ORIGINAL physical scalar equations on an affine basis.
    for j in 0:2N
        z = zeros(Rat, 2N)
        j > 0 && (z[j] = 1)
        next = zeros(Rat, 2N)
        currents = [
            converter.current_reference + base.current_scale * z[2n - 1] for
            (n, converter) in enumerate(base.modules)
        ]
        voltages = [base.voltage_reference + base.voltage_scale * z[2n] for n in 1:N]
        for (n, converter) in enumerate(base.modules)
            u =
                theta.a[n] +
                (
                    -theta.ki[n] * base.current_scale * z[2n - 1] +
                    theta.kv[n] * base.voltage_scale * z[2n]
                ) / converter.source
            ip =
                currents[n] +
                base.dt / converter.L *
                (converter.source * u - converter.r * currents[n] - voltages[n])
            vp =
                voltages[n] +
                base.dt / converter.C * (
                    currents[n] - converter.G * voltages[n] +
                    sum(base.conductance[n, l] * (voltages[l] - voltages[n]) for l in 1:N)
                )
            next[2n - 1] = (ip - converter.current_reference) / base.current_scale
            next[2n] = (vp - base.voltage_reference) / base.voltage_scale
        end
        @assert next == A * z + c "physical affine dynamics mismatch"
    end
    return true
end

"""
Match the negative aggregate to a sum of clique-supported PSD matrices.
Matching every entry, including fill entries that must cancel, preserves the
original aggregate inequality; checking clique principal submatrices would not.
"""
function add_aggregate_constraint!(problem, modeldata, supplies, name)
    base = modeldata.base
    m = 2base.N
    aggregate = zero_affine_matrix(m)
    for n in 1:base.N
        R = pullback(supplies[n], base.T[n])
        aggregate = [aggregate[i, j] + R[i, j] for i in 1:m, j in 1:m]
    end
    decomposition = CPC.composition_cliques(base)
    sumcones = zero_affine_matrix(m)
    for (j, C) in enumerate(decomposition.cliques)
        Z = add_psd_matrix!(problem, length(C), "$(name)_clique_$(j)")
        for (a, i) in enumerate(C), (b, l) in enumerate(C)
            sumcones[i, l] = sumcones[i, l] + Z[a, b]
        end
    end
    for j in 1:m, i in 1:j
        add_identity!(problem, aggregate[i, j] + sumcones[i, j])
    end
    aggregate
end

function validate_progress_steps(base, allocation = nothing)
    steps = isnothing(allocation) ? fill(base.epsilon_local, base.N) : Rat.(allocation)
    length(steps) == base.N || throw(ArgumentError("one progress step per subsystem is required"))
    all(steps .>= 0) || throw(ArgumentError("local progress steps must be nonnegative"))
    sum(steps) > 0 || throw(ArgumentError("the aggregate odd-priority progress must be positive"))
    steps
end

"Theorem 3 allocation: nonnegative local steps with positive aggregate progress."
function select_progress_allocation(data, policy::Symbol)
    policy == :uniform && return nothing
    policy in (:allocated, :auto) ||
        throw(ArgumentError("progress policy must be uniform, allocated, or auto"))
    distances = [sum(abs2, data.equilibrium[(2n - 1):2n]) for n in 1:data.base.N]
    policy == :auto && all(v -> v < data.base.radius ^ 2, distances) && return nothing
    eligible = findall(v -> v < data.base.radius ^ 2, distances)
    isempty(eligible) && throw(
        ArgumentError("no strictly inside-target equilibrium for an allocated-progress candidate"),
    )
    anchor = eligible[argmin(distances[eligible])]
    allocation = zeros(Rat, data.base.N)
    allocation[anchor] = data.base.N * data.base.epsilon_local
    allocation
end

"""
    build_certificate_problem(data, method; epsilon_allocation=nothing)

Assemble the same ordered degree-two feasibility problem as the published run.
Do not reorder polynomial, supply, Gram, or identity creation: saved witnesses
refer to these exact variable indices, and the solver sees this same order.
"""
function build_certificate_problem(data, method; epsilon_allocation = nothing)
    problem = CertificateProblem()
    steps = validate_progress_steps(data.base, epsilon_allocation)
    if method == :monolithic
        add_monolithic_conditions!(problem, data, steps)
    else
        add_compositional_conditions!(problem, data, steps)
    end
    @objective(problem.model, Min, 0.0)
    return problem
end

"Encode the global good union branchwise without restricting the dense B,V."
function add_monolithic_conditions!(problem, data, steps)
    base = data.base
    N = base.N
    m = 2N
    alpha = base.alpha
    B = [add_free_quadratic!(problem, m, "B_$(q)") for q in ("bad", "good")]
    V = [add_free_quadratic!(problem, m, "V_$(q)") for q in ("bad", "good")]
    boxes = [box_polynomial(m, i, -1 - data.equilibrium[i], 1 - data.equilibrium[i]) for i in 1:m]
    initials = [
        box_polynomial(
            m,
            i,
            base.initial_lower[i] - data.equilibrium[i],
            base.initial_upper[i] - data.equilibrium[i],
        ) for i in 1:m
    ]
    add_sos_constraint!(problem, -B[1], initials, "initial_bad")
    for q in 1:2
        add_sos_constraint!(problem, V[q] + alpha * B[q], boxes, "nonnegative_$(q)")
    end
    transition = zeros(Rat, m + 1, m + 1)
    transition[1, 1] = 1
    transition[2:end, 2:end] = data.A
    good = [
        target_polynomial(m, [2n - 1, 2n], data.equilibrium[(2n - 1):2n], base.radius) for n in 1:N
    ]
    cells = [(; name = "good_branch_$(n)", next = 2, gens = vcat(boxes, [good[n]])) for n in 1:N]
    push!(cells, (; name = "bad", next = 1, gens = vcat(boxes, [-g for g in good])))
    for q in 1:2, cell in cells
        residualB = alpha * B[q] - pullback(B[cell.next], transition)
        residualV = V[q] + alpha * B[q] - pullback(V[cell.next], transition)
        q == 1 && (residualV[1, 1] = residualV[1, 1] - sum(steps))
        face = q == 2 && cell.next == 2 && all(g[1, 1] > 0 for g in cell.gens)
        add_sos_constraint!(
            problem,
            residualB,
            cell.gens,
            "B_$(q)_$(cell.name)";
            stationary_face = face,
        )
        add_sos_constraint!(
            problem,
            residualV,
            cell.gens,
            "V_$(q)_$(cell.name)";
            stationary_face = face,
        )
    end
end

"Create all local unknowns before their constraints to keep witness indexing stable."
function add_compositional_conditions!(problem, data, steps)
    base = data.base
    N = base.N
    alpha = base.alpha
    Bs = [[add_free_quadratic!(problem, 2, "B_$(n)_$(q)") for q in ("bad", "good")] for n in 1:N]
    Vs = [[add_free_quadratic!(problem, 2, "V_$(n)_$(q)") for q in ("bad", "good")] for n in 1:N]
    Gammas = [add_supply_matrix!(problem, 3, "Gamma_$(n)") for n in 1:N]
    Upsilons = [add_supply_matrix!(problem, 3, "Upsilon_$(n)") for n in 1:N]
    for n in 1:N
        add_local_conditions!(problem, data, n, Bs[n], Vs[n], Gammas[n], Upsilons[n], steps[n])
    end
    add_aggregate_constraint!(problem, data, Gammas, "Gamma")
    add_aggregate_constraint!(problem, data, Upsilons, "Upsilon")
end

"""
Enforce every local successor at the same current global automaton state.
A local bad letter has two projected successors; choosing just one would
silently weaken Definition 8 and invalidate the compositional certificate.
"""
function add_local_conditions!(problem, data, n, B, V, Gamma, Upsilon, step)
    base = data.base
    alpha = base.alpha
    H = Rat[1 0 0 0; 0 0 1 0; 0 0 0 1]
    e = data.equilibrium[(2n - 1):2n]
    boxes = [box_polynomial(2, i, -1 - e[i], 1 - e[i]) for i in 1:2]
    initials = [
        box_polynomial(
            2,
            i,
            base.initial_lower[2n - 2 + i] - e[i],
            base.initial_upper[2n - 2 + i] - e[i],
        ) for i in 1:2
    ]
    add_sos_constraint!(problem, -B[1], initials, "initial_$(n)_bad")
    for q in 1:2
        add_sos_constraint!(problem, V[q] + alpha * B[q], boxes, "nonnegative_$(n)_$(q)")
    end
    signalboxes = vcat(
        [box_polynomial(3, 1, -1 - data.we[n], 1 - data.we[n])],
        [box_polynomial(3, i + 1, -1 - e[i], 1 - e[i]) for i in 1:2],
    )
    target = target_polynomial(3, [2, 3], e, base.radius)
    transition = zeros(Rat, 3, 4)
    transition[1, 1] = 1
    transition[2:end, 2:end] = data.F[n]
    G = zero_affine_matrix(4)
    G[2:end, 2:end] = Gamma
    U = zero_affine_matrix(4)
    U[2:end, 2:end] = Upsilon
    for q in 1:2, good in (true, false), qn in (good ? (2,) : (1, 2))
        gens = vcat(signalboxes, [good ? target : -target])
        rb = alpha * pullback(B[q], H) + U - pullback(B[qn], transition)
        rv = pullback(V[q] + alpha * B[q], H) + G - pullback(V[qn], transition)
        q == 1 && (rv[1, 1] = rv[1, 1] - step)
        # At a stationary self-transition with zero required decrease,
        # the barrier/rank pair forces B_q(0)=0. Both residuals vanish
        # there. This also applies at an odd state allocated step zero.
        zero_decrease = q == 2 || iszero(step)
        face = q == qn && zero_decrease && all(g[1, 1] > 0 for g in gens)
        name = "$(n)_$(q)_$(good ? "a" : "b")_$(qn)"
        add_sos_constraint!(problem, rb, gens, "B_" * name; stationary_face = face)
        add_sos_constraint!(problem, rv, gens, "V_" * name; stationary_face = face)
    end
end

"Sparse exact row echelon factorization of the coefficient identities."
function factor_affine_identities(equations, priority)
    basis = Dict{Int, ExactAffine}()
    for original in equations
        row = original
        while !isempty(row.coefficients)
            pivot = argmin(i -> priority[i], collect(keys(row.coefficients)))
            v = row.coefficients[pivot]
            if haskey(basis, pivot)
                row = row - v * basis[pivot]
            else
                basis[pivot] = row / v
                break
            end
        end
        isempty(row.coefficients) &&
            !iszero(row.constant) &&
            error("inconsistent exact affine SOS identities")
    end
    basis
end
"Accept only exact identities, nonnegative multipliers and rational PSD tests."
function verify_exact_solution(problem, u)
    all(iszero(evaluate_exact(e, u)) for e in problem.coefficient_identities) || return false
    all(u[i] >= 0 for i in problem.nonnegative_ids) || return false
    for b in problem.psd_blocks
        Q = [u[i] for i in b.ids]
        CPC.exact_psd(Q).pass || return false
    end
    true
end
function verify_aggregate_supplies(problem, data, u)
    checks = Dict{String, Any}()
    for family in ("Gamma", "Upsilon")
        S = zeros(Rat, 2data.base.N, 2data.base.N)
        for n in 1:data.base.N
            G = evaluate_matrix(problem.supplies["$(family)_$(n)"], u)
            S += data.base.T[n]' * G * data.base.T[n]
        end
        @assert CPC.exact_psd(-S).pass "original undecomposed $(family) aggregate"
        checks[family] = Dict(
            "original_aggregate_exact" => fraction_rows(S),
            "maximum_eigenvalue_diagnostic" => eigmax(Symmetric(Float64.(S))),
            "exact_nonpositive" => true,
        )
    end
    checks
end
"""
Project a numerical iterate onto the exact affine identity system, then test cones.
Auxiliary Gram/clique entries are eliminated first so that the physical
polynomial coefficients are not reconstructed through ill-conditioned identities.
The rounding-grid and threshold order is fixed to preserve witness selection.
"""
function repair_exact_solution(problem, raw)
    # Reconstruct auxiliary Grams from the actual polynomial coefficients,
    # rather than solving ill-conditioned combinations for those coefficients.
    order = sort!(
        collect(eachindex(problem.variables));
        by = i -> (
            occursin("Gram", problem.variable_names[i]) ||
            occursin("clique", problem.variable_names[i]) ? 0 : 1,
            i,
        ),
    )
    priority = zeros(Int, length(order))
    for (i, j) in enumerate(order)
        priority[j] = i
    end
    basis = factor_affine_identities(problem.coefficient_identities, priority)
    pivots = sort!(collect(keys(basis)); by = i -> priority[i], rev = true)
    for bits in (32, 42, 52), threshold in (1e-9, 1e-7, 0.0)
        den = BigInt(2) ^ bits
        u = Rat[
            abs(v) < threshold ? CPC.rat(0) : round(BigInt, BigFloat(v) * den)//den for v in raw
        ]
        for i in pivots
            row = basis[i]
            u[i] =
                -row.constant -
                sum((v * u[j] for (j, v) in row.coefficients if j != i); init = CPC.rat(0))
        end
        if verify_exact_solution(problem, u)
            return (;
                u,
                bits,
                threshold,
                rank = length(pivots),
                maximum_correction = maximum(abs.(Float64.(u) - raw)),
            )
        end
    end
    error("Exact affine projection could not preserve all SOS cones")
end

fraction(x) = CPC.fraction(x)
fraction_rows(Q) = CPC.rows(fraction.(Q))
parse_fraction(s) = begin
    a, b = split(s, '/')

    parse(BigInt, a)//parse(BigInt, b)
end
function export_polynomial(H, center, physical_origin, physical_scale)
    k = length(center)
    normalized = Matrix{Rat}(I, k + 1, k + 1)
    normalized[2:end, 1] = -center
    original = permutedims(normalized) * H * normalized
    physical = zeros(Rat, k + 1, k + 1)
    physical[1, 1] = 1
    for i in 1:k
        physical[i + 1, 1] = -physical_origin[i] / physical_scale[i]
        physical[i + 1, i + 1] = 1 / physical_scale[i]
    end
    Dict(
        "centered_coefficient_matrix_exact" => fraction_rows(H),
        "original_normalized_coefficient_matrix_exact" => fraction_rows(original),
        "original_physical_coefficient_matrix_exact" =>
            fraction_rows(physical' * original * physical),
        "centered_coefficient_matrix_decimal" => CPC.rows(Float64.(H)),
    )
end
"Export the exact witness and physical-coordinate pullbacks using the stable schema."
function export_certificate(problem, data, method, repaired, raw; epsilon_allocation = nothing)
    base = data.base
    u = repaired.u
    steps = validate_progress_steps(base, epsilon_allocation)
    polys = Dict{String, Any}()
    origins = reduce(vcat, ([m.current_reference, base.voltage_reference] for m in base.modules))
    scales = repeat([base.current_scale, base.voltage_scale], base.N)
    for (name, H) in problem.polynomials
        if method == :monolithic
            center = data.equilibrium
            origin = origins
            scale = scales
        else
            n = parse(Int, split(name, '_')[2])
            idx = (2n - 1):2n
            center = data.equilibrium[idx]
            origin = origins[idx]
            scale = scales[idx]
        end
        polys[name] = export_polynomial(evaluate_matrix(H, u), center, origin, scale)
    end
    witnesses = [
        Dict(
            "name" => s.name,
            "stationary_face" => s.stationary_face,
            "residual_exact" => fraction_rows(evaluate_matrix(s.residual, u)),
            "Gram_exact" => fraction_rows(evaluate_matrix(s.gram, u)),
            "multipliers_exact" => [fraction(evaluate_exact(l, u)) for l in s.lambdas],
            "domain_generators_exact" => fraction_rows.(s.generators),
        ) for s in problem.sos_constraints
    ]
    result = Dict{String, Any}(
        "method" => string(method),
        "direct_BV_search" => true,
        "degree" => 2,
        "alpha_exact" => fraction(base.alpha),
        "epsilon_local_exact" => fraction(base.epsilon_local),
        "epsilon_global_exact" => fraction(sum(steps)),
        "unit_decrease_normalization" => "Divide monolithic B,V, or the sums of local B,V, by epsilon_global.",
        "controller_offset_exact" => fraction.(data.theta.a),
        "controller_ki_exact" => fraction.(data.theta.ki),
        "controller_kv_exact" => fraction.(data.theta.kv),
        "equilibrium_exact" => fraction.(data.equilibrium),
        "A_original_normalized_exact" => fraction_rows(data.A),
        "drift_original_normalized_exact" => fraction.(data.c),
        "variable_names" => problem.variable_names,
        "raw_solver_coefficients" => raw,
        "repaired_exact_coefficients" => fraction.(u),
        "coefficient_identity_count" => length(problem.coefficient_identities),
        "SOS_condition_count" => length(problem.sos_constraints),
        "PSD_block_count" => length(problem.psd_blocks),
        "maximum_coefficient_correction" => repaired.maximum_correction,
        "repair_rounding_bits" => repaired.bits,
        "repair_zero_threshold" => repaired.threshold,
        "repair_identity_rank" => repaired.rank,
        "polynomials" => polys,
        "SOS_witnesses" => witnesses,
        "original_aggregate_checks" =>
            method == :compositional ? verify_aggregate_supplies(problem, data, u) :
            Dict{String, Any}(),
        "supplies_exact" => Dict(
            name => fraction_rows(evaluate_matrix(H, u)) for (name, H) in problem.supplies
        ),
        "exact_proof_pass" => true,
    )
    if !isnothing(epsilon_allocation)
        delete!(result, "epsilon_local_exact")
        result["epsilon_reference_exact"] = fraction(base.epsilon_local)
        result["epsilon_allocation_exact"] = fraction.(steps)
        result["compositional_theorem"] = "Theorem 3: nonnegative local allocation with positive aggregate"
    end
    result
end

function synthesize_with_candidates(base, method, budget; controller_coordinates = :normalized)
    started = time_ns()
    history = Dict{String, Any}[]
    for attempt in 1:budget
        record = Dict{String, Any}("attempt" => attempt)
        push!(history, record)
        try
            t=@elapsed theta =
                generate_controller_candidate(base, attempt; coordinates = controller_coordinates)
            record["controller_LP_seconds"] = t
            record["controller_LP_count"] = theta.LP_count
            record["controller_offset_exact"] = fraction.(theta.a)
            record["controller_ki_exact"] = fraction.(theta.ki)
            record["controller_kv_exact"] = fraction.(theta.kv)
            t=@elapsed data = reconstruct_closed_loop(base, theta)
            record["physical_model_seconds"] = t
            record["candidate_equilibrium_decimal"] = Float64.(data.equilibrium)
            println(method, " attempt ", attempt, ": solving free B,V SOS problem")
            flush(stdout)
            t=@elapsed problem = build_certificate_problem(data, method)
            record["SOS_build_seconds"] = t
            t=@elapsed optimize!(problem.model)
            record["SOS_solve_seconds"] = t
            status = termination_status(problem.model)
            record["SOS_status"] = string(status)
            has_values(problem.model) || error("SOS solver returned $(status) without values")
            raw = value.(problem.variables)
            record["raw_solver_coefficients"] = raw
            record["raw_minimum_Gram_eigenvalue"] =
                minimum(eigmin(Symmetric([raw[i] for i in b.ids])) for b in problem.psd_blocks)
            println(method, " attempt ", attempt, ": repairing and verifying exact SOS identities")
            flush(stdout)
            t=@elapsed repaired = repair_exact_solution(problem, raw)
            record["exact_repair_verification_seconds"] = t
            @assert verify_exact_solution(problem, repaired.u)
            # A changed coefficient must not pass the exact identity check.
            tampered = copy(repaired.u)
            tampered[1] += 1
            @assert !verify_exact_solution(problem, tampered)
            result = export_certificate(problem, data, method, repaired, raw)
            record["outcome"] = "exact verified CPC and controller"
            result["attempts"] = history
            result["total_search_and_verification_seconds"] = (time_ns() - started) * 1e-9
            return result
        catch err
            record["outcome"] = "rejected"
            record["reason"] = sprint(showerror, err)
            record["elapsed_through_attempt_seconds"] = (time_ns() - started) * 1e-9
            println(method, " attempt ", attempt, " rejected: ", record["reason"])
            flush(stdout)
        end
    end
    Dict{String, Any}(
        "method" => string(method),
        "direct_BV_search" => true,
        "exact_proof_pass" => false,
        "attempts" => history,
        "total_search_and_verification_seconds" => (time_ns() - started) * 1e-9,
    )
end
"""
    verify_controller_vertices(base, theta)

Check an exported rational controller against the original physical equations.
Affine input and successor maps attain their box extrema at vertices, so these
four/eight-point checks establish admissibility and robust box preservation on
the full domains. The internal input here is nominal w0 in [-1,1], not the
translated supply coordinate omega. No numerical tolerance is used.
"""
function verify_controller_vertices(base, theta)
    @assert length(theta.a) == length(theta.ki) == length(theta.kv) == base.N
    reports = Dict{String, Any}[]
    for (n, converter) in enumerate(base.modules)
        current_lower = converter.current_reference - base.current_scale
        current_upper = converter.current_reference + base.current_scale
        voltage_lower = base.voltage_reference - base.voltage_scale
        voltage_upper = base.voltage_reference + base.voltage_scale
        degree = sum(base.conductance[n, :])
        input_vertices = Dict{String, Any}[]
        successor_vertices = Dict{String, Any}[]
        for zi in (-1, 1), zv in (-1, 1)
            current = converter.current_reference + base.current_scale * zi
            voltage = base.voltage_reference + base.voltage_scale * zv
            input =
                theta.a[n] +
                (
                    -theta.ki[n] * (current - converter.current_reference) +
                    theta.kv[n] * (voltage - base.voltage_reference)
                ) / converter.source
            @assert 0 <= input <= 1 "input vertex bound failed for converter $n"
            push!(input_vertices, Dict("z" => [zi, zv], "input_exact" => fraction(input)))
            for nominal_input in (-1, 1)
                next_current =
                    current +
                    base.dt / converter.L *
                    (converter.source * input - converter.r * current - voltage)
                next_voltage =
                    voltage +
                    base.dt / converter.C * (
                        current - converter.G * voltage +
                        degree *
                        (base.voltage_scale * nominal_input - (voltage - base.voltage_reference))
                    )
                @assert current_lower <= next_current <= current_upper "current vertex bound failed for converter $n"
                @assert voltage_lower <= next_voltage <= voltage_upper "voltage vertex bound failed for converter $n"
                push!(
                    successor_vertices,
                    Dict(
                        "z" => [zi, zv],
                        "nominal_internal_input" => nominal_input,
                        "next_current_exact" => fraction(next_current),
                        "next_voltage_exact" => fraction(next_voltage),
                        "lower_slack_exact" =>
                            fraction.([next_current - current_lower, next_voltage - voltage_lower]),
                        "upper_slack_exact" =>
                            fraction.([current_upper - next_current, voltage_upper - next_voltage]),
                    ),
                )
            end
        end
        push!(
            reports,
            Dict(
                "converter" => n,
                "current_interval_exact" => fraction.([current_lower, current_upper]),
                "voltage_interval_exact" => fraction.([voltage_lower, voltage_upper]),
                "input_vertices" => input_vertices,
                "successor_vertices" => successor_vertices,
            ),
        )
    end
    return Dict(
        "exact_pass" => true,
        "internal_input" => "nominal w0 in [-1,1]; not centered omega",
        "input_vertex_count" => 4 * base.N,
        "successor_vertex_count" => 8 * base.N,
        "converters" => reports,
    )
end

"""
Rebuild a saved numerical witness using exact coefficients and the same ordering.
Comparing every exported polynomial and SOS witness prevents a valid internal
solution from masking an incorrect physical-coordinate or file export.
"""
function verify_exported_certificate(base, result, method)
    theta = (
        a = parse_fraction.(result["controller_offset_exact"]),
        ki = parse_fraction.(result["controller_ki_exact"]),
        kv = parse_fraction.(result["controller_kv_exact"]),
    )
    verify_controller_vertices(base, theta)
    allocation =
        haskey(result, "epsilon_allocation_exact") ?
        parse_fraction.(result["epsilon_allocation_exact"]) : nothing
    data = reconstruct_closed_loop(base, theta)
    problem = build_certificate_problem(data, method; epsilon_allocation = allocation)
    @assert problem.variable_names == result["variable_names"]
    coefficients = parse_fraction.(result["repaired_exact_coefficients"])
    @assert verify_exact_solution(problem, coefficients) "fresh exact proof failed"
    @assert fraction_rows(data.A) == result["A_original_normalized_exact"]
    repaired = (
        u = coefficients,
        bits = result["repair_rounding_bits"],
        threshold = result["repair_zero_threshold"],
        rank = result["repair_identity_rank"],
        maximum_correction = result["maximum_coefficient_correction"],
    )
    expected = export_certificate(
        problem,
        data,
        method,
        repaired,
        result["raw_solver_coefficients"];
        epsilon_allocation = allocation,
    )
    for key in ("polynomials", "SOS_witnesses", "supplies_exact", "original_aggregate_checks")
        @assert expected[key] == result[key] "exported $key differs from exact solution"
    end
    return problem
end

"Recheck the paired-result format produced by the standalone synthesis CLI."
function verify_saved_results(path)
    saved = TOML.parsefile(path)
    base = CPC.generate_instance(saved["N"], saved["seed"])
    for method in (:monolithic, :compositional)
        result = saved[string(method)]
        @assert result["exact_proof_pass"]
        problem = verify_exported_certificate(base, result, method)
        println(
            "Fresh reload ",
            method,
            ": exact PASS; ",
            length(problem.sos_constraints),
            " SOS obligations, ",
            length(problem.coefficient_identities),
            " coefficient identities",
        )
    end
end

function main(args = ARGS)
    BLAS.set_num_threads(1)
    if !isempty(args) && args[1] == "verify"
        verify_saved_results(args[2])
        return
    end
    N = length(args) >= 1 ? parse(Int, args[1]) : 2
    seed = length(args) >= 2 ? parse(Int, args[2]) : 20261122
    budget = length(args) >= 3 ? parse(Int, args[3]) : 7
    coordinates = length(args) >= 4 ? Symbol(args[4]) : :normalized
    coordinates in (:normalized, :physical) ||
        error("fourth argument must be normalized or physical")
    base = CPC.generate_instance(N, seed)
    println("Warming both direct CPC search paths on a separate N=2 parameter seed")
    flush(stdout)
    warm = CPC.generate_instance(2, 20261121)
    warm_status = Dict{String, Any}()
    warm_seeds = Dict{String, Any}()
    for method in (:monolithic, :compositional)
        wr = synthesize_with_candidates(warm, method, 1; controller_coordinates = coordinates)
        warm_seeds[string(method)] = 20261121
        if !wr["exact_proof_pass"]
            println(
                "Completing ",
                method,
                " warmup on the test plant; no solution values are reused",
            )
            flush(stdout)
            wr = synthesize_with_candidates(base, method, 1; controller_coordinates = coordinates)
            warm_seeds[string(method)] = seed
        end
        warm_status[string(method)] = wr["exact_proof_pass"]
    end
    directory = joinpath(
        get(ENV, "CPC_RESULTS_DIR", joinpath(@__DIR__, "Results")),
        "Direct_CPC_N$(N)_" * Dates.format(now(UTC), "yyyymmddTHHMMSSsss") * "Z",
    )
    mkpath(directory)
    path = joinpath(directory, "results.toml")
    result = Dict{String, Any}(
        "N" => N,
        "seed" => seed,
        "candidate_budget" => budget,
        "conda_environment" => get(ENV, "CONDA_DEFAULT_ENV", ""),
        "Julia_version" => string(VERSION),
        "JuMP_version" => string(pkgversion(JuMP)),
        "controller_search" => "admissibility LP candidates, full B,V SOS oracle",
        "controller_coordinates" => string(coordinates),
        "SOS_variable_units" => "original; no certificate-unit scaling or additional robustness margin",
        "specification" => "F G (a_1 OR ... OR a_N)",
        "initial_automaton_state" => "bad",
        "timing_scope" => "all candidate LP construction/solves, physical reconstruction, B,V SOS assembly/solve, exact repair/checks and witness construction; excludes library loading, separate-seed warmup and file export",
        "warmup_exact_status" => warm_status,
        "warmup_seed" => 20261121,
        "warmup_seed_by_method" => warm_seeds,
        "source_sha256" => bytes2hex(sha256(read(@__FILE__))),
    )
    println("Direct free B,V search, N=", N, "; no stability decay constraints")
    flush(stdout)
    for method in (:monolithic, :compositional)
        result[string(method)] =
            synthesize_with_candidates(base, method, budget; controller_coordinates = coordinates)
        open(path, "w") do io
            TOML.print(io, result; sorted = true)
        end
        println(method, ": exact_proof_pass=", result[string(method)]["exact_proof_pass"])
        flush(stdout)
    end
    cp(@__FILE__, joinpath(directory, "Synthesis_Direct_CPC.jl.txt"))
    cp(
        joinpath(@__DIR__, "Synthesis_Buck_CPC.jl"),
        joinpath(directory, "Synthesis_Buck_CPC.jl.txt"),
    )
    println("Results: ", path)
    flush(stdout)
    all(result[string(m)]["exact_proof_pass"] for m in (:monolithic, :compositional)) &&
        verify_saved_results(path)
end
abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
