if !isdefined(@__MODULE__, :LocalCPCConfig)
    include(joinpath(@__DIR__, "CPC_SOS_core.jl"))
end
using Random

Base.@kwdef struct HeterogeneousCPCConfig
    clusters_size_2::Int = 20
    clusters_size_3::Int = 20
    certificate_degree::Int = 2
    alpha::Float64 = 0.01
    epsilon_lower_bound::Float64 = 1.0e-6
    epsilon_upper_bound::Float64 = 0.1
    coefficient_regularization::Float64 = 1.0e-5
    supply_regularization::Float64 = 1.0
    epsilon_reward::Float64 = 1.0e-3
    verification_samples::Int = 20_000
    verification_tolerance::Float64 = 1.0e-7
    quiet::Bool = true
end

function alternating_cluster_sizes(config::HeterogeneousCPCConfig)
    config.clusters_size_2 == config.clusters_size_3 ||
        throw(ArgumentError("strict alternation requires equal numbers of size-2 and size-3 clusters"))
    return repeat([2, 3], config.clusters_size_2)
end

function cluster_interconnection_matrix(cluster_count::Int)
    # Boundary ordering per cluster is [left, right].
    M = zeros(2cluster_count, 2cluster_count)
    for cluster in 1:cluster_count
        previous_cluster = mod1(cluster - 1, cluster_count)
        next_cluster = mod1(cluster + 1, cluster_count)
        M[2cluster - 1, 2previous_cluster] = 1.0
        M[2cluster, 2next_cluster - 1] = 1.0
    end
    return M
end

function heterogeneous_global_lmi(P_size_2, P_size_3, cluster_sizes)
    cluster_count = length(cluster_sizes)
    signal_dimension = 2cluster_count
    stacked_dimension = 2signal_dimension
    Q = [AffExpr(0.0) for _ in 1:stacked_dimension, _ in 1:stacked_dimension]
    for cluster in 1:cluster_count
        P = cluster_sizes[cluster] == 2 ? P_size_2 : P_size_3
        local_indices = (2cluster - 1, 2cluster,
                         signal_dimension + 2cluster - 1,
                         signal_dimension + 2cluster)
        for row in 1:4, column in 1:4
            add_to_expression!(Q[local_indices[row], local_indices[column]],
                               P[row, column])
        end
    end
    M = cluster_interconnection_matrix(cluster_count)
    T = [M; I(signal_dimension)]
    return T' * Q * T
end

function numeric_heterogeneous_global_lmi(P_size_2, P_size_3, cluster_sizes)
    cluster_count = length(cluster_sizes)
    signal_dimension = 2cluster_count
    Q = zeros(2signal_dimension, 2signal_dimension)
    for cluster in 1:cluster_count
        P = cluster_sizes[cluster] == 2 ? P_size_2 : P_size_3
        indices = (2cluster - 1, 2cluster,
                   signal_dimension + 2cluster - 1,
                   signal_dimension + 2cluster)
        Q[collect(indices), collect(indices)] .+= P
    end
    M = cluster_interconnection_matrix(cluster_count)
    T = [M; I(signal_dimension)]
    return Symmetric(T' * Q * T)
end

function cluster_supply_polynomial(P, signals)
    return sum(P[i, j] * signals[i] * signals[j]
               for i in eachindex(signals), j in eachindex(signals))
end

function add_cluster_certificate!(model, dimension::Int, supplies,
                                  config::HeterogeneousCPCConfig)
    degree = config.certificate_degree
    @polyvar cluster_xhat[1:dimension] cluster_what[1:2]
    x = physical_x.(cluster_xhat)
    external = physical_x.(cluster_what)
    neighbor_sums = [
        (i == 1 ? external[1] : x[i - 1]) +
        (i == dimension ? external[2] : x[i + 1])
        for i in 1:dimension
    ]
    xhat_next = [(room_dynamics(x[i], neighbor_sums[i]) - X_CENTER) / X_SCALE
                 for i in 1:dimension]

    Bq0, c_Bq0, basis_Bq0 = add_poly!(model, cluster_xhat, degree)
    Bq1, c_Bq1, basis_Bq1 = add_poly!(model, cluster_xhat, degree)
    Vq0, c_Vq0, basis_Vq0 = add_poly!(model, cluster_xhat, degree)
    Vq1, c_Vq1, basis_Vq1 = add_poly!(model, cluster_xhat, degree)
    Bq0_next = Bq0(cluster_xhat => xhat_next)
    Bq1_next = Bq1(cluster_xhat => xhat_next)
    Vq0_next = Vq0(cluster_xhat => xhat_next)
    Vq1_next = Vq1(cluster_xhat => xhat_next)

    epsilon_local = @variable(model,
        lower_bound=config.epsilon_lower_bound,
        upper_bound=config.epsilon_upper_bound,
        base_name="epsilon_size_$(dimension)")
    Upsilon_q0, Upsilon_q1, Gamma_q0, Gamma_q1 = supplies
    signals = [external[1], external[2], x[1], x[end]]
    U0 = cluster_supply_polynomial(Upsilon_q0, signals)
    U1 = cluster_supply_polynomial(Upsilon_q1, signals)
    G0 = cluster_supply_polynomial(Gamma_q0, signals)
    G1 = cluster_supply_polynomial(Gamma_q1, signals)

    x0_lower, x0_upper = normalized_x(20.0), normalized_x(25.0)
    xa_lower, xa_upper = normalized_x(23.0), normalized_x(26.0)
    full_state = [1.0 - z^2 for z in cluster_xhat]
    full_input = [1.0 - z^2 for z in cluster_what]
    initial_domain = [interval_polynomial(z, x0_lower, x0_upper)
                      for z in cluster_xhat]
    a_domain = vcat([interval_polynomial(z, xa_lower, xa_upper)
                     for z in cluster_xhat], full_input)
    b_domains = Any[]
    for i in 1:dimension
        low = vcat(copy(full_state), full_input)
        low[i] = interval_polynomial(cluster_xhat[i], -1.0, xa_lower)
        push!(b_domains, low)
        high = vcat(copy(full_state), full_input)
        high[i] = interval_polynomial(cluster_xhat[i], xa_upper, 1.0)
        push!(b_domains, high)
    end

    infos = Any[]
    state_order = cld(degree, 2)
    transition_order = degree
    add_psatz_constraint!(infos, model, -Bq0, cluster_xhat,
                          initial_domain, state_order)
    add_psatz_constraint!(infos, model, Vq0 + config.alpha * Bq0,
                          cluster_xhat, full_state, state_order)
    add_psatz_constraint!(infos, model, Vq1 + config.alpha * Bq1,
                          cluster_xhat, full_state, state_order)

    variables = vcat(cluster_xhat, cluster_what)
    function add_region!(domain, B0n, V0n, B1n, V1n)
        residuals = (
            -B0n + config.alpha * Bq0 + U0,
            -V0n + Vq0 + config.alpha * Bq0 - epsilon_local + G0,
            -B1n + config.alpha * Bq1 + U1,
            -V1n + Vq1 + config.alpha * Bq1 + G1,
        )
        for residual in residuals
            add_psatz_constraint!(infos, model, residual, variables,
                                  domain, transition_order)
        end
    end
    add_region!(a_domain, Bq1_next, Vq1_next, Bq1_next, Vq1_next)
    for domain in b_domains
        add_region!(domain, Bq0_next, Vq0_next, Bq0_next, Vq0_next)
    end

    return (
        dimension=dimension,
        xhat=cluster_xhat,
        what=cluster_what,
        coefficient_refs=(c_Bq0, c_Bq1, c_Vq0, c_Vq1),
        bases=(basis_Bq0, basis_Bq1, basis_Vq0, basis_Vq1),
        supplies=supplies,
        epsilon_local=epsilon_local,
        infos=infos,
    )
end

function build_heterogeneous_model(config::HeterogeneousCPCConfig)
    iseven(config.certificate_degree) ||
        throw(ArgumentError("certificate_degree must be even"))
    cluster_sizes = alternating_cluster_sizes(config)
    sum(cluster_sizes) == 100 ||
        throw(ArgumentError("the requested benchmark must contain 100 scalar rooms"))

    model = Model(Mosek.Optimizer)
    config.quiet && set_silent(model)
    function supply_tuple(prefix)
        matrices = Any[]
        for suffix in ("Upsilon_q0", "Upsilon_q1", "Gamma_q0", "Gamma_q1")
            name = Symbol(prefix, "_", suffix)
            push!(matrices, @variable(model, [1:4, 1:4], Symmetric,
                                      base_name=String(name)))
        end
        return Tuple(matrices)
    end
    supplies_size_2 = supply_tuple("size2")
    supplies_size_3 = supply_tuple("size3")

    # Common supplies across q0/q1 make the interconnection LMI valid for
    # every mixed state of the product of the local co-Buchi automata.
    for supplies in (supplies_size_2, supplies_size_3)
        @constraint(model, supplies[1] .== supplies[2])
        @constraint(model, supplies[3] .== supplies[4])
    end

    for condition in 1:4
        global_matrix = heterogeneous_global_lmi(supplies_size_2[condition],
                                                 supplies_size_3[condition],
                                                 cluster_sizes)
        @constraint(model, -global_matrix in PSDCone())
    end

    certificate_size_2 = add_cluster_certificate!(model, 2, supplies_size_2, config)
    certificate_size_3 = add_cluster_certificate!(model, 3, supplies_size_3, config)
    certificates = (certificate_size_2, certificate_size_3)

    coefficient_cost = sum(c^2 for certificate in certificates
                           for coefficients in certificate.coefficient_refs
                           for c in coefficients)
    supply_cost = sum(P[i, j]^2 for supplies in (supplies_size_2, supplies_size_3)
                      for P in supplies for i in 1:4 for j in 1:4)
    epsilon_total = certificate_size_2.epsilon_local +
                    certificate_size_3.epsilon_local
    @objective(model, Min,
        config.coefficient_regularization * coefficient_cost +
        config.supply_regularization * supply_cost -
        config.epsilon_reward * epsilon_total)
    return model, (cluster_sizes=cluster_sizes, certificates=certificates,
                   supplies=(supplies_size_2, supplies_size_3))
end

function verify_cluster_randomly(certificate, config::HeterogeneousCPCConfig, rng)
    dimension = certificate.dimension
    polynomials = map(numeric_polynomial, certificate.coefficient_refs,
                      certificate.bases)
    term_data = map(p -> polynomial_term_data(p, certificate.xhat), polynomials)
    B0data, B1data, V0data, V1data = term_data
    supplies = map(P -> value.(P), certificate.supplies)
    U0, U1, G0, G1 = supplies
    epsilon_local = value(certificate.epsilon_local)
    minimum_residual = Inf
    worst = nothing

    function check!(name, residual, state, input=zeros(2))
        if residual < minimum_residual
            minimum_residual = residual
            worst = (name=name, state=copy(state), input=copy(input))
        end
    end
    function next_state(state, input)
        x, external = physical_x.(state), physical_x.(input)
        return [normalized_x(room_dynamics(x[i],
                    (i == 1 ? external[1] : x[i - 1]) +
                    (i == dimension ? external[2] : x[i + 1])))
                for i in 1:dimension]
    end

    for _ in 1:config.verification_samples
        initial = [rand(rng) * (normalized_x(25.0) - normalized_x(20.0)) +
                   normalized_x(20.0) for _ in 1:dimension]
        check!("initial_B_q0", -evaluate_term_data(B0data, initial), initial)
        state = 2rand(rng, dimension) .- 1
        B0, B1 = evaluate_term_data(B0data, state), evaluate_term_data(B1data, state)
        V0, V1 = evaluate_term_data(V0data, state), evaluate_term_data(V1data, state)
        check!("lower_bound_q0", V0 + config.alpha * B0, state)
        check!("lower_bound_q1", V1 + config.alpha * B1, state)

        if rand(rng, Bool)
            state = [rand(rng) * (normalized_x(26.0) - normalized_x(23.0)) +
                     normalized_x(23.0) for _ in 1:dimension]
            successor_q1 = true
        else
            state = 2rand(rng, dimension) .- 1
            index = rand(rng, 1:dimension)
            state[index] = rand(rng, Bool) ?
                rand(rng) * (normalized_x(23.0) + 1.0) - 1.0 :
                rand(rng) * (1.0 - normalized_x(26.0)) + normalized_x(26.0)
            successor_q1 = false
        end
        input = 2rand(rng, 2) .- 1
        successor = next_state(state, input)
        x, external = physical_x.(state), physical_x.(input)
        signals = [external[1], external[2], x[1], x[end]]
        B0, B1 = evaluate_term_data(B0data, state), evaluate_term_data(B1data, state)
        V0, V1 = evaluate_term_data(V0data, state), evaluate_term_data(V1data, state)
        Bn = evaluate_term_data(successor_q1 ? B1data : B0data, successor)
        Vn = evaluate_term_data(successor_q1 ? V1data : V0data, successor)
        supplies_at_point = map(P -> dot(signals, P * signals), supplies)
        u0, u1, g0, g1 = supplies_at_point
        label = successor_q1 ? "a" : "b"
        check!("$(label)_B_q0", -Bn + config.alpha * B0 + u0, state, input)
        check!("$(label)_V_q0", -Vn + V0 + config.alpha * B0 - epsilon_local + g0,
               state, input)
        check!("$(label)_B_q1", -Bn + config.alpha * B1 + u1, state, input)
        check!("$(label)_V_q1", -Vn + V1 + config.alpha * B1 + g1, state, input)
    end
    return (minimum_residual=minimum_residual, worst=worst,
            epsilon_local=epsilon_local, polynomials=polynomials,
            supplies=supplies)
end

function solve_heterogeneous_cpc(config::HeterogeneousCPCConfig)
    build_seconds = @elapsed model, metadata = build_heterogeneous_model(config)
    solve_seconds = @elapsed optimize!(model)
    status = termination_status(model)
    if !(status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)) || !has_values(model)
        return (valid=false, status=status, raw_status=raw_status(model),
                build_seconds=build_seconds, solve_seconds=solve_seconds,
                reason="heterogeneous SDP did not return a usable solution")
    end
    verification_seconds = @elapsed begin
        rng = MersenneTwister(20260902)
        cluster_results = map(c -> verify_cluster_randomly(c, config, rng),
                              metadata.certificates)
        minimum_residual = minimum(r.minimum_residual for r in cluster_results)
        infos = vcat(metadata.certificates[1].infos,
                     metadata.certificates[2].infos)
        gram_minimum, gram_count = minimum_gram_eigenvalue(infos)
        feasibility_report = primal_feasibility_report(model)
        maximum_conic_violation = isempty(feasibility_report) ? 0.0 :
                                  maximum(values(feasibility_report))
        numeric_supplies = map(group -> map(P -> value.(P), group),
                               metadata.supplies)
        global_margins = ntuple(4) do condition
            matrix = numeric_heterogeneous_global_lmi(
                numeric_supplies[1][condition], numeric_supplies[2][condition],
                metadata.cluster_sizes)
            -maximum(eigvals(matrix))
        end
    end
    tolerance = config.verification_tolerance
    valid = minimum_residual >= -tolerance && gram_minimum >= -tolerance &&
            maximum_conic_violation <= 1.0e-6 &&
            minimum(global_margins) >= -tolerance
    return (valid=valid, status=status, raw_status=raw_status(model),
            build_seconds=build_seconds, solve_seconds=solve_seconds,
            verification_seconds=verification_seconds,
            minimum_residual=minimum_residual, cluster_results=cluster_results,
            minimum_gram_eigenvalue=gram_minimum, gram_matrix_count=gram_count,
            maximum_conic_violation=maximum_conic_violation,
            global_lmi_margins=global_margins)
end

function main()
    degree = isempty(ARGS) ? 2 : parse(Int, ARGS[1])
    alpha = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.01
    config = HeterogeneousCPCConfig(certificate_degree=degree, alpha=alpha)
    result = solve_heterogeneous_cpc(config)
    println("Heterogeneous deterministic-local CPC synthesis")
    println("  clusters: 20 x dimension 2 and 20 x dimension 3 (100 rooms)")
    println("  degree / alpha: ", (degree, alpha))
    @printf("  build / solve: %.3f / %.3f s\n",
            result.build_seconds, result.solve_seconds)
    println("  status: ", result.status, "; verified: ", result.valid)
    if hasproperty(result, :minimum_residual)
        @printf("  verification: %.3f s; minimum sampled residual: %.9e\n",
                result.verification_seconds, result.minimum_residual)
        println("  epsilons (size 2, size 3): ",
                map(r -> r.epsilon_local, result.cluster_results))
        println("  maximum conic violation: ", result.maximum_conic_violation)
        println("  minimum Gram eigenvalue: ", result.minimum_gram_eigenvalue)
        println("  global LMI margins: ", result.global_lmi_margins)
    else
        println("  reason: ", result.reason, "; raw status: ", result.raw_status)
    end
    result.valid || exit(2)
    output_path = joinpath(@__DIR__,
        "CPC_Heterogeneous_results_20x2_20x3_degree$(degree).txt")
    open(output_path, "w") do file
        println(file, "status: numerically verified heterogeneous deterministic-local CPC found")
        println(file, "cluster_pattern: 20 repetitions of [2, 3]")
        println(file, "scalar_rooms: 100")
        println(file, "common_supply_across_states: true")
        println(file, "degree: ", degree)
        println(file, "alpha: ", alpha)
        println(file, "build_seconds: ", result.build_seconds)
        println(file, "solve_seconds: ", result.solve_seconds)
        println(file, "verification_seconds: ", result.verification_seconds)
        println(file, "minimum_sampled_residual: ", result.minimum_residual)
        println(file, "maximum_conic_violation: ", result.maximum_conic_violation)
        println(file, "minimum_gram_eigenvalue: ", result.minimum_gram_eigenvalue)
        println(file, "global_lmi_margins: ", result.global_lmi_margins)
        for (index, cluster_result) in enumerate(result.cluster_results)
            cluster_dimension = index == 1 ? 2 : 3
            println(file, "epsilon_size_", cluster_dimension, ": ",
                    cluster_result.epsilon_local)
            for (name, polynomial) in zip(("Bq0", "Bq1", "Vq0", "Vq1"),
                                          cluster_result.polynomials)
                println(file, "size", cluster_dimension, "_", name,
                        "(xhat): ", polynomial)
            end
            for (name, matrix) in zip(("Upsilon_q0", "Upsilon_q1",
                                       "Gamma_q0", "Gamma_q1"),
                                      cluster_result.supplies)
                println(file, "size", cluster_dimension, "_", name, ": ", matrix)
            end
        end
    end
    println("Verified result written to ", output_path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
