if !isdefined(@__MODULE__, :LocalCPCConfig)
    include(joinpath(@__DIR__, "CPC_SOS_core.jl"))
end
using Random

Base.@kwdef struct MonolithicCPCConfig
    rooms::Int = 10
    certificate_degree::Int = 2
    alpha::Float64 = 0.01
    epsilon_lower_bound::Float64 = 1.0e-6
    epsilon_upper_bound::Float64 = 0.1
    coefficient_regularization::Float64 = 1.0e-5
    epsilon_reward::Float64 = 1.0e-3
    verification_samples::Int = 10_000
    verification_tolerance::Float64 = 1.0e-7
    quiet::Bool = true
end

function monolithic_size_estimate(rooms::Int, certificate_degree::Int)
    transition_order = certificate_degree
    main_basis = binomial(rooms + transition_order, transition_order)
    multiplier_basis = binomial(rooms + transition_order - 1,
                                transition_order - 1)
    transition_regions = 1 + 2rooms
    transition_inequalities = 4transition_regions
    main_gram_entries = transition_inequalities * main_basis * (main_basis + 1) ÷ 2
    domain_gram_entries = transition_inequalities * rooms *
                          multiplier_basis * (multiplier_basis + 1) ÷ 2
    return (main_basis=main_basis, multiplier_basis=multiplier_basis,
            transition_regions=transition_regions,
            transition_inequalities=transition_inequalities,
            approximate_psd_scalar_variables=main_gram_entries + domain_gram_entries)
end

function build_monolithic_cpc_model(config::MonolithicCPCConfig)
    n, degree = config.rooms, config.certificate_degree
    n >= 2 || throw(ArgumentError("the monolithic ring needs at least two rooms"))
    iseven(degree) || throw(ArgumentError("certificate_degree must be even"))

    model = Model(Mosek.Optimizer)
    config.quiet && set_silent(model)
    @polyvar xhat[1:n]
    x = physical_x.(xhat)
    w = [x[mod1(i - 1, n)] + x[mod1(i + 1, n)] for i in 1:n]
    xhat_next = [(room_dynamics(x[i], w[i]) - X_CENTER) / X_SCALE for i in 1:n]

    Bq0, c_Bq0, basis_Bq0 = add_poly!(model, xhat, degree)
    Bq1, c_Bq1, basis_Bq1 = add_poly!(model, xhat, degree)
    Vq0, c_Vq0, basis_Vq0 = add_poly!(model, xhat, degree)
    Vq1, c_Vq1, basis_Vq1 = add_poly!(model, xhat, degree)
    Bq0_next = Bq0(xhat => xhat_next)
    Bq1_next = Bq1(xhat => xhat_next)
    Vq0_next = Vq0(xhat => xhat_next)
    Vq1_next = Vq1(xhat => xhat_next)

    @variable(model,
        config.epsilon_lower_bound <= epsilon_local <= config.epsilon_upper_bound)

    x0_lower, x0_upper = normalized_x(20.0), normalized_x(25.0)
    xa_lower, xa_upper = normalized_x(23.0), normalized_x(26.0)
    full_domain = [1.0 - xhat[i]^2 for i in 1:n]
    initial_domain = [interval_polynomial(xhat[i], x0_lower, x0_upper)
                      for i in 1:n]
    all_a_domain = [interval_polynomial(xhat[i], xa_lower, xa_upper)
                    for i in 1:n]

    b_domains = Vector{Vector{Any}}()
    for i in 1:n
        low_domain = copy(full_domain)
        low_domain[i] = interval_polynomial(xhat[i], -1.0, xa_lower)
        push!(b_domains, low_domain)
        high_domain = copy(full_domain)
        high_domain[i] = interval_polynomial(xhat[i], xa_upper, 1.0)
        push!(b_domains, high_domain)
    end

    infos = Any[]
    scalar_order = cld(degree, 2)
    transition_order = degree
    add_psatz_constraint!(infos, model, -Bq0, xhat, initial_domain, scalar_order)
    add_psatz_constraint!(infos, model, Vq0 + config.alpha * Bq0,
                          xhat, full_domain, scalar_order)
    add_psatz_constraint!(infos, model, Vq1 + config.alpha * Bq1,
                          xhat, full_domain, scalar_order)

    function add_transition_region!(domain, B0n, V0n, B1n, V1n)
        residuals = (
            -B0n + config.alpha * Bq0,
            -V0n + Vq0 + config.alpha * Bq0 - epsilon_local,
            -B1n + config.alpha * Bq1,
            -V1n + Vq1 + config.alpha * Bq1,
        )
        for residual in residuals
            add_psatz_constraint!(infos, model, residual, xhat, domain,
                                  transition_order)
        end
    end

    # Global a: q0 -> q1 and q1 -> q1.
    add_transition_region!(all_a_domain, Bq1_next, Vq1_next,
                           Bq1_next, Vq1_next)
    # Global b is a union: at least one room is below 23 or above 26.
    for domain in b_domains
        add_transition_region!(domain, Bq0_next, Vq0_next,
                               Bq0_next, Vq0_next)
    end

    coefficient_refs = (c_Bq0, c_Bq1, c_Vq0, c_Vq1)
    coefficient_cost = sum(c^2 for coefficients in coefficient_refs for c in coefficients)
    @objective(model, Min,
        config.coefficient_regularization * coefficient_cost -
        config.epsilon_reward * epsilon_local)

    metadata = (
        xhat=xhat,
        xhat_next=xhat_next,
        coefficient_refs=coefficient_refs,
        bases=(basis_Bq0, basis_Bq1, basis_Vq0, basis_Vq1),
        epsilon_local=epsilon_local,
        infos=infos,
    )
    return model, metadata
end

function monolithic_next_state(xhat_point)
    n = length(xhat_point)
    x = physical_x.(xhat_point)
    return [normalized_x(room_dynamics(x[i],
                x[mod1(i - 1, n)] + x[mod1(i + 1, n)])) for i in 1:n]
end

function verify_monolithic_solution(model, metadata, config::MonolithicCPCConfig)
    status = termination_status(model)
    if !(status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)) || !has_values(model)
        return (valid=false, status=status, primal=primal_status(model),
                raw_status=raw_status(model),
                reason="the monolithic SDP did not return a usable solution")
    end

    polynomials = map(numeric_polynomial, metadata.coefficient_refs, metadata.bases)
    term_data = map(p -> polynomial_term_data(p, metadata.xhat), polynomials)
    epsilon_local = value(metadata.epsilon_local)
    rng = MersenneTwister(20260902)
    minimum_residual = Inf
    worst = nothing

    function check!(name, residual, point)
        if residual < minimum_residual
            minimum_residual = residual
            worst = (name=name, point=copy(point))
        end
    end

    B0data, B1data, V0data, V1data = term_data
    for _ in 1:config.verification_samples
        initial_point = [rand(rng) * (normalized_x(25.0) - normalized_x(20.0)) +
                         normalized_x(20.0) for _ in 1:config.rooms]
        check!("initial_B_q0", -evaluate_term_data(B0data, initial_point),
               initial_point)

        point = 2rand(rng, config.rooms) .- 1
        B0, B1 = evaluate_term_data(B0data, point), evaluate_term_data(B1data, point)
        V0, V1 = evaluate_term_data(V0data, point), evaluate_term_data(V1data, point)
        check!("lower_bound_q0", V0 + config.alpha * B0, point)
        check!("lower_bound_q1", V1 + config.alpha * B1, point)

        # Half the transition samples are drawn from all-a; the remainder
        # are forced into one of the 2N components of the global-b union.
        if isodd(rand(rng, UInt))
            point = [rand(rng) * (normalized_x(26.0) - normalized_x(23.0)) +
                     normalized_x(23.0) for _ in 1:config.rooms]
            successor_is_q1 = true
        else
            point = 2rand(rng, config.rooms) .- 1
            index = rand(rng, 1:config.rooms)
            if rand(rng, Bool)
                point[index] = rand(rng) * (normalized_x(23.0) + 1.0) - 1.0
            else
                point[index] = rand(rng) * (1.0 - normalized_x(26.0)) +
                               normalized_x(26.0)
            end
            successor_is_q1 = false
        end
        next_point = monolithic_next_state(point)
        B0, B1 = evaluate_term_data(B0data, point), evaluate_term_data(B1data, point)
        V0, V1 = evaluate_term_data(V0data, point), evaluate_term_data(V1data, point)
        Bn = evaluate_term_data(successor_is_q1 ? B1data : B0data, next_point)
        Vn = evaluate_term_data(successor_is_q1 ? V1data : V0data, next_point)
        label = successor_is_q1 ? "a" : "b"
        check!("$(label)_B_q0", -Bn + config.alpha * B0, point)
        check!("$(label)_V_q0", -Vn + V0 + config.alpha * B0 - epsilon_local, point)
        check!("$(label)_B_q1", -Bn + config.alpha * B1, point)
        check!("$(label)_V_q1", -Vn + V1 + config.alpha * B1, point)
    end

    feasibility_report = primal_feasibility_report(model)
    maximum_conic_violation = isempty(feasibility_report) ? 0.0 :
                              maximum(values(feasibility_report))
    minimum_gram_value, gram_count = minimum_gram_eigenvalue(metadata.infos)
    valid = minimum_residual >= -config.verification_tolerance &&
            maximum_conic_violation <= 1.0e-6 &&
            minimum_gram_value >= -config.verification_tolerance
    return (valid=valid, status=status, primal=primal_status(model),
            raw_status=raw_status(model), epsilon_local=epsilon_local,
            polynomials=polynomials, minimum_sampled_residual=minimum_residual,
            worst=worst, maximum_conic_violation=maximum_conic_violation,
            minimum_gram_eigenvalue=minimum_gram_value,
            gram_matrix_count=gram_count)
end

function solve_monolithic_cpc(config::MonolithicCPCConfig)
    estimate = monolithic_size_estimate(config.rooms, config.certificate_degree)
    println("Monolithic size estimate: ", estimate)
    if estimate.approximate_psd_scalar_variables > 5_000_000
        return (valid=false, config=config, estimate=estimate,
                reason="estimated SDP exceeds the five-million-variable safety limit")
    end
    build_seconds = @elapsed model, metadata = build_monolithic_cpc_model(config)
    solve_seconds = @elapsed optimize!(model)
    verification_seconds = @elapsed verification =
        verify_monolithic_solution(model, metadata, config)
    return (valid=verification.valid, config=config, estimate=estimate,
            model=model, metadata=metadata, build_seconds=build_seconds,
            solve_seconds=solve_seconds,
            verification_seconds=verification_seconds,
            verification=verification)
end

function main()
    rooms = isempty(ARGS) ? 10 : parse(Int, ARGS[1])
    degree = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
    alpha = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.01
    config = MonolithicCPCConfig(rooms=rooms, certificate_degree=degree,
                                 alpha=alpha)
    result = solve_monolithic_cpc(config)
    if !hasproperty(result, :verification)
        println("Monolithic benchmark not run: ", result.reason)
        exit(2)
    end
    verification = result.verification
    println("Monolithic CPC synthesis")
    println("  rooms / degree / alpha: ", (rooms, degree, alpha))
    @printf("  build / solve / verify: %.3f / %.3f / %.3f s\n",
            result.build_seconds, result.solve_seconds,
            result.verification_seconds)
    println("  status: ", verification.status,
            "; verified: ", verification.valid)
    if hasproperty(verification, :minimum_sampled_residual)
        println("  epsilon: ", verification.epsilon_local)
        println("  minimum sampled residual: ",
                verification.minimum_sampled_residual,
                " at ", verification.worst)
        println("  maximum conic violation: ",
                verification.maximum_conic_violation)
        println("  minimum Gram eigenvalue: ",
                verification.minimum_gram_eigenvalue)
    else
        println("  reason: ", verification.reason)
    end
    verification.valid || exit(2)
    output_path = joinpath(@__DIR__,
        "CPC_Monolithic_results_N$(rooms)_degree$(degree).txt")
    open(output_path, "w") do file
        println(file, "status: numerically verified monolithic CPC found")
        println(file, "rooms: ", rooms)
        println(file, "degree: ", degree)
        println(file, "alpha: ", alpha)
        println(file, "epsilon: ", verification.epsilon_local)
        println(file, "build_seconds: ", result.build_seconds)
        println(file, "solve_seconds: ", result.solve_seconds)
        println(file, "verification_seconds: ", result.verification_seconds)
        println(file, "minimum_sampled_residual: ",
                verification.minimum_sampled_residual)
        println(file, "maximum_conic_violation: ",
                verification.maximum_conic_violation)
        println(file, "minimum_gram_eigenvalue: ",
                verification.minimum_gram_eigenvalue)
        for (name, polynomial) in zip(("Bq0", "Bq1", "Vq0", "Vq1"),
                                      verification.polynomials)
            println(file, name, "(xhat): ", polynomial)
        end
    end
    println("Verified result written to ", output_path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
