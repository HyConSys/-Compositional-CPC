module CenteredCPCADMM

using JuMP
using MosekTools
using DynamicPolynomials
using MultivariatePolynomials
using LinearAlgebra
using Printf
using TSSOS

include(joinpath(@__DIR__, "Verify_Centered_Analytic_CPC.jl"))
using .VerifyCenteredAnalyticCPC

const MOI = JuMP.MOI
const CLOSED_LOOP_A = 0.5
const NEIGHBOR_COEFFICIENT = 0.1
const LOCAL_EPSILON = 1.0
const INPUT_LOWER = 0.0
const INPUT_UPPER = 3.0

Base.@kwdef struct CenteredADMMConfig
    ring_size::Int = 100
    certificate_degree::Int = 2
    alpha::Float64 = 0.01
    rho::Float64 = 1.0
    absolute_tolerance::Float64 = 1.0e-7
    relative_tolerance::Float64 = 1.0e-6
    composition_tolerance::Float64 = 1.0e-9
    solver_feasibility_tolerance::Float64 = 1.0e-9
    certificate_regularization::Float64 = 1.0e-8
    structured_template::Bool = true
    maximum_iterations::Int = 25
    minimum_iterations::Int = 1
    quiet::Bool = true
end

interval_polynomial(z, lower::Real, upper::Real) =
    (z - lower) * (upper - z)

supply_polynomial(P, v, z) =
    P[1, 1] * v^2 + 2P[1, 2] * v * z + P[2, 2] * z^2

ring_eigenvalues(n::Int) = begin
    n >= 3 || throw(ArgumentError("ring_size must be at least 3"))
    [2cos(2pi * k / n) for k in 0:(n - 1)]
end

function add_ring_constraints!(model, P, n::Int)
    for lambda in ring_eigenvalues(n)
        @constraint(model,
            P[1, 1] * lambda^2 + 2P[1, 2] * lambda + P[2, 2] <= 0)
    end
    return nothing
end

function composition_maximum(P::AbstractMatrix, n::Int)
    maximum(P[1, 1] * lambda^2 + 2P[1, 2] * lambda + P[2, 2]
            for lambda in ring_eigenvalues(n))
end

function analytic_supply_templates()
    exact = VerifyCenteredAnalyticCPC.exact_template()
    return (Matrix{Float64}(exact.Upsilon), Matrix{Float64}(exact.Gamma))
end

function set_symmetric_start!(P, start::AbstractMatrix)
    for (i, j) in ((1, 1), (1, 2), (2, 2))
        set_start_value(P[i, j], start[i, j])
    end
    return nothing
end

frobenius_distance_squared(P, target) =
    sum((P[i, j] - target[i, j])^2 for i in 1:2, j in 1:2)

tuple_norm(matrices) =
    sqrt(sum(sum(abs2, matrix) for matrix in matrices))

tuple_difference(left, right) =
    ntuple(index -> left[index] - right[index], length(left))

tuple_add(left, right) =
    ntuple(index -> left[index] + right[index], length(left))

tuple_subtract(left, right) =
    ntuple(index -> left[index] - right[index], length(left))

tuple_scaled_add(left, scale::Real, right) =
    ntuple(index -> left[index] + scale * right[index], length(left))

function configure_mosek!(model, config::CenteredADMMConfig)
    config.quiet && set_silent(model)
    tolerance = config.solver_feasibility_tolerance
    tolerance > 0 || throw(ArgumentError("solver tolerance must be positive"))
    for attribute in ("MSK_DPAR_INTPNT_CO_TOL_PFEAS",
                      "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
                      "MSK_DPAR_INTPNT_CO_TOL_INFEAS")
        set_optimizer_attribute(model, attribute, tolerance)
    end
    return model
end

function add_psatz_constraint!(infos, model, polynomial, variables, domain,
                               order)
    info = add_psatz!(model, polynomial, variables, domain, [], order;
        QUIET=true, CS=false, TS=false, GroebnerBasis=false)
    push!(infos, info)
    return info
end

function coefficient_target(basis, variable; q0::Bool)
    power = MultivariatePolynomials.degree(basis, variable)
    power == 2 && return 2.0
    q0 && power == 0 && return 1.0
    return 0.0
end

"Build the local CPC proximal step, without imposing the composition LMI."
function build_local_step(config::CenteredADMMConfig, consensus, scaled_dual)
    degree = config.certificate_degree
    degree >= 2 || throw(ArgumentError("certificate_degree must be at least 2"))
    config.rho > 0 || throw(ArgumentError("rho must be positive"))

    model = configure_mosek!(Model(Mosek.Optimizer), config)
    @polyvar z v
    znext = CLOSED_LOOP_A * z + NEIGHBOR_COEFFICIENT * v

    Vq0, Vq0_coefficients, Vq0_basis = add_poly!(model, [z], degree)
    Vq1, Vq1_coefficients, Vq1_basis = add_poly!(model, [z], degree)
    Vq0_next = Vq0([z] => [znext])
    Vq1_next = Vq1([z] => [znext])

    @variable(model, Upsilon[1:2, 1:2], Symmetric)
    @variable(model, Gamma[1:2, 1:2], Symmetric)
    supplies = (Upsilon, Gamma)

    proximal_targets = tuple_subtract(consensus, scaled_dual)
    for index in eachindex(supplies)
        set_symmetric_start!(supplies[index], proximal_targets[index])
    end
    for (coefficient, basis) in zip(Vq0_coefficients, Vq0_basis)
        set_start_value(coefficient, coefficient_target(basis, z; q0=true))
    end
    for (coefficient, basis) in zip(Vq1_coefficients, Vq1_basis)
        set_start_value(coefficient, coefficient_target(basis, z; q0=false))
    end

    state_domain = [interval_polynomial(z, -10.0, 5.0)]
    input_domain = interval_polynomial(v, -20.0, 10.0)
    good_domain = [interval_polynomial(z, -2.0, 1.0), input_domain]
    bad_low_domain = [interval_polynomial(z, -10.0, -2.0), input_domain]
    bad_high_domain = [interval_polynomial(z, 1.0, 5.0), input_domain]
    regions = ((:good, good_domain), (:bad_low, bad_low_domain),
               (:bad_high, bad_high_domain))

    order = cld(max(degree, 2), 2)
    infos = Any[]
    add_psatz_constraint!(infos, model, Vq0, [z], state_domain, order)
    add_psatz_constraint!(infos, model, Vq1, [z], state_domain, order)

    U = supply_polynomial(Upsilon, v, z)
    G = supply_polynomial(Gamma, v, z)
    residuals = Dict{String,Any}()
    for (label, domain) in regions
        successor_V = label == :good ? Vq1_next : Vq0_next
        barrier_residual = U # Bq0=Bq1=0 exactly.
        q0_rank_residual = -successor_V + Vq0 - LOCAL_EPSILON + G
        q1_rank_residual = -successor_V + Vq1 + G
        for (name, residual) in
            (("$(label)_B", barrier_residual),
             ("$(label)_q0_V", q0_rank_residual),
             ("$(label)_q1_V", q1_rank_residual))
            residuals[name] = residual
            add_psatz_constraint!(infos, model, residual, [z, v], domain,
                                  order)
        end
    end

    coefficient_cost = 0.0
    for (coefficient, basis) in zip(Vq0_coefficients, Vq0_basis)
        target = coefficient_target(basis, z; q0=true)
        coefficient_cost += (coefficient - target)^2
    end
    for (coefficient, basis) in zip(Vq1_coefficients, Vq1_basis)
        target = coefficient_target(basis, z; q0=false)
        coefficient_cost += (coefficient - target)^2
    end
    proximal_cost = sum(frobenius_distance_squared(supplies[index],
                                                   proximal_targets[index])
                          for index in eachindex(supplies))
    @objective(model, Min,
        config.rho / 2 * proximal_cost +
        config.certificate_regularization * coefficient_cost)

    metadata = (supplies=supplies,
                coefficients=(Vq0_coefficients, Vq1_coefficients),
                bases=(Vq0_basis, Vq1_basis),
                polynomials=(Vq0=Vq0, Vq1=Vq1),
                residuals=residuals,
                infos=infos)
    return model, metadata
end

function acceptable_status(model)
    termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL,
                                  MOI.SLOW_PROGRESS) && has_values(model)
end

function solve_generic_local_step(config, consensus, scaled_dual)
    model, metadata = build_local_step(config, consensus, scaled_dual)
    solve_seconds = @elapsed optimize!(model)
    acceptable_status(model) || error(
        "local SOS step failed: $(termination_status(model)); $(raw_status(model))")
    supplies = ntuple(index -> Matrix(value.(metadata.supplies[index])), 2)
    coefficients = ntuple(index -> value.(metadata.coefficients[index]), 2)
    return (supplies=supplies, coefficients=coefficients, model=model,
            metadata=metadata, solve_seconds=solve_seconds,
            status=termination_status(model), raw_status=raw_status(model))
end

function structured_gamma(kappa::Real, c::Real)
    a, eta = CLOSED_LOOP_A, NEIGHBOR_COEFFICIENT
    return [kappa * eta^2 kappa * a * eta;
            kappa * a * eta kappa * (a^2 - 1) + c]
end

function structured_local_objective(theta, target, config)
    kappa, c = theta
    gamma = structured_gamma(kappa, c)
    certificate_penalty = (kappa - 2.0)^2 + (c - 1.0)^2
    return config.rho / 2 * sum(abs2, gamma - target) +
           config.certificate_regularization * certificate_penalty
end

"Exact two-variable proximal solve on the facially reduced SOS template."
function solve_structured_local_step(config, consensus, scaled_dual)
    started = time_ns()
    targets = tuple_subtract(consensus, scaled_dual)
    gamma_target = targets[2]

    # In Frobenius coordinates vec_F(Gamma)=A*[kappa,c].  The local step is
    # a strictly convex two-variable QP with kappa>=0 and c>=1.  Enumerating
    # the two possible active constraints avoids another numerical SDP/QP.
    a, eta = CLOSED_LOOP_A, NEIGHBOR_COEFFICIENT
    A = [eta^2 0.0;
         sqrt(2.0) * a * eta 0.0;
         a^2 - 1.0 1.0]
    target = frobenius_coordinates(gamma_target)
    regularization = config.certificate_regularization
    H = config.rho * (transpose(A) * A) + 2regularization * I(2)
    rhs = config.rho * transpose(A) * target +
          2regularization * [2.0, 1.0]

    candidates = Vector{Vector{Float64}}()
    unconstrained = H \ rhs
    unconstrained[1] >= 0 && unconstrained[2] >= 1 &&
        push!(candidates, collect(unconstrained))

    # Active kappa=0.
    c_only = (config.rho * dot(A[:, 2], target) + 2regularization) /
             (config.rho * dot(A[:, 2], A[:, 2]) + 2regularization)
    push!(candidates, [0.0, max(1.0, c_only)])

    # Active c=1.
    target_without_c = target - A[:, 2]
    kappa_only = (config.rho * dot(A[:, 1], target_without_c) +
                  4regularization) /
                 (config.rho * dot(A[:, 1], A[:, 1]) + 2regularization)
    push!(candidates, [max(0.0, kappa_only), 1.0])
    push!(candidates, [0.0, 1.0])

    objectives = [structured_local_objective(candidate, gamma_target, config)
                  for candidate in candidates]
    theta = candidates[argmin(objectives)]
    kappa, c = theta
    kappa >= 0 || error("structured local solve violated kappa>=0")
    c >= 1 || error("structured local solve violated c>=1")

    upsilon = zeros(2, 2)
    gamma = structured_gamma(kappa, c)
    Vq0_coefficients = zeros(config.certificate_degree + 1)
    Vq1_coefficients = zeros(config.certificate_degree + 1)
    Vq0_coefficients[1] = 1.0
    Vq0_coefficients[3] = kappa
    Vq1_coefficients[3] = kappa
    coefficients = (Vq0_coefficients, Vq1_coefficients)
    solve_seconds = (time_ns() - started) / 1.0e9
    metadata = (structured=true, kappa=kappa, c=c,
                local_residuals=(good="$(c) z^2",
                                 bad="$(c) z^2-1"))
    return (supplies=(upsilon, gamma), coefficients=coefficients,
            model=nothing, metadata=metadata, solve_seconds=solve_seconds,
            status=:EXACT_STRUCTURED_QP,
            raw_status="facially reduced analytic local SOS cone")
end

function solve_local_step(config, consensus, scaled_dual)
    return config.structured_template ?
        solve_structured_local_step(config, consensus, scaled_dual) :
        solve_generic_local_step(config, consensus, scaled_dual)
end

function frobenius_coordinates(P::AbstractMatrix)
    # ||P||_F^2 = a^2+2b^2+c^2 for P=[a b;b c].  The sqrt(2) coordinate
    # converts the projection to an ordinary Euclidean projection in R^3.
    return [P[1, 1], sqrt(2.0) * P[1, 2], P[2, 2]]
end

function matrix_from_frobenius_coordinates(coordinates)
    a, scaled_b, c = coordinates
    b = scaled_b / sqrt(2.0)
    return [a b; b c]
end

"Project one symmetric supply matrix onto all ring halfspaces with Dykstra."
function project_ring_matrix(P::AbstractMatrix, ring_size::Int;
                             tolerance=1.0e-13,
                             maximum_sweeps=100_000)
    lambdas = ring_eigenvalues(ring_size)
    normals = [[lambda^2, sqrt(2.0) * lambda, 1.0]
               for lambda in lambdas]
    coordinates = frobenius_coordinates(P)
    violations = [dot(normal, coordinates) for normal in normals]
    if maximum(violations) <= tolerance
        return (matrix=Matrix{Float64}(P), sweeps=0,
                maximum_violation=maximum(violations), converged=true)
    end

    corrections = [zeros(3) for _ in normals]
    for sweep in 1:maximum_sweeps
        previous = copy(coordinates)
        for index in eachindex(normals)
            normal = normals[index]
            shifted = coordinates + corrections[index]
            violation = dot(normal, shifted)
            projected = violation > 0 ?
                shifted - (violation / dot(normal, normal)) * normal : shifted
            corrections[index] = shifted - projected
            coordinates = projected
        end
        maximum_violation = maximum(dot(normal, coordinates)
                                    for normal in normals)
        change = norm(coordinates - previous)
        if maximum_violation <= tolerance &&
           change <= tolerance * (1 + norm(coordinates))
            return (matrix=matrix_from_frobenius_coordinates(coordinates),
                    sweeps=sweep,
                    maximum_violation=maximum_violation,
                    converged=true)
        end
    end
    maximum_violation = maximum(dot(normal, coordinates) for normal in normals)
    return (matrix=matrix_from_frobenius_coordinates(coordinates),
            sweeps=maximum_sweeps,
            maximum_violation=maximum_violation,
            converged=false)
end

"Euclidean projection of both supplies onto the homogeneous ring cone."
function project_composition(config::CenteredADMMConfig, targets;
                             starts=targets)
    # `starts` is retained in the interface because it is part of the ADMM
    # state, but Dykstra's exact projection starts from the projection target.
    # Unlike an interior-point QP, it returns an already feasible target
    # unchanged, which is essential on the semidefinite Upsilon=0 face.
    projections = nothing
    solve_seconds = @elapsed projections = ntuple(index ->
        project_ring_matrix(targets[index], config.ring_size), 2)
    all(item.converged for item in projections) ||
        error("composition projection did not converge")
    values = ntuple(index -> projections[index].matrix, 2)
    maxima = ntuple(index -> composition_maximum(values[index],
                                                  config.ring_size), 2)
    return (supplies=values, maxima=maxima, model=nothing,
            solve_seconds=solve_seconds, status=:PROJECTED,
            raw_status="Dykstra projection in Frobenius coordinates",
            sweeps=ntuple(index -> projections[index].sweeps, 2))
end

function run_admm(config::CenteredADMMConfig=CenteredADMMConfig();
                  initial_consensus=nothing, verbose=true)
    config.maximum_iterations >= config.minimum_iterations >= 1 ||
        throw(ArgumentError("iteration bounds must satisfy 1 <= minimum <= maximum"))
    template = analytic_supply_templates()
    consensus = isnothing(initial_consensus) ?
        ntuple(index -> copy(template[index]), 2) :
        ntuple(index -> Matrix{Float64}(initial_consensus[index]), 2)
    scaled_dual = (zeros(2, 2), zeros(2, 2))
    history = NamedTuple[]
    local_result = nothing

    for iteration in 1:config.maximum_iterations
        local_result = solve_local_step(config, consensus, scaled_dual)
        local_supplies = local_result.supplies
        previous_consensus = consensus
        projection_target = tuple_add(local_supplies, scaled_dual)
        global_result = project_composition(config, projection_target;
                                            starts=previous_consensus)
        consensus = global_result.supplies
        primal_difference = tuple_difference(local_supplies, consensus)
        scaled_dual = tuple_add(scaled_dual, primal_difference)

        primal_residual = tuple_norm(primal_difference)
        dual_residual = config.rho * tuple_norm(
            tuple_difference(consensus, previous_consensus))
        ambient_dimension = 8 # two 2-by-2 matrices in Frobenius coordinates
        epsilon_primal = sqrt(ambient_dimension) * config.absolute_tolerance +
            config.relative_tolerance * max(tuple_norm(local_supplies),
                                            tuple_norm(consensus))
        epsilon_dual = sqrt(ambient_dimension) * config.absolute_tolerance +
            config.relative_tolerance * config.rho * tuple_norm(scaled_dual)
        maximum_composition_violation = max(0.0, maximum(global_result.maxima))
        template_deviation = tuple_norm(tuple_difference(consensus, template))

        record = (iteration=iteration,
                  primal_residual=primal_residual,
                  dual_residual=dual_residual,
                  epsilon_primal=epsilon_primal,
                  epsilon_dual=epsilon_dual,
                  composition_maxima=global_result.maxima,
                  maximum_composition_violation=maximum_composition_violation,
                  template_deviation=template_deviation,
                  local_solve_seconds=local_result.solve_seconds,
                  global_solve_seconds=global_result.solve_seconds)
        push!(history, record)
        if verbose
            @printf("ADMM %2d: r=%.3e (<=%.3e), s=%.3e (<=%.3e), comp=%.3e, template=%.3e\n",
                    iteration, primal_residual, epsilon_primal,
                    dual_residual, epsilon_dual,
                    maximum_composition_violation, template_deviation)
        end

        converged = iteration >= config.minimum_iterations &&
            primal_residual <= epsilon_primal &&
            dual_residual <= epsilon_dual &&
            maximum_composition_violation <= config.composition_tolerance
        if converged
            return (converged=true, config=config, template=template,
                    local_supplies=local_supplies, consensus=consensus,
                    scaled_dual=scaled_dual, local_result=local_result,
                    history=history)
        end
    end

    return (converged=false, config=config, template=template,
            local_supplies=local_result.supplies, consensus=consensus,
            scaled_dual=scaled_dual, local_result=local_result,
            history=history)
end

function projection_self_test(config::CenteredADMMConfig)
    template = analytic_supply_templates()
    feasible_projection = project_composition(config, template)
    feasible_error = tuple_norm(tuple_difference(feasible_projection.supplies,
                                                  template))

    perturbed = (template[1] + [0.0 0.0; 0.0 0.01],
                 template[2] + [0.0 0.0; 0.0 0.01])
    before = maximum(composition_maximum(P, config.ring_size)
                     for P in perturbed)
    corrected = project_composition(config, perturbed; starts=template)
    after = maximum(corrected.maxima)
    before > 0 || error("projection self-test perturbation was not infeasible")
    after <= config.composition_tolerance || error(
        "composition projector retained violation $after")
    @printf("Projection self-test: feasible error %.3e; violation %.3e -> %.3e\n",
            feasible_error, before, after)
    return (valid=true, feasible_error=feasible_error,
            violation_before=before, violation_after=after)
end

function admm_refinement_self_test(config::CenteredADMMConfig)
    template = analytic_supply_templates()
    initial = (template[1] + [0.0 0.0; 0.0 0.01],
               template[2] + [0.0 0.0; 0.0 0.03])
    initial_maxima = ntuple(index ->
        composition_maximum(initial[index], config.ring_size), 2)
    maximum(initial_maxima) > 0 ||
        error("ADMM refinement test did not start outside the composition cone")

    test_config = CenteredADMMConfig(
        ring_size=config.ring_size,
        certificate_degree=config.certificate_degree,
        alpha=config.alpha,
        rho=config.rho,
        absolute_tolerance=config.absolute_tolerance,
        relative_tolerance=config.relative_tolerance,
        composition_tolerance=config.composition_tolerance,
        solver_feasibility_tolerance=config.solver_feasibility_tolerance,
        certificate_regularization=config.certificate_regularization,
        structured_template=config.structured_template,
        maximum_iterations=max(config.maximum_iterations, 250),
        minimum_iterations=config.minimum_iterations,
        quiet=config.quiet)
    result = run_admm(test_config; initial_consensus=initial, verbose=false)
    result.converged || error("ADMM did not repair the infeasible warm start")
    final_record = last(result.history)
    final_record.maximum_composition_violation <=
        config.composition_tolerance ||
        error("ADMM refinement retained a composition violation")
    @printf("ADMM refinement self-test: violation %.3e -> %.3e in %d iterations\n",
            maximum(initial_maxima),
            final_record.maximum_composition_violation,
            length(result.history))
    return (valid=true, initial_composition_maxima=initial_maxima,
            iterations=length(result.history), final_record=final_record,
            final_consensus=result.consensus,
            final_local_parameters=result.local_result.metadata)
end

function write_results(path, result, exact_result, projection_test,
                       refinement_test)
    open(path, "w") do io
        println(io, "Centered constant-u=2 CPC and corrected ADMM")
        println(io, "exact_template_valid: ", exact_result.valid)
        println(io, "admm_converged: ", result.converged)
        println(io, "ring_size: ", result.config.ring_size)
        println(io, "certificate_degree: ", result.config.certificate_degree)
        println(io, "alpha: ", result.config.alpha)
        println(io, "input_set: [", INPUT_LOWER, ", ", INPUT_UPPER, "]")
        println(io, "input_set_role: physical actuator set; 0 means heater off")
        println(io, "state_certificate_coordinates: physical x through z=x-25")
        println(io, "supply_coordinates: [v,z]=[w-50,x-25]")
        println(io, "physical_augmented_Gamma: ",
                exact_result.physical_supply_matrix)
        println(io, "eta: ", NEIGHBOR_COEFFICIENT)
        println(io, "iterations: ", length(result.history))
        println(io, "projection_self_test: ", projection_test)
        println(io, "admm_refinement_self_test: ", refinement_test)
        println(io, "Upsilon_template: ", result.template[1])
        println(io, "Gamma_template: ", result.template[2])
        println(io, "Upsilon_local: ", result.local_supplies[1])
        println(io, "Gamma_local: ", result.local_supplies[2])
        println(io, "Upsilon_consensus: ", result.consensus[1])
        println(io, "Gamma_consensus: ", result.consensus[2])
        println(io, "Vq0_coefficients: ", result.local_result.coefficients[1])
        println(io, "Vq1_coefficients: ", result.local_result.coefficients[2])
        println(io, "history:")
        for row in result.history
            println(io, row)
        end
        println(io, "IMPORTANT: the exact rational template and explicit SOS ",
                    "decompositions are the certificate; the floating-point ",
                    "ADMM iterate is a numerical reproduction only.")
    end
    return path
end

function main()
    ring_size = isempty(ARGS) ? 100 : parse(Int, ARGS[1])
    degree = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
    config = CenteredADMMConfig(ring_size=ring_size,
                                certificate_degree=degree)
    println("Centered CPC: u=2, z=x-25, v=w-50, N=", ring_size,
            ", degree=", degree)
    exact_result = VerifyCenteredAnalyticCPC.verify_exact_template()
    projection_test = projection_self_test(config)
    result = run_admm(config)
    refinement_test = admm_refinement_self_test(config)
    output_path = joinpath(@__DIR__,
        "CPC_Centered_ADMM_results_N$(ring_size)_degree$(degree).txt")
    write_results(output_path, result, exact_result, projection_test,
                  refinement_test)
    println("ADMM converged: ", result.converged)
    println("Results written to ", output_path)
    result.converged || exit(2)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end

end # module
