using JuMP
using MosekTools
using DynamicPolynomials
using MultivariatePolynomials
using LinearAlgebra
using Printf
using TSSOS

const MOI = JuMP.MOI

Base.@kwdef struct LocalCPCConfig
    certificate_degree::Int = 4
    ring_size::Int = 100
    fix_supply_matrices::Bool = false
    common_supply_across_states::Bool = true
    projected_local_pa::Bool = true
    alpha::Float64 = 0.1
    epsilon_lower_bound::Float64 = 1.0e-6
    epsilon_upper_bound::Float64 = 0.1
    epsilon_reward::Float64 = 1.0e-3
    verification_tolerance::Float64 = 1.0e-7
    coefficient_regularization::Float64 = 1.0e-4
    quiet::Bool = true
end

const ETA = 0.001
const EXTERIOR_COEFFICIENT = 0.05
const HEATER_COEFFICIENT = 0.125
const EXTERIOR_TEMPERATURE = 0.0
const HEATER_TEMPERATURE = 30.0
const ALPHA_CPC = 0.1
# Multiplying B, V, Gamma, Upsilon, and epsilon by a common positive
# constant preserves every local CPC inequality and the global LMI.  The
# original data are divided by this factor to keep the SDP near unit scale.
const CPC_NUMERICAL_SCALE = 1000.0
const EPSILON_LOCAL_UNSCALED = 100.0
const EPSILON_LOCAL = EPSILON_LOCAL_UNSCALED / CPC_NUMERICAL_SCALE

const X_CENTER = 22.5
const X_SCALE = 7.5
const W_CENTER = 45.0
const W_SCALE = 15.0

const UPSILON_Q0_REFERENCE = [-1.8 -0.3; -0.3 -2.6] ./ CPC_NUMERICAL_SCALE
const UPSILON_Q1_REFERENCE = [-1.8 -0.3; -0.3 -2.6] ./ CPC_NUMERICAL_SCALE
const GAMMA_Q0_REFERENCE = [0.0 0.0; 0.0 -1.4] ./ CPC_NUMERICAL_SCALE
const GAMMA_Q1_REFERENCE = [0.2 0.1; 0.1 -1.3] ./ CPC_NUMERICAL_SCALE

normalized_x(x::Real) = (x - X_CENTER) / X_SCALE
physical_x(xhat) = X_CENTER + X_SCALE * xhat
physical_w(what) = W_CENTER + W_SCALE * what

function controller(x)
    return -0.01 * x + 2.0
end

function room_dynamics(x, w)
    u = controller(x)
    return (1.0 - 2ETA - EXTERIOR_COEFFICIENT - HEATER_COEFFICIENT * u) * x +
           HEATER_COEFFICIENT * HEATER_TEMPERATURE * u + ETA * w +
           EXTERIOR_COEFFICIENT * EXTERIOR_TEMPERATURE
end

interval_polynomial(z, lower::Real, upper::Real) = (z - lower) * (upper - z)

function supply_polynomial(P, w, x)
    return P[1, 1] * w^2 + 2P[1, 2] * w * x + P[2, 2] * x^2
end

function numeric_polynomial(coefficients_ref, basis)
    return sum(value(coefficients_ref[i]) * basis[i] for i in eachindex(coefficients_ref))
end

function polynomial_term_data(polynomial, variables)
    coefficients = Float64.(MultivariatePolynomials.coefficients(polynomial))
    monomial_list = MultivariatePolynomials.monomials(polynomial)
    exponents = [MultivariatePolynomials.degree(monomial, variable)
                 for monomial in monomial_list, variable in variables]
    return coefficients, exponents
end

function evaluate_term_data(term_data, point)
    coefficients, exponents = term_data
    total = 0.0
    for term in eachindex(coefficients)
        monomial_value = 1.0
        for variable in eachindex(point)
            monomial_value *= point[variable]^exponents[term, variable]
        end
        total += coefficients[term] * monomial_value
    end
    return total
end

function ring_eigenvalues(n::Int)
    n >= 3 || throw(ArgumentError("ring_size must be at least 3"))
    return [2cos(2pi * k / n) for k in 0:(n - 1)]
end

function add_ring_lmi_constraints!(model, P, n::Int)
    # For a symmetric ring adjacency M, M has eigenvalues 2*cos(2*pi*k/n).
    # Hence [M;I]'*(I kron P)*[M;I] <= 0 is equivalent to the scalar
    # inequalities below at every eigenvalue.  This is deliberately
    # semidefinite rather than artificially strict.
    for lambda in ring_eigenvalues(n)
        @constraint(model,
            P[1, 1] * lambda^2 + 2P[1, 2] * lambda + P[2, 2] <= 0.0)
    end
    return nothing
end

function add_psatz_constraint!(infos, model, polynomial, variables, domain, order)
    push!(infos, add_psatz!(model, polynomial, variables, domain, [], order;
        QUIET=true, CS=false, TS=false, GroebnerBasis=false))
    return nothing
end

function build_local_cpc_model(config::LocalCPCConfig)
    degree = config.certificate_degree
    iseven(degree) || throw(ArgumentError("certificate_degree must be even"))
    degree >= 2 || throw(ArgumentError("certificate_degree must be at least 2"))

    model = Model(Mosek.Optimizer)
    config.quiet && set_silent(model)

    @polyvar xhat what
    x = physical_x(xhat)
    w = physical_w(what)
    xhat_next = (room_dynamics(x, w) - X_CENTER) / X_SCALE

    Bq0, Bq0_coefficients, Bq0_basis = add_poly!(model, [xhat], degree)
    Bq1, Bq1_coefficients, Bq1_basis = add_poly!(model, [xhat], degree)
    Vq0, Vq0_coefficients, Vq0_basis = add_poly!(model, [xhat], degree)
    Vq1, Vq1_coefficients, Vq1_basis = add_poly!(model, [xhat], degree)

    Bq0_next = Bq0([xhat] => [xhat_next])
    Bq1_next = Bq1([xhat] => [xhat_next])
    Vq0_next = Vq0([xhat] => [xhat_next])
    Vq1_next = Vq1([xhat] => [xhat_next])

    config.epsilon_upper_bound >= config.epsilon_lower_bound > 0 ||
        throw(ArgumentError("epsilon bounds must satisfy 0 < lower <= upper"))
    @variable(model,
        config.epsilon_lower_bound <= epsilon_local <= config.epsilon_upper_bound)
    @variable(model, Upsilon_q0[1:2, 1:2], Symmetric)
    @variable(model, Upsilon_q1[1:2, 1:2], Symmetric)
    @variable(model, Gamma_q0[1:2, 1:2], Symmetric)
    @variable(model, Gamma_q1[1:2, 1:2], Symmetric)

    supplies = (Upsilon_q0, Upsilon_q1, Gamma_q0, Gamma_q1)
    references = (UPSILON_Q0_REFERENCE, UPSILON_Q1_REFERENCE,
                  GAMMA_Q0_REFERENCE, GAMMA_Q1_REFERENCE)

    config.fix_supply_matrices && config.common_supply_across_states &&
        throw(ArgumentError("the reference q0/q1 Gamma matrices differ; use optimized supplies when common_supply_across_states=true"))
    if config.common_supply_across_states
        @constraint(model, Upsilon_q0 .== Upsilon_q1)
        @constraint(model, Gamma_q0 .== Gamma_q1)
    end

    if config.fix_supply_matrices
        for (P, reference) in zip(supplies, references)
            @constraint(model, P .== reference)
        end
    else
        for P in supplies
            # The interconnection condition is semidefinite, with no
            # artificial strict margin.  A positive margin is incompatible
            # with the equilibrium contained in the labelled regions.
            add_ring_lmi_constraints!(model, P, config.ring_size)
        end
    end

    Upsilon0 = supply_polynomial(Upsilon_q0, w, x)
    Upsilon1 = supply_polynomial(Upsilon_q1, w, x)
    Gamma0 = supply_polynomial(Gamma_q0, w, x)
    Gamma1 = supply_polynomial(Gamma_q1, w, x)

    x0_lower, x0_upper = normalized_x(20.0), normalized_x(25.0)
    xa_lower, xa_upper = normalized_x(23.0), normalized_x(26.0)
    xb1_lower, xb1_upper = normalized_x(15.0), normalized_x(23.0)
    xb2_lower, xb2_upper = normalized_x(26.0), normalized_x(30.0)

    domain_x0 = [interval_polynomial(xhat, x0_lower, x0_upper)]
    domain_x = [1.0 - xhat^2]
    domain_xa = [interval_polynomial(xhat, xa_lower, xa_upper), 1.0 - what^2]
    domain_xb1 = [interval_polynomial(xhat, xb1_lower, xb1_upper), 1.0 - what^2]
    domain_xb2 = [interval_polynomial(xhat, xb2_lower, xb2_upper), 1.0 - what^2]

    scalar_order = cld(degree, 2)
    transition_order = degree # deg(B(f)) = 2*degree for the quadratic closed loop.
    infos = Any[]

    # Initial and lower-bound conditions.
    add_psatz_constraint!(infos, model, -Bq0, [xhat], domain_x0,
                          scalar_order)
    add_psatz_constraint!(infos, model, Vq0 + config.alpha * Bq0,
                          [xhat], domain_x, scalar_order)
    add_psatz_constraint!(infos, model, Vq1 + config.alpha * Bq1,
                          [xhat], domain_x, scalar_order)

    transition_domains = (("a", domain_xa), ("b_low", domain_xb1),
                          ("b_high", domain_xb2))
    transition_residuals = Dict{String, Any}()

    for (label, domain) in transition_domains
        # A local b letter forces the global b transition.  A local a letter
        # is compatible with either global a or global b, so the projected
        # local PA has both q0 and q1 as possible successors.  Enforcing each
        # successor is equivalent to the maxima in Definition 6.
        successors = label == "a" && config.projected_local_pa ?
            (("q0", Bq0_next, Vq0_next), ("q1", Bq1_next, Vq1_next)) :
            label == "a" ? (("q1", Bq1_next, Vq1_next),) :
            (("q0", Bq0_next, Vq0_next),)
        for (successor_name, B_successor, V_successor) in successors
            residual_B_q0 = -B_successor + config.alpha * Bq0 + Upsilon0
            residual_V_q0 = -V_successor + Vq0 + config.alpha * Bq0 -
                            epsilon_local + Gamma0
            residual_B_q1 = -B_successor + config.alpha * Bq1 + Upsilon1
            residual_V_q1 = -V_successor + Vq1 + config.alpha * Bq1 + Gamma1

            residual_names = ("$(label)_q0_to_$(successor_name)_B",
                              "$(label)_q0_to_$(successor_name)_V",
                              "$(label)_q1_to_$(successor_name)_B",
                              "$(label)_q1_to_$(successor_name)_V")
            for (name, residual) in zip(residual_names,
                                        (residual_B_q0, residual_V_q0,
                                         residual_B_q1, residual_V_q1))
                transition_residuals[name] = residual
                add_psatz_constraint!(infos, model, residual,
                                      [xhat, what], domain, transition_order)
            end
        end
    end

    all_coefficients = vcat(Bq0_coefficients, Bq1_coefficients,
                            Vq0_coefficients, Vq1_coefficients)
    objective = config.coefficient_regularization * sum(c^2 for c in all_coefficients) -
                config.epsilon_reward * epsilon_local
    if !config.fix_supply_matrices
        objective += sum((P[i, j] - reference[i, j])^2
                         for (P, reference) in zip(supplies, references)
                         for i in 1:2 for j in 1:2)
    end
    @objective(model, Min, objective)

    metadata = (
        xhat=xhat,
        what=what,
        x=x,
        w=w,
        xhat_next=xhat_next,
        polynomials=(Bq0, Bq1, Vq0, Vq1),
        coefficient_refs=(Bq0_coefficients, Bq1_coefficients,
                          Vq0_coefficients, Vq1_coefficients),
        bases=(Bq0_basis, Bq1_basis, Vq0_basis, Vq1_basis),
        supplies=supplies,
        epsilon_local=epsilon_local,
        infos=infos,
        transition_residuals=transition_residuals,
        projected_local_pa=config.projected_local_pa,
        domains=(
            x0=(x0_lower, x0_upper),
            x=(-1.0, 1.0),
            a=(xa_lower, xa_upper, -1.0, 1.0),
            b_low=(xb1_lower, xb1_upper, -1.0, 1.0),
            b_high=(xb2_lower, xb2_upper, -1.0, 1.0),
        ),
    )
    return model, metadata
end

function minimum_gram_eigenvalue(infos)
    minimum_eigenvalue = Inf
    matrix_count = 0
    for info in infos, clique in info.GramMat, constraint_blocks in clique,
        gram_matrix in constraint_blocks
        matrix_count += 1
        eigenvalue = minimum(eigvals(Symmetric(value.(gram_matrix))))
        minimum_eigenvalue = min(minimum_eigenvalue, eigenvalue)
    end
    return minimum_eigenvalue, matrix_count
end

function evaluate_polynomial_grid(polynomial, variables, bounds; points_per_axis=301)
    if length(variables) == 1
        lower, upper = bounds
        best = (value=Inf, point=(NaN,))
        for x in range(lower, upper; length=points_per_axis)
            value_at_point = polynomial(variables => [x])
            if value_at_point < best.value
                best = (value=Float64(value_at_point), point=(Float64(x),))
            end
        end
        return best
    elseif length(variables) == 2
        xlower, xupper, wlower, wupper = bounds
        best = (value=Inf, point=(NaN, NaN))
        for x in range(xlower, xupper; length=points_per_axis),
            w in range(wlower, wupper; length=points_per_axis)
            value_at_point = polynomial(variables => [x, w])
            if value_at_point < best.value
                best = (value=Float64(value_at_point),
                        point=(Float64(x), Float64(w)))
            end
        end
        return best
    end
    throw(ArgumentError("grid verification supports one or two variables"))
end

function supply_lmi_margin(P::AbstractMatrix, n::Int)
    return -maximum(P[1, 1] * lambda^2 + 2P[1, 2] * lambda + P[2, 2]
                    for lambda in ring_eigenvalues(n))
end

function verify_local_solution(model, metadata, config::LocalCPCConfig;
                               points_per_axis=301)
    status = termination_status(model)
    primal = primal_status(model)
    if !(status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)) || !has_values(model)
        return (valid=false, status=status, primal=primal,
                reason="the optimization problem did not return a usable solution")
    end

    Bq0, Bq1, Vq0, Vq1 = map(numeric_polynomial,
        metadata.coefficient_refs, metadata.bases)
    Upsilon_q0, Upsilon_q1, Gamma_q0, Gamma_q1 =
        map(P -> value.(P), metadata.supplies)
    epsilon_local = value(metadata.epsilon_local)

    xhat, what = metadata.xhat, metadata.what
    x, w, xhat_next = metadata.x, metadata.w, metadata.xhat_next
    Bq0_next = Bq0([xhat] => [xhat_next])
    Bq1_next = Bq1([xhat] => [xhat_next])
    Vq0_next = Vq0([xhat] => [xhat_next])
    Vq1_next = Vq1([xhat] => [xhat_next])
    Upsilon0 = supply_polynomial(Upsilon_q0, w, x)
    Upsilon1 = supply_polynomial(Upsilon_q1, w, x)
    Gamma0 = supply_polynomial(Gamma_q0, w, x)
    Gamma1 = supply_polynomial(Gamma_q1, w, x)

    residuals = Vector{NamedTuple}()
    push!(residuals, (name="initial_B_q0", polynomial=-Bq0,
                      variables=[xhat], bounds=metadata.domains.x0))
    push!(residuals, (name="lower_bound_q0",
                      polynomial=Vq0 + config.alpha * Bq0,
                      variables=[xhat], bounds=metadata.domains.x))
    push!(residuals, (name="lower_bound_q1",
                      polynomial=Vq1 + config.alpha * Bq1,
                      variables=[xhat], bounds=metadata.domains.x))

    for label in ("a", "b_low", "b_high")
        domain = getproperty(metadata.domains, Symbol(label))
        successors = label == "a" && config.projected_local_pa ?
            (("q0", Bq0_next, Vq0_next), ("q1", Bq1_next, Vq1_next)) :
            label == "a" ? (("q1", Bq1_next, Vq1_next),) :
            (("q0", Bq0_next, Vq0_next),)
        for (successor_name, Bn, Vn) in successors
            append!(residuals, [
                (name="$(label)_q0_to_$(successor_name)_B",
                 polynomial=-Bn + config.alpha * Bq0 + Upsilon0,
                 variables=[xhat, what], bounds=domain),
                (name="$(label)_q0_to_$(successor_name)_V",
                 polynomial=-Vn + Vq0 + config.alpha * Bq0 - epsilon_local + Gamma0,
                 variables=[xhat, what], bounds=domain),
                (name="$(label)_q1_to_$(successor_name)_B",
                 polynomial=-Bn + config.alpha * Bq1 + Upsilon1,
                 variables=[xhat, what], bounds=domain),
                (name="$(label)_q1_to_$(successor_name)_V",
                 polynomial=-Vn + Vq1 + config.alpha * Bq1 + Gamma1,
                 variables=[xhat, what], bounds=domain),
            ])
        end
    end

    grid_results = [(name=item.name,
                     evaluate_polynomial_grid(item.polynomial, item.variables,
                                              item.bounds;
                                              points_per_axis=points_per_axis)...)
                    for item in residuals]
    minimum_grid_residual = minimum(result.value for result in grid_results)

    feasibility_report = primal_feasibility_report(model)
    maximum_conic_violation = isempty(feasibility_report) ? 0.0 :
                              maximum(values(feasibility_report))
    minimum_gram_eigenvalue_value, gram_matrix_count =
        minimum_gram_eigenvalue(metadata.infos)

    lmi_margins = map(P -> supply_lmi_margin(P, config.ring_size),
                      (Upsilon_q0, Upsilon_q1, Gamma_q0, Gamma_q1))
    required_grid_margin = -config.verification_tolerance
    valid = minimum_grid_residual >= required_grid_margin &&
            maximum_conic_violation <= 1.0e-6 &&
            minimum_gram_eigenvalue_value >= -config.verification_tolerance &&
            minimum(lmi_margins) >= -config.verification_tolerance

    return (
        valid=valid,
        status=status,
        primal=primal,
        raw_status=raw_status(model),
        polynomials=(Bq0=Bq0, Bq1=Bq1, Vq0=Vq0, Vq1=Vq1),
        supplies=(Upsilon_q0=Upsilon_q0, Upsilon_q1=Upsilon_q1,
                  Gamma_q0=Gamma_q0, Gamma_q1=Gamma_q1),
        epsilon_local=epsilon_local,
        grid_results=grid_results,
        minimum_grid_residual=minimum_grid_residual,
        maximum_conic_violation=maximum_conic_violation,
        minimum_gram_eigenvalue=minimum_gram_eigenvalue_value,
        gram_matrix_count=gram_matrix_count,
        lmi_margins=lmi_margins,
    )
end

function solve_local_cpc(config::LocalCPCConfig; points_per_axis=301)
    build_seconds = @elapsed model, metadata = build_local_cpc_model(config)
    solve_seconds = @elapsed optimize!(model)
    verification_seconds = @elapsed verification =
        verify_local_solution(model, metadata, config;
                              points_per_axis=points_per_axis)
    return (config=config, model=model, metadata=metadata,
            build_seconds=build_seconds, solve_seconds=solve_seconds,
            verification_seconds=verification_seconds,
            verification=verification)
end

function print_local_summary(result)
    config, verification = result.config, result.verification
    println("Local CPC synthesis")
    println("  degree: ", config.certificate_degree)
    println("  ring size: ", config.ring_size)
    println("  fixed supply matrices: ", config.fix_supply_matrices)
    println("  local-PA mode: ",
            config.projected_local_pa ? "projected" : "deterministic-local")
    println("  alpha: ", config.alpha)
    @printf("  build / solve / verify: %.3f / %.3f / %.3f s\n",
            result.build_seconds, result.solve_seconds,
            result.verification_seconds)
    println("  solver status: ", verification.status)
    println("  verification passed: ", verification.valid)
    if hasproperty(verification, :minimum_grid_residual)
        @printf("  minimum sampled residual: %.9e\n",
                verification.minimum_grid_residual)
        @printf("  maximum conic violation: %.9e\n",
                verification.maximum_conic_violation)
        @printf("  minimum Gram eigenvalue: %.9e\n",
                verification.minimum_gram_eigenvalue)
        println("  ring-LMI margins: ", verification.lmi_margins)
        println("  epsilon: ", verification.epsilon_local)
        if !verification.valid
            for item in verification.grid_results
                item.value < -config.verification_tolerance &&
                    println("    failing residual ", item.name, ": ", item.value,
                            " at ", item.point)
            end
        end
    else
        println("  reason: ", verification.reason)
    end
end
