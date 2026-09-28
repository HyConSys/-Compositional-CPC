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
    barrier_degree::Union{Nothing,Int} = nothing
    rank_degree::Union{Nothing,Int} = nothing
    ring_size::Int = 100
    fix_supply_matrices::Bool = false
    common_supply_across_states::Bool = true
    projected_local_pa::Bool = true
    alpha::Float64 = 0.01
    epsilon_lower_bound::Float64 = 1.0e-6
    epsilon_upper_bound::Float64 = 0.1
    epsilon_reward::Float64 = 1.0e-3
    verification_tolerance::Float64 = 1.0e-7
    coefficient_regularization::Float64 = 1.0e-4
    fixed_supply_values::Union{Nothing,NTuple{4,Matrix{Float64}}} = nothing
    fixed_epsilon::Union{Nothing,Float64} = nothing
    zero_barriers::Bool = false
    equilibrium_factored_barriers::Bool = false
    structured_equilibrium_supplies::Bool = false
    equilibrium_facial_reduction::Bool = false
    vanishing_slack::Float64 = 0.0
    solver_feasibility_tolerance::Union{Nothing,Float64} = nothing
    quiet::Bool = true
end

const ETA = 0.1
const EXTERIOR_COEFFICIENT = 0.05
const HEATER_COEFFICIENT = 0.125
const EXTERIOR_TEMPERATURE = 0.0
const HEATER_TEMPERATURE = 30.0
const INPUT_LOWER = 0.0
const INPUT_UPPER = 3.0
# Physical actuator limits; zero is the heater-off command. The certificate
# programs below verify the fixed controller u=2, including its state-domain
# preservation. They do not assert that every value in [0,3] preserves the
# compact certification interval at every boundary state.
const CONTROLLER_SLOPE = 0.0
const CONTROLLER_INTERCEPT = 2.0
const ALPHA_CPC = 0.01
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
# The uniform normalized residual has dominant coefficient 9/32. Dividing by
# this value keeps its squared factor near unit scale; the free multiplier H_q
# absorbs the constant without changing the represented barrier family.
const EQUILIBRIUM_RESIDUAL_SCALE = 9.0 / 32.0

is_comfortable_temperature(x::Real) = 23.0 <= x <= 26.0
global_good_label(x::AbstractVector{<:Real}) =
    all(is_comfortable_temperature, x)
global_bad_label(x::AbstractVector{<:Real}) = !global_good_label(x)
automaton_successor(x::AbstractVector{<:Real}) =
    global_good_label(x) ? :q1 : :q0

function verify_global_label_regression()
    cases = (
        (fill(25.0, 4), true, :q1),
        ([25.0, 22.0, 25.0, 25.0], false, :q0),
        ([25.0, 27.0, 25.0, 25.0], false, :q0),
        ([22.0, 25.0, 27.0, 25.0], false, :q0),
    )
    for (state, expected_good, expected_successor) in cases
        @assert global_good_label(state) == expected_good
        @assert global_bad_label(state) == !expected_good
        @assert automaton_successor(state) == expected_successor
    end
    return true
end

const UPSILON_Q0_REFERENCE = [-1.8 -0.3; -0.3 -2.6] ./ CPC_NUMERICAL_SCALE
const UPSILON_Q1_REFERENCE = [-1.8 -0.3; -0.3 -2.6] ./ CPC_NUMERICAL_SCALE
const GAMMA_Q0_REFERENCE = [0.0 0.0; 0.0 -1.4] ./ CPC_NUMERICAL_SCALE
const GAMMA_Q1_REFERENCE = [0.2 0.1; 0.1 -1.3] ./ CPC_NUMERICAL_SCALE

normalized_x(x::Real) = (x - X_CENTER) / X_SCALE
physical_x(xhat) = X_CENTER + X_SCALE * xhat
physical_w(what) = W_CENTER + W_SCALE * what

function controller(x)
    u = CONTROLLER_SLOPE * x + CONTROLLER_INTERCEPT
    if u isa Real
        INPUT_LOWER <= u <= INPUT_UPPER ||
            throw(DomainError(u, "controller value is outside the admissible input set"))
    end
    return u
end

function uniform_equilibrium()
    # For w=2x, the interconnection terms containing ETA cancel. The fixed
    # affine controller therefore gives a scalar quadratic equilibrium
    # equation a*x^2+b*x+c=0.
    a = -HEATER_COEFFICIENT * CONTROLLER_SLOPE
    b = -EXTERIOR_COEFFICIENT -
        HEATER_COEFFICIENT * CONTROLLER_INTERCEPT +
        HEATER_COEFFICIENT * HEATER_TEMPERATURE * CONTROLLER_SLOPE
    c = HEATER_COEFFICIENT * HEATER_TEMPERATURE * CONTROLLER_INTERCEPT +
        EXTERIOR_COEFFICIENT * EXTERIOR_TEMPERATURE
    roots = if iszero(a)
        iszero(b) && error("uniform equilibrium equation is degenerate")
        (-c / b,)
    else
        discriminant = b^2 - 4a * c
        discriminant >= 0 || error("uniform equilibrium equation has no real root")
        ((-b - sqrt(discriminant)) / (2a),
         (-b + sqrt(discriminant)) / (2a))
    end
    admissible = filter(root -> 15.0 <= root <= 30.0, roots)
    length(admissible) == 1 ||
        error("expected exactly one uniform equilibrium in [15,30]")
    return only(admissible)
end

normalized_uniform_equilibrium() = normalized_x(uniform_equilibrium())

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
    isempty(coefficients_ref) && return 0.0 * first(basis)
    return sum(value(coefficients_ref[i]) * basis[i] for i in eachindex(coefficients_ref))
end

solution_value(x::Number) = Float64(x)
solution_value(x) = value(x)

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

function add_equilibrium_kernel_supply!(model, name::String)
    b = @variable(model, lower_bound=0.0, base_name="$(name)_b")
    d = @variable(model, lower_bound=0.0, base_name="$(name)_d")
    a = (d - b) / 2
    # For q(lambda) = [lambda;1]'P[lambda;1], this gives
    # q(lambda)=(lambda-2)(a*lambda+2a+2b). The second factor is
    # nonnegative on [-2,2], so q <= 0 there, and q(2)=0 exactly.
    P = [a b; b -4a-4b]
    return P, (b=b, d=d)
end

function add_reduced_face_psatz_constraint!(infos, model, polynomial,
                                            variables, domain, order,
                                            zero_point,
                                            facial_reduction_counts)
    length(zero_point) == length(variables) ||
        throw(ArgumentError("reduced-face zero point has the wrong dimension"))
    all(Float64(domain_polynomial(variables => zero_point)) > 0
        for domain_polynomial in domain) ||
        throw(ArgumentError("reduced-face point must be in the strict interior of every inequality generator"))

    generators = Any[one(first(variables)); domain...]
    certificate = 0.0 * polynomial
    gram_matrices = Any[]
    reduced_bases = Any[]
    eliminated_kernel_equations = 0

    for generator in generators
        generator_degree = MultivariatePolynomials.maxdegree(generator)
        basis_degree = order - cld(generator_degree, 2)
        basis_degree >= 0 ||
            throw(ArgumentError("SOS order is too small for a domain generator"))
        basis = collect(MultivariatePolynomials.monomials(
            variables, 0:basis_degree))
        evaluation = [Float64(monomial(variables => zero_point))
                      for monomial in basis]
        evaluation_norm = norm(evaluation)
        evaluation_norm > 0 ||
            error("cannot reduce an SOS basis with zero evaluation vector")

        # Columns of N span the orthogonal complement of m(z*). Every
        # polynomial in N'*m therefore vanishes at z*, and a PSD Gram matrix
        # in this reduced basis lies directly on the required exposed face.
        N = nullspace(reshape(evaluation / evaluation_norm, 1, :))
        reduced_dimension = size(N, 2)
        reduced_dimension == 0 && continue
        reduced_basis = [sum(N[row, column] * basis[row]
                             for row in eachindex(basis))
                         for column in 1:reduced_dimension]
        gram = @variable(model,
            [1:reduced_dimension, 1:reduced_dimension], PSD)
        sigma = sum(gram[row, column] * reduced_basis[row] *
                    reduced_basis[column]
                    for row in 1:reduced_dimension,
                        column in 1:reduced_dimension)
        certificate += generator * sigma
        push!(gram_matrices, gram)
        push!(reduced_bases, reduced_basis)
        eliminated_kernel_equations += length(basis)
    end

    identity = certificate - polynomial
    identity_coefficients = MultivariatePolynomials.coefficients(identity)
    @constraint(model, identity_coefficients .== 0.0)

    # Match the nested GramMat layout used by TSSOS so the common numerical
    # diagnostics can inspect both ordinary and reduced-face certificates.
    info = (GramMat=[[[gram] for gram in gram_matrices]],
            reduced_bases=reduced_bases,
            zero_point=copy(zero_point))
    push!(infos, info)
    push!(facial_reduction_counts, eliminated_kernel_equations)
    return info
end

function add_psatz_constraint!(infos, model, polynomial, variables, domain, order)
    info = add_psatz!(model, polynomial, variables, domain, [], order;
        QUIET=true, CS=false, TS=false, GroebnerBasis=false)
    push!(infos, info)
    return info
end

function build_local_cpc_model(config::LocalCPCConfig)
    barrier_degree = something(config.barrier_degree, config.certificate_degree)
    rank_degree = something(config.rank_degree, config.certificate_degree)
    barrier_degree >= 2 || throw(ArgumentError("barrier degree must be at least 2"))
    rank_degree >= 2 || throw(ArgumentError("rank degree must be at least 2"))
    config.zero_barriers && config.equilibrium_factored_barriers &&
        throw(ArgumentError("zero and equilibrium-factored barriers are mutually exclusive"))
    config.equilibrium_factored_barriers && barrier_degree < 4 &&
        throw(ArgumentError("an equilibrium-factored barrier needs degree at least 4 because the quadratic equilibrium residual is squared"))
    config.structured_equilibrium_supplies && !isnothing(config.fixed_supply_values) &&
        throw(ArgumentError("structured and fixed supplies are mutually exclusive"))
    config.structured_equilibrium_supplies && config.fix_supply_matrices &&
        throw(ArgumentError("structured and reference-fixed supplies are mutually exclusive"))
    fixed_supplies_have_uniform_kernel =
        !isnothing(config.fixed_supply_values) &&
        all(P -> 4P[1, 1] + 4P[1, 2] + P[2, 2] == 0.0,
            config.fixed_supply_values)
    config.equilibrium_facial_reduction &&
        !(config.equilibrium_factored_barriers &&
          (config.structured_equilibrium_supplies ||
           fixed_supplies_have_uniform_kernel)) &&
        throw(ArgumentError("equilibrium facial reduction requires factored barriers and supplies with an exact uniform-mode kernel"))
    config.vanishing_slack >= 0 ||
        throw(ArgumentError("vanishing slack must be nonnegative"))

    model = Model(Mosek.Optimizer)
    config.quiet && set_silent(model)
    if !isnothing(config.solver_feasibility_tolerance)
        tolerance = config.solver_feasibility_tolerance
        tolerance > 0 || throw(ArgumentError("solver feasibility tolerance must be positive"))
        for attribute in ("MSK_DPAR_INTPNT_CO_TOL_PFEAS",
                          "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
                          "MSK_DPAR_INTPNT_CO_TOL_INFEAS")
            set_optimizer_attribute(model, attribute, tolerance)
        end
    end

    @polyvar xhat what
    x = physical_x(xhat)
    w = physical_w(what)
    xhat_next = (room_dynamics(x, w) - X_CENTER) / X_SCALE
    uniform_successor = subs(xhat_next, what => xhat)
    normalized_equilibrium_residual =
        (uniform_successor - xhat) / EQUILIBRIUM_RESIDUAL_SCALE
    equilibrium_measure = normalized_equilibrium_residual^2 + (what - xhat)^2

    barrier_multiplier_refs = nothing
    barrier_multiplier_bases = nothing
    barrier_factor = nothing
    Hq0 = nothing
    Hq1 = nothing
    if config.zero_barriers
        Bq0 = zero(xhat)
        Bq1 = zero(xhat)
        Bq0_coefficients = VariableRef[]
        Bq1_coefficients = VariableRef[]
        Bq0_basis = [xhat^0]
        Bq1_basis = [xhat^0]
    elseif config.equilibrium_factored_barriers
        barrier_factor = normalized_equilibrium_residual^2
        multiplier_degree = barrier_degree - 4
        Hq0, Hq0_coefficients, Hq0_basis =
            add_poly!(model, [xhat], multiplier_degree)
        Hq1, Hq1_coefficients, Hq1_basis =
            add_poly!(model, [xhat], multiplier_degree)
        Bq0, Bq1 = barrier_factor * Hq0, barrier_factor * Hq1
        Bq0_coefficients, Bq1_coefficients = Hq0_coefficients, Hq1_coefficients
        Bq0_basis = [barrier_factor * basis for basis in Hq0_basis]
        Bq1_basis = [barrier_factor * basis for basis in Hq1_basis]
        barrier_multiplier_refs = (Hq0_coefficients, Hq1_coefficients)
        barrier_multiplier_bases = (Hq0_basis, Hq1_basis)
    else
        Bq0, Bq0_coefficients, Bq0_basis =
            add_poly!(model, [xhat], barrier_degree)
        Bq1, Bq1_coefficients, Bq1_basis =
            add_poly!(model, [xhat], barrier_degree)
    end
    Vq0, Vq0_coefficients, Vq0_basis = add_poly!(model, [xhat], rank_degree)
    Vq1, Vq1_coefficients, Vq1_basis = add_poly!(model, [xhat], rank_degree)

    Bq0_next = Bq0([xhat] => [xhat_next])
    Bq1_next = Bq1([xhat] => [xhat_next])
    Vq0_next = Vq0([xhat] => [xhat_next])
    Vq1_next = Vq1([xhat] => [xhat_next])

    config.epsilon_upper_bound >= config.epsilon_lower_bound > 0 ||
        throw(ArgumentError("epsilon bounds must satisfy 0 < lower <= upper"))
    epsilon_local = if isnothing(config.fixed_epsilon)
        @variable(model,
            config.epsilon_lower_bound <= epsilon_variable <= config.epsilon_upper_bound)
        epsilon_variable
    else
        config.epsilon_lower_bound <= config.fixed_epsilon <= config.epsilon_upper_bound ||
            throw(ArgumentError("fixed epsilon must lie inside the epsilon bounds"))
        config.fixed_epsilon
    end
    supply_parameter_refs = nothing
    supplies = if config.structured_equilibrium_supplies
        Upsilon, Upsilon_parameters =
            add_equilibrium_kernel_supply!(model, "Upsilon")
        Gamma, Gamma_parameters =
            add_equilibrium_kernel_supply!(model, "Gamma")
        supply_parameter_refs =
            (Upsilon=Upsilon_parameters, Gamma=Gamma_parameters)
        (Upsilon, Upsilon, Gamma, Gamma)
    elseif isnothing(config.fixed_supply_values)
        @variable(model, Upsilon_q0[1:2, 1:2], Symmetric)
        @variable(model, Upsilon_q1[1:2, 1:2], Symmetric)
        @variable(model, Gamma_q0[1:2, 1:2], Symmetric)
        @variable(model, Gamma_q1[1:2, 1:2], Symmetric)
        (Upsilon_q0, Upsilon_q1, Gamma_q0, Gamma_q1)
    else
        map(P -> Symmetric(copy(P)), config.fixed_supply_values)
    end
    references = (UPSILON_Q0_REFERENCE, UPSILON_Q1_REFERENCE,
                  GAMMA_Q0_REFERENCE, GAMMA_Q1_REFERENCE)

    config.fix_supply_matrices && !isnothing(config.fixed_supply_values) &&
        throw(ArgumentError("choose either reference or custom fixed supply matrices"))
    config.fix_supply_matrices && config.common_supply_across_states &&
        throw(ArgumentError("the reference q0/q1 Gamma matrices differ; use optimized supplies when common_supply_across_states=true"))
    if config.structured_equilibrium_supplies
        # The same structured Upsilon and Gamma are already shared by q0/q1.
    elseif config.common_supply_across_states && isnothing(config.fixed_supply_values)
        @constraint(model, Upsilon_q0 .== Upsilon_q1)
        @constraint(model, Gamma_q0 .== Gamma_q1)
    elseif config.common_supply_across_states
        supplies[1] == supplies[2] ||
            throw(ArgumentError("fixed Upsilon matrices must agree across states"))
        supplies[3] == supplies[4] ||
            throw(ArgumentError("fixed Gamma matrices must agree across states"))
    end

    if !isnothing(config.fixed_supply_values)
        all(size(P) == (2, 2) for P in config.fixed_supply_values) ||
            throw(ArgumentError("each fixed supply matrix must be 2 by 2"))
    elseif config.fix_supply_matrices
        for (P, reference) in zip(supplies, references)
            @constraint(model, P .== reference)
        end
    elseif !config.structured_equilibrium_supplies
        for P in supplies
            # The interconnection condition is semidefinite, with no
            # artificial strict margin.  A positive margin is incompatible
            # with the equilibrium contained in the labelled regions.
            add_ring_lmi_constraints!(model, P, config.ring_size)
        end
    end

    Upsilon0 = supply_polynomial(supplies[1], w, x)
    Upsilon1 = supply_polynomial(supplies[2], w, x)
    Gamma0 = supply_polynomial(supplies[3], w, x)
    Gamma1 = supply_polynomial(supplies[4], w, x)

    x0_lower, x0_upper = normalized_x(20.0), normalized_x(25.0)
    xa_lower, xa_upper = normalized_x(23.0), normalized_x(26.0)
    xb1_lower, xb1_upper = normalized_x(15.0), normalized_x(23.0)
    xb2_lower, xb2_upper = normalized_x(26.0), normalized_x(30.0)

    domain_x0 = [interval_polynomial(xhat, x0_lower, x0_upper)]
    domain_x = [1.0 - xhat^2]
    domain_xa = [interval_polynomial(xhat, xa_lower, xa_upper), 1.0 - what^2]
    domain_xb1 = [interval_polynomial(xhat, xb1_lower, xb1_upper), 1.0 - what^2]
    domain_xb2 = [interval_polynomial(xhat, xb2_lower, xb2_upper), 1.0 - what^2]

    initial_order = cld(barrier_degree, 2)
    lower_order = cld(max(barrier_degree, rank_degree), 2)
    # The closed-loop dynamics are quadratic. Thus B(f) has degree at most
    # 2*barrier_degree, while V(f) has degree at most 2*rank_degree.
    barrier_transition_order = barrier_degree
    rank_transition_order = cld(max(2rank_degree, barrier_degree, 2), 2)
    infos = Any[]
    facial_reduction_counts = Int[]
    equilibrium_hat = normalized_uniform_equilibrium()

    # Initial and lower-bound conditions.
    if !config.zero_barriers
        if config.equilibrium_factored_barriers
            # Since Bq0=e^2*Hq0, Bq0<=0 is equivalent (by continuity) to
            # Hq0<=0 on the initial interval. Certifying -Hq0 removes the
            # forced interior zero from this univariate SOS constraint.
            multiplier_order = max(1, cld(barrier_degree - 4, 2))
            add_psatz_constraint!(infos, model, -Hq0, [xhat], domain_x0,
                                  multiplier_order)
        else
            initial_residual = -Bq0 -
                config.vanishing_slack * normalized_equilibrium_residual^2
            add_psatz_constraint!(infos, model, initial_residual, [xhat],
                                  domain_x0, initial_order)
        end
    end
    add_psatz_constraint!(infos, model, Vq0 + config.alpha * Bq0,
                          [xhat], domain_x, lower_order)
    add_psatz_constraint!(infos, model, Vq1 + config.alpha * Bq1,
                          [xhat], domain_x, lower_order)

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
            residual_pairs = zip(residual_names,
                                 (residual_B_q0, residual_V_q0,
                                  residual_B_q1, residual_V_q1))
            for (name, residual) in residual_pairs
                transition_residuals[name] = residual
                config.zero_barriers && endswith(name, "_B") && continue
                transition_order = endswith(name, "_B") ?
                                   barrier_transition_order :
                                   rank_transition_order
                conditioned_residual = residual -
                    config.vanishing_slack * equilibrium_measure
                has_forced_equilibrium_zero =
                    config.equilibrium_facial_reduction && label == "a" &&
                    (endswith(name, "_B") || name == "a_q1_to_q1_V")
                gram_zero_point = has_forced_equilibrium_zero ?
                                  [equilibrium_hat, equilibrium_hat] : nothing
                if has_forced_equilibrium_zero
                    add_reduced_face_psatz_constraint!(
                        infos, model, conditioned_residual, [xhat, what],
                        domain, transition_order, gram_zero_point,
                        facial_reduction_counts)
                else
                    add_psatz_constraint!(infos, model, conditioned_residual,
                                          [xhat, what], domain,
                                          transition_order)
                end
            end
        end
    end

    all_coefficients = vcat(Bq0_coefficients, Bq1_coefficients,
                            Vq0_coefficients, Vq1_coefficients)
    objective = config.coefficient_regularization * sum(c^2 for c in all_coefficients) -
                config.epsilon_reward * epsilon_local
    if !config.fix_supply_matrices && isnothing(config.fixed_supply_values)
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
        supply_parameter_refs=supply_parameter_refs,
        epsilon_local=epsilon_local,
        barrier_degree=barrier_degree,
        rank_degree=rank_degree,
        barrier_factor=barrier_factor,
        normalized_equilibrium_residual=normalized_equilibrium_residual,
        equilibrium_measure=equilibrium_measure,
        equilibrium_hat=equilibrium_hat,
        facial_reduction_counts=facial_reduction_counts,
        barrier_multiplier_refs=barrier_multiplier_refs,
        barrier_multiplier_bases=barrier_multiplier_bases,
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
        map(P -> solution_value.(P), metadata.supplies)
    epsilon_local = solution_value(metadata.epsilon_local)
    barrier_multipliers = if isnothing(metadata.barrier_multiplier_refs)
        nothing
    else
        Hq0, Hq1 = map(numeric_polynomial,
                        metadata.barrier_multiplier_refs,
                        metadata.barrier_multiplier_bases)
        (Hq0=Hq0, Hq1=Hq1)
    end
    supply_parameters = if isnothing(metadata.supply_parameter_refs)
        nothing
    else
        (Upsilon=(b=solution_value(metadata.supply_parameter_refs.Upsilon.b),
                  d=solution_value(metadata.supply_parameter_refs.Upsilon.d)),
         Gamma=(b=solution_value(metadata.supply_parameter_refs.Gamma.b),
                d=solution_value(metadata.supply_parameter_refs.Gamma.d)))
    end

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
            minimum(lmi_margins) >= 0.0

    return (
        valid=valid,
        status=status,
        primal=primal,
        raw_status=raw_status(model),
        polynomials=(Bq0=Bq0, Bq1=Bq1, Vq0=Vq0, Vq1=Vq1),
        barrier_multipliers=barrier_multipliers,
        supplies=(Upsilon_q0=Upsilon_q0, Upsilon_q1=Upsilon_q1,
                  Gamma_q0=Gamma_q0, Gamma_q1=Gamma_q1),
        supply_parameters=supply_parameters,
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
    println("  barrier / rank degree: ",
            something(config.barrier_degree, config.certificate_degree), " / ",
            something(config.rank_degree, config.certificate_degree))
    println("  ring size: ", config.ring_size)
    println("  fixed supply matrices: ", config.fix_supply_matrices)
    println("  equilibrium facial reduction: ",
            config.equilibrium_facial_reduction,
            " (", sum(result.metadata.facial_reduction_counts),
            " eliminated Gram-kernel coordinates)")
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
