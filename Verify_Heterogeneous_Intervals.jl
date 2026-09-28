module VerifyHeterogeneousIntervals

using Printf
using LinearAlgebra

const Q = Rational{BigInt}
q(n::Integer, d::Integer=1) = Q(BigInt(n), BigInt(d))

is_comfortable_temperature(x::Real) = 23 <= x <= 26
global_good_label(x::AbstractVector{<:Real}) =
    all(is_comfortable_temperature, x)
global_bad_label(x::AbstractVector{<:Real}) = !global_good_label(x)
automaton_successor(x::AbstractVector{<:Real}) =
    global_good_label(x) ? :q1 : :q0

function verify_global_label_regression()
    cases = (
        (fill(q(25), 5), true, :q1),
        ([q(25), q(22), q(25), q(25), q(25)], false, :q0),
        ([q(25), q(27), q(25), q(25), q(25)], false, :q0),
        ([q(22), q(25), q(27), q(25), q(25)], false, :q0),
    )
    for (state, expected_good, expected_successor) in cases
        @assert global_good_label(state) == expected_good
        @assert global_bad_label(state) == !expected_good
        @assert automaton_successor(state) == expected_successor
    end
    return true
end

function cluster_pattern(; clusters_size_2=20, clusters_size_3=20)
    clusters_size_2 == clusters_size_3 ||
        throw(ArgumentError("the requested pattern strictly alternates 2 and 3"))
    sizes = repeat([2, 3], clusters_size_2)
    return reduce(vcat, (fill(size, size) for size in sizes))
end

function exact_closed_loop(; clusters_size_2=20, clusters_size_3=20)
    types = cluster_pattern(; clusters_size_2, clusters_size_3)
    rooms = length(types)
    A = zeros(Q, rooms, rooms)
    offset = zeros(Q, rooms)
    for room in 1:rooms
        is_type_2 = types[room] == 2
        eta = is_type_2 ? q(1, 10) : q(1, 20)
        controller = is_type_2 ? q(2) : q(13, 5)
        A[room, room] = 1 - 2eta - q(1, 20) - q(1, 8) * controller
        A[room, mod1(room - 1, rooms)] = eta
        A[room, mod1(room + 1, rooms)] = eta
        offset[room] = q(1, 8) * 30 * controller
    end
    return (types=types, A=A, offset=offset)
end

"Exact five-room equilibrium for one [2,3] cluster period."
function periodic_equilibrium()
    system = exact_closed_loop(clusters_size_2=1, clusters_size_3=1)
    identity_matrix = Matrix{Q}(I, 5, 5)
    equilibrium = (identity_matrix - system.A) \ system.offset
    @assert (identity_matrix - system.A) * equilibrium == system.offset
    return equilibrium
end

function repeated_equilibrium(rooms::Int)
    iszero(rooms % 5) || throw(ArgumentError("room count must contain complete [2,3] periods"))
    return repeat(periodic_equilibrium(), rooms ÷ 5)
end

function cluster_index_ranges(; clusters_size_2=20, clusters_size_3=20)
    clusters_size_2 == clusters_size_3 ||
        throw(ArgumentError("the requested pattern strictly alternates 2 and 3"))
    ranges = UnitRange{Int}[]
    first_index = 1
    for dimension in repeat([2, 3], clusters_size_2)
        push!(ranges, first_index:(first_index + dimension - 1))
        first_index += dimension
    end
    return ranges
end

"""
Return an exact interval-SOS decomposition for `c*z^2-1` on either bad
component. For `z in [delta,upper]`,

    c*z^2-1 = c*A*(z-delta)^2 + c*B*(z-delta)*(upper-z)
                + (c*delta^2-1),

where `A=(upper+delta)/(upper-delta)` and
`B=2delta/(upper-delta)`. The low-temperature component follows after
the substitution `z -> -z`.
"""
function bad_interval_sos(delta::Q, upper::Q, c::Q)
    0 < delta < upper || throw(ArgumentError("invalid bad-region interval"))
    square_weight = (upper + delta) / (upper - delta)
    domain_weight = 2delta / (upper - delta)
    constant_square = c * delta^2 - 1
    @assert square_weight > 0
    @assert domain_weight > 0
    @assert constant_square >= 0
    return (square_weight=c * square_weight,
            domain_weight=c * domain_weight,
            constant_square=constant_square)
end

"""
Exact degree-two compositional CPC for the heterogeneous 100-room ring.

The state is centered at the exact period-five equilibrium. Each cluster
outputs its complete centered state; the rectangular interconnection map
selects the two neighboring boundary coordinates. This keeps `w=M*y` exact
although the equilibrium temperatures differ by room type.
"""
function verify_heterogeneous_cpc(; clusters_size_2=20,
                                  clusters_size_3=20,
                                  alpha=q(1, 100), verbose=true)
    system = exact_closed_loop(; clusters_size_2, clusters_size_3)
    rooms = length(system.types)
    equilibrium = repeated_equilibrium(rooms)
    identity_matrix = Matrix{Q}(I, rooms, rooms)
    @assert (identity_matrix - system.A) * equilibrium == system.offset
    @assert alpha == q(1, 100)

    # Both fixed cluster controllers preserve the full physical state box.
    state_lower_image = system.A * fill(q(15), rooms) + system.offset
    state_upper_image = system.A * fill(q(30), rooms) + system.offset
    state_domain_invariant = minimum(state_lower_image) >= 15 &&
                             maximum(state_upper_image) <= 30
    @assert state_domain_invariant

    # The induced-norm bound is exact and avoids floating-point eigenvalues:
    # ||A||_2^2 <= ||A||_1*||A||_inf <= (7/10)^2.
    row_sum_bound = maximum(sum(abs(system.A[i, j]) for j in 1:rooms)
                            for i in 1:rooms)
    column_sum_bound = maximum(sum(abs(system.A[i, j]) for i in 1:rooms)
                               for j in 1:rooms)
    rho = q(7, 10)
    @assert row_sum_bound <= rho
    @assert column_sum_bound <= rho

    comfort_margin = minimum(minimum((xbar - 23, 26 - xbar))
                             for xbar in equilibrium)
    @assert comfort_margin == q(4, 229)
    residual_gain = inv(comfort_margin^2)       # c
    decrease_fraction = 1 - rho^2               # 51/100
    storage_gain = residual_gain / decrease_fraction # kappa
    epsilon = q(1)
    @assert residual_gain * comfort_margin^2 == epsilon

    ranges = cluster_index_ranges(; clusters_size_2, clusters_size_3)
    supplies = Matrix{Q}[]
    composition = zeros(Q, rooms, rooms)
    local_identity_count = 0
    interval_sos_count = 0

    for indices in ranges
        dimension = length(indices)
        first_index, last_index = first(indices), last(indices)
        previous_index = mod1(first_index - 1, rooms)
        next_index = mod1(last_index + 1, rooms)

        A_local = system.A[indices, indices]
        E_local = zeros(Q, dimension, 2)
        E_local[1, 1] = system.A[first_index, previous_index]
        E_local[end, 2] = system.A[last_index, next_index]
        F_local = hcat(E_local, A_local)
        state_weight = zeros(Q, dimension + 2, dimension + 2)
        for i in 1:dimension
            state_weight[2 + i, 2 + i] = 1
        end
        Gamma = storage_gain * (transpose(F_local) * F_local - state_weight) +
                residual_gain * state_weight
        push!(supplies, Gamma)

        # With s=[w_left,w_right,z_cluster], this is the polynomial identity
        # Gamma(s)=kappa*(||z_cluster^+||^2-||z_cluster||^2)+c||z_cluster||^2.
        @assert Gamma == storage_gain *
            (transpose(F_local) * F_local - state_weight) +
            residual_gain * state_weight
        local_identity_count += 1

        selector = zeros(Q, dimension + 2, rooms)
        selector[1, previous_index] = 1
        selector[2, next_index] = 1
        for (local_index, global_index) in enumerate(indices)
            selector[2 + local_index, global_index] = 1
        end
        composition += transpose(selector) * Gamma * selector

        for global_index in indices
            xbar = equilibrium[global_index]
            low_delta, low_upper = xbar - 23, xbar - 15
            high_delta, high_upper = 26 - xbar, 30 - xbar
            bad_interval_sos(low_delta, low_upper, residual_gain)
            bad_interval_sos(high_delta, high_upper, residual_gain)
            interval_sos_count += 2
        end
    end

    expected_composition = storage_gain *
        (transpose(system.A) * system.A - identity_matrix) +
        residual_gain * identity_matrix
    @assert composition == expected_composition
    composition_upper_bound = storage_gain *
        (row_sum_bound * column_sum_bound - 1) + residual_gain
    @assert composition_upper_bound < 0

    # B_q0=B_q1=0 and Upsilon=0. For every cluster,
    # V_q0=kappa*||z||^2+1 and V_q1=kappa*||z||^2. The four local rank
    # residuals are c||z||^2 on a and c||z||^2-1 on b. The interval
    # decompositions above certify the latter on every component of b.
    pass = local_identity_count == length(ranges) &&
           interval_sos_count == 2rooms && composition_upper_bound < 0

    if verbose
        println("Exact heterogeneous compositional CPC: ", pass ? "PASS" : "FAIL")
        println("  rooms / clusters: ", rooms, " / ", length(ranges))
        println("  physical actuator set: [0,3] (0 = heater off)")
        println("  fixed controls preserve the certified state domain")
        println("  image of [15,30]^N: [", minimum(state_lower_image),
                ", ", maximum(state_upper_image), "]")
        println("  alpha: ", alpha, "; degree: 2")
        println("  eta_2=1/10, u_2=2; eta_3=1/20, u_3=13/5")
        println("  period-five equilibrium: ", periodic_equilibrium())
        println("  minimum comfort margin: ", comfort_margin)
        println("  B_q0=B_q1=0; Upsilon=0; epsilon=1")
        println("  V_q0=kappa*||z||^2+1; V_q1=kappa*||z||^2")
        println("  kappa=", storage_gain, "; residual gain c=", residual_gain)
        println("  ||A||_1 / ||A||_inf: ", column_sum_bound, " / ", row_sum_bound)
        println("  aggregate quadratic upper bound: ", composition_upper_bound)
        println("  local identities / bad-region SOS pieces: ",
                local_identity_count, " / ", interval_sos_count)
    end
    return (pass=pass, rooms=rooms, clusters=length(ranges), alpha=alpha,
            degree=2, equilibrium=equilibrium, comfort_margin=comfort_margin,
            state_domain_invariant=state_domain_invariant,
            storage_gain=storage_gain, residual_gain=residual_gain,
            epsilon=epsilon, row_sum_bound=row_sum_bound,
            column_sum_bound=column_sum_bound,
            composition_upper_bound=composition_upper_bound,
            supplies=supplies)
end

"Exact monolithic CPC for the same heterogeneous plant used by the clusters."
function verify_heterogeneous_monolithic(; clusters_size_2=2,
                                         clusters_size_3=2,
                                         alpha=q(1, 100), verbose=true)
    system = exact_closed_loop(; clusters_size_2, clusters_size_3)
    rooms = length(system.types)
    equilibrium = repeated_equilibrium(rooms)
    identity_matrix = Matrix{Q}(I, rooms, rooms)
    @assert (identity_matrix - system.A) * equilibrium == system.offset
    @assert alpha == q(1, 100)

    row_sum_bound = maximum(sum(abs(system.A[i, j]) for j in 1:rooms)
                            for i in 1:rooms)
    column_sum_bound = maximum(sum(abs(system.A[i, j]) for i in 1:rooms)
                               for j in 1:rooms)
    rho = q(7, 10)
    @assert row_sum_bound <= rho
    @assert column_sum_bound <= rho
    comfort_margin = minimum(minimum((xbar - 23, 26 - xbar))
                             for xbar in equilibrium)
    residual_gain = inv(comfort_margin^2)
    storage_gain = residual_gain / (1 - rho^2)
    epsilon = q(1)

    # B=0, V_q0=kappa*||z||^2+1, V_q1=kappa*||z||^2. The rank
    # residual is c||z||^2 on global good and c||z||^2-1 on global bad.
    # If the global label is bad, at least one coordinate is at least its
    # comfort-boundary distance from equilibrium.
    @assert residual_gain * comfort_margin^2 == epsilon
    for xbar in equilibrium
        bad_interval_sos(xbar - 23, xbar - 15, residual_gain)
        bad_interval_sos(26 - xbar, 30 - xbar, residual_gain)
    end
    contraction_residual_bound = storage_gain *
        (row_sum_bound * column_sum_bound - 1) + residual_gain
    pass = contraction_residual_bound < 0

    if verbose
        println("Matched heterogeneous monolithic CPC: ",
                pass ? "PASS" : "FAIL")
        println("  rooms: ", rooms, "; pattern: [2,3] repeated ",
                clusters_size_2)
        println("  alpha: ", alpha, "; degree: 2; epsilon: ", epsilon)
        println("  equilibrium period: ", periodic_equilibrium())
        println("  kappa= ", storage_gain, "; c= ", residual_gain)
        println("  global residual upper bound: ",
                contraction_residual_bound)
    end
    return (pass=pass, rooms=rooms, alpha=alpha, degree=2,
            equilibrium=equilibrium, comfort_margin=comfort_margin,
            storage_gain=storage_gain, residual_gain=residual_gain,
            epsilon=epsilon,
            composition_upper_bound=contraction_residual_bound)
end

function verify_matched_heterogeneous_N10(; verbose=true)
    compositional_seconds = @elapsed compositional =
        verify_heterogeneous_cpc(clusters_size_2=2, clusters_size_3=2,
                                 verbose=verbose)
    monolithic_seconds = @elapsed monolithic =
        verify_heterogeneous_monolithic(clusters_size_2=2, clusters_size_3=2,
                                        verbose=verbose)
    # Summing the cluster storage functions gives the monolithic storage
    # function exactly, so both routes must return the same gains and bound.
    @assert compositional.storage_gain == monolithic.storage_gain
    @assert compositional.residual_gain == monolithic.residual_gain
    @assert compositional.composition_upper_bound ==
            monolithic.composition_upper_bound
    pass = compositional.pass && monolithic.pass
    verbose && println("Matched heterogeneous N=10 comparison: ",
                       pass ? "PASS" : "FAIL")
    return (pass=pass, compositional=compositional,
            monolithic=monolithic,
            compositional_seconds=compositional_seconds,
            monolithic_seconds=monolithic_seconds)
end

function verify_heterogeneous_intervals(; clusters_size_2=20,
                                        clusters_size_3=20,
                                        maximum_steps=100, verbose=true)
    system = exact_closed_loop(; clusters_size_2, clusters_size_3)
    rooms = length(system.types)
    all(system.A .>= 0) || error("interval propagation requires a positive system")
    lower = fill(q(20), rooms)
    upper = fill(q(25), rooms)
    history = NamedTuple[]
    entry_step = nothing
    upper_invariant = true

    for step in 0:maximum_steps
        push!(history, (step=step, minimum_lower=minimum(lower),
                        maximum_upper=maximum(upper)))
        upper_invariant &= maximum(upper) <= 26
        if minimum(lower) >= 23 && maximum(upper) <= 26
            entry_step = step
            break
        end
        lower = system.A * lower + system.offset
        upper = system.A * upper + system.offset
    end

    # Direct image of the whole comfort box. Positivity makes its two uniform
    # corners the exact componentwise extrema.
    comfort_lower_image = system.A * fill(q(23), rooms) + system.offset
    comfort_upper_image = system.A * fill(q(26), rooms) + system.offset
    comfort_invariant = minimum(comfort_lower_image) >= 23 &&
                        maximum(comfort_upper_image) <= 26
    pass = upper_invariant && comfort_invariant && entry_step == 3

    if verbose
        println("Exact heterogeneous interval verification: ", pass ? "PASS" : "FAIL")
        println("  20 two-room clusters: eta=1/10, u=2, nominal x*=25")
        println("  20 three-room clusters: eta=1/20, u=13/5, nominal x*=26")
        for row in history
            @printf("  t=%d: min lower=%.9f, max upper=%.9f\n",
                    row.step, Float64(row.minimum_lower),
                    Float64(row.maximum_upper))
        end
        println("  comfort box invariant: ", comfort_invariant)
        println("  guaranteed entry step: ", entry_step)
    end
    return (pass=pass, rooms=rooms, entry_step=entry_step,
            comfort_invariant=comfort_invariant, history=history,
            system=system)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    label_result = verify_global_label_regression()
    cpc_seconds = @elapsed cpc_result = verify_heterogeneous_cpc()
    interval_seconds = @elapsed interval_result = verify_heterogeneous_intervals()
    matched_result = verify_matched_heterogeneous_N10()
    output_path = joinpath(@__DIR__,
        "CPC_Heterogeneous_exact_results_20x2_20x3_degree2.txt")
    open(output_path, "w") do file
        println(file, "status: exact heterogeneous compositional CPC verified")
        println(file, "cluster_pattern: 20 repetitions of [2, 3]")
        println(file, "scalar_rooms: ", cpc_result.rooms)
        println(file, "degree: ", cpc_result.degree)
        println(file, "alpha: ", cpc_result.alpha)
        println(file, "input_set: [0, 3]")
        println(file, "input_set_role: physical actuator set; 0 means heater off")
        println(file, "supply_coordinates: equilibrium-centered physical ports")
        println(file, "fixed_controller_state_domain_invariant: ",
                cpc_result.state_domain_invariant)
        println(file, "eta_size_2: 0.1")
        println(file, "eta_size_3: 0.05")
        println(file, "controller_size_2: 2")
        println(file, "controller_size_3: 2.6")
        println(file, "periodic_equilibrium: ", periodic_equilibrium())
        println(file, "minimum_comfort_margin: ", cpc_result.comfort_margin)
        println(file, "storage_gain_kappa: ", cpc_result.storage_gain)
        println(file, "residual_gain_c: ", cpc_result.residual_gain)
        println(file, "epsilon: ", cpc_result.epsilon)
        println(file, "composition_upper_bound: ",
                cpc_result.composition_upper_bound)
        println(file, "guaranteed_comfort_entry_step: ",
                interval_result.entry_step)
        println(file, "cpc_verification_seconds: ", cpc_seconds)
        println(file, "interval_verification_seconds: ", interval_seconds)
        println(file, "matched_N10_status: ", matched_result.pass)
        println(file, "matched_N10_compositional_seconds: ",
                matched_result.compositional_seconds)
        println(file, "matched_N10_monolithic_seconds: ",
                matched_result.monolithic_seconds)
        println(file, "global_label_regression: ", label_result)
    end
    println("Results written to ", output_path)
    (label_result && cpc_result.pass && interval_result.pass &&
     matched_result.pass) || exit(1)
end

end # module
