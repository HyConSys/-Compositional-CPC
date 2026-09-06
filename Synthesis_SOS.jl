if !isdefined(@__MODULE__, :LocalCPCConfig)
    include(joinpath(@__DIR__, "CPC_SOS_core.jl"))
end

function parse_integer_argument(index::Int, default::Int)
    return length(ARGS) >= index ? parse(Int, ARGS[index]) : default
end

function main()
    degree = parse_integer_argument(1, 4)
    ring_size = parse_integer_argument(2, 100)
    fixed_supplies = length(ARGS) >= 3 ? ARGS[3] == "fixed" : false
    alpha = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.1
    projected_local_pa = length(ARGS) >= 5 ? ARGS[5] == "projected" : true

    config = LocalCPCConfig(
        certificate_degree=degree,
        ring_size=ring_size,
        fix_supply_matrices=fixed_supplies,
        alpha=alpha,
        projected_local_pa=projected_local_pa,
    )
    result = solve_local_cpc(config)
    print_local_summary(result)

    result.verification.valid || exit(2)
    mode_name = config.projected_local_pa ? "projected" : "deterministic"
    alpha_name = replace(string(config.alpha), "." => "p")
    output_path = joinpath(@__DIR__,
        "CPC_SOS_results_N$(ring_size)_degree$(degree)_alpha$(alpha_name)_$(mode_name).txt")
    open(output_path, "w") do file
        verification = result.verification
        println(file, "status: verified CPC found")
        println(file, "solver_status: ", verification.status)
        println(file, "certificate_degree: ", degree)
        println(file, "ring_size: ", ring_size)
        println(file, "alpha: ", config.alpha)
        println(file, "local_pa_mode: ",
                config.projected_local_pa ? "projected" : "deterministic-local")
        println(file, "common_supply_across_states: ",
                config.common_supply_across_states)
        println(file, "heater_coefficient: ", HEATER_COEFFICIENT)
        println(file, "common_numerical_scale: ", CPC_NUMERICAL_SCALE)
        println(file, "local_epsilon_scaled: ", verification.epsilon_local)
        println(file, "local_epsilon_unscaled: ",
                verification.epsilon_local * CPC_NUMERICAL_SCALE)
        println(file, "neighbor_domain: [15, 30] per neighbor")
        println(file, "summed_internal_input_domain: [30, 60]")
        println(file, "build_seconds: ", result.build_seconds)
        println(file, "solve_seconds: ", result.solve_seconds)
        println(file, "verification_seconds: ", result.verification_seconds)
        println(file, "minimum_grid_residual: ",
                verification.minimum_grid_residual)
        println(file, "maximum_conic_violation: ",
                verification.maximum_conic_violation)
        println(file, "minimum_gram_eigenvalue: ",
                verification.minimum_gram_eigenvalue)
        for (name, polynomial) in pairs(verification.polynomials)
            println(file, name, "(xhat): ", polynomial)
        end
        for (name, matrix) in pairs(verification.supplies)
            println(file, name, ": ", matrix)
        end
        println(file, "ring_lmi_margins: ", verification.lmi_margins)
    end
    println("Verified result written to ", output_path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
