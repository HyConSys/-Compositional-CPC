include(joinpath(@__DIR__, "CPC_SOS_core.jl"))
include(joinpath(@__DIR__, "Synthesis_Monolithic_SOS.jl"))
include(joinpath(@__DIR__, "Synthesis_Heterogeneous_SOS.jl"))

function benchmark_row(name, continuous_dimension, degree, alpha, result,
                       minimum_residual)
    is_valid = hasproperty(result, :valid) ? result.valid : result.verification.valid
    return (
        experiment=name,
        continuous_dimension=continuous_dimension,
        certificate_degree=degree,
        alpha=alpha,
        build_seconds=result.build_seconds,
        solve_seconds=result.solve_seconds,
        verification_seconds=result.verification_seconds,
        valid=is_valid,
        minimum_residual=minimum_residual,
    )
end

function run_benchmarks()
    println("Warm-up: normalized local SOS and MOSEK interface")
    warm_config = LocalCPCConfig(certificate_degree=2, ring_size=10,
        alpha=0.01, projected_local_pa=false)
    warm_result = solve_local_cpc(warm_config; points_per_axis=31)
    warm_result.verification.valid || error("local warm-up did not verify")

    println("Benchmark: homogeneous compositional N=10")
    compositional_10 = solve_local_cpc(LocalCPCConfig(
        certificate_degree=2, ring_size=10, alpha=0.01,
        projected_local_pa=false); points_per_axis=301)
    print_local_summary(compositional_10)

    println("Benchmark: homogeneous compositional N=100")
    compositional_100 = solve_local_cpc(LocalCPCConfig(
        certificate_degree=2, ring_size=100, alpha=0.01,
        projected_local_pa=false); points_per_axis=301)
    print_local_summary(compositional_100)

    println("Warm-up: monolithic model methods at N=3")
    monolithic_warm = solve_monolithic_cpc(MonolithicCPCConfig(
        rooms=3, certificate_degree=2, alpha=0.01,
        verification_samples=100))
    hasproperty(monolithic_warm, :verification) ||
        error("monolithic warm-up was not run")

    println("Benchmark: monolithic N=10")
    monolithic_10 = solve_monolithic_cpc(MonolithicCPCConfig(
        rooms=10, certificate_degree=2, alpha=0.01,
        verification_samples=10_000))
    hasproperty(monolithic_10, :verification) ||
        error("monolithic N=10 exceeded the safety limit")
    println("  monolithic status / valid: ",
            (monolithic_10.verification.status,
             monolithic_10.verification.valid))
    @printf("  monolithic build / solve / verify: %.3f / %.3f / %.3f s\n",
            monolithic_10.build_seconds, monolithic_10.solve_seconds,
            monolithic_10.verification_seconds)

    println("Benchmark: heterogeneous 20x2 plus 20x3")
    heterogeneous = solve_heterogeneous_cpc(HeterogeneousCPCConfig(
        certificate_degree=2, alpha=0.01, verification_samples=20_000))
    println("  heterogeneous status / valid: ",
            (heterogeneous.status, heterogeneous.valid))
    @printf("  heterogeneous build / solve / verify: %.3f / %.3f / %.3f s\n",
            heterogeneous.build_seconds, heterogeneous.solve_seconds,
            heterogeneous.verification_seconds)

    rows = [
        benchmark_row("compositional_homogeneous_N10", 10, 2, 0.01,
            compositional_10, compositional_10.verification.minimum_grid_residual),
        benchmark_row("compositional_homogeneous_N100", 100, 2, 0.01,
            compositional_100, compositional_100.verification.minimum_grid_residual),
        benchmark_row("monolithic_N10", 10, 2, 0.01, monolithic_10,
            hasproperty(monolithic_10.verification, :minimum_sampled_residual) ?
                monolithic_10.verification.minimum_sampled_residual : NaN),
        benchmark_row("heterogeneous_20x2_20x3", 100, 2, 0.01, heterogeneous,
            hasproperty(heterogeneous, :minimum_residual) ?
                heterogeneous.minimum_residual : NaN),
    ]

    output_path = joinpath(@__DIR__, "CPC_benchmark_results.tsv")
    open(output_path, "w") do file
        println(file,
            "experiment\tcontinuous_dimension\tcertificate_degree\talpha\tbuild_seconds\tsolve_seconds\tverification_seconds\tvalid\tminimum_residual")
        for row in rows
            println(file, join((row.experiment, row.continuous_dimension,
                row.certificate_degree, row.alpha, row.build_seconds,
                row.solve_seconds, row.verification_seconds, row.valid,
                row.minimum_residual), '\t'))
        end
    end
    println("Benchmark results written to ", output_path)
    for row in rows
        println(row)
    end
    all(row.valid for row in rows) || exit(2)
    return rows
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_benchmarks()
end
