#!/usr/bin/env julia
#=
Isolated Julia worker for controller search, CPC synthesis and exact verification.

    work N method output_dir seed candidates solver_seconds [uniform|allocated|auto]
    verify result.toml output_dir
    show result.toml [physical|normalized|centered] [decimal|exact] [subsystem|all]

Progress is written before expensive phases so an external resource guard can
recover completed work and identify the interrupted stage without an iterate.
=#
include(joinpath(@__DIR__, "Synthesis_Direct_CPC.jl"))
using Printf

const TIMED_PHASES =
    ("controller_search", "physical_reconstruction", "SOS_build", "SOS_solve", "exact_verification")
const WORKER_DIRECTORY = Ref("")
const WORKER_LEDGER = Ref(Dict{String, Any}())

"Replace whole ledgers atomically; the supervisor must never parse a partial file."
function write_toml_atomic(path, data)
    mkpath(dirname(path))
    open(path * ".tmp", "w") do io
        TOML.print(io, data; sorted = true)
    end
    mv(path * ".tmp", path; force = true)
end

function save_checkpoint!()
    write_toml_atomic(joinpath(WORKER_DIRECTORY[], "progress.toml"), WORKER_LEDGER[])
end

"""
Time a phase even when it throws. Rejected-attempt costs remain part of the
phase totals; adding their subtotal again would double charge failed searches.
"""
function time_phase!(operation, phase)
    ledger = WORKER_LEDGER[]
    ledger["phase"] = phase
    ledger["phase_started_unix"] = time()
    save_checkpoint!()
    started = time_ns()
    try
        operation()
    finally
        elapsed = (time_ns() - started) * 1e-9
        if get(ledger, "timed_search", false)
            ledger["phase_seconds"][phase] = get(ledger["phase_seconds"], phase, 0.0) + elapsed
            if !isempty(ledger["attempts"])
                attempt = last(ledger["attempts"])
                attempt["phase_seconds"][phase] =
                    get(attempt["phase_seconds"], phase, 0.0) + elapsed
            end
        end
        ledger["last_phase"] = phase
        ledger["phase"] = "between_stages"
        ledger["phase_started_unix"] = time()
        save_checkpoint!()
    end
end

"Hash the physical task, excluding legacy generated gains unused by this search."
function plant_fingerprint(base)
    physical = [(p.L, p.C, p.r, p.G, p.source, p.current_reference) for p in base.modules]
    return bytes2hex(
        sha256(
            string(
                base.N,
                base.seed,
                physical,
                base.conductance,
                base.M,
                base.dt,
                base.current_scale,
                base.voltage_scale,
                base.voltage_reference,
                base.radius,
                base.alpha,
                base.epsilon_local,
                base.initial_lower,
                base.initial_upper,
            ),
        ),
    )
end

"Record a terminal outcome without interpreting resource limits as infeasibility."
function finish_worker!(status, reason; code = 2)
    ledger = WORKER_LEDGER[]
    ledger["status"] = status
    ledger["reason"] = reason
    if status in ("TIMEOUT", "MEMORY_LIMIT")
        ledger["resource_stage"] = get(ledger, "last_phase", "startup")
    end
    ledger["timed_search"] = false
    ledger["finished_unix"] = time()
    ledger["phase"] = "finished"
    save_checkpoint!()
    write_toml_atomic(joinpath(WORKER_DIRECTORY[], "outcome.toml"), ledger)
    println(status, ": ", reason)
    flush(stdout)
    return code
end

function reject_attempt!(attempt, reason, started)
    attempt["outcome"] = "rejected"
    attempt["reason"] = reason
    attempt["elapsed_seconds"] = (time_ns() - started) * 1e-9
    save_checkpoint!()
end

"Rebuild a single campaign result, including every exported exact witness."
function verify_result_file(path)
    saved = TOML.parsefile(path)
    base = CPC.generate_instance(saved["N"], saved["seed"])
    @assert plant_fingerprint(base) == saved["plant_sha256"] "plant fingerprint mismatch"
    verify_exported_certificate(base, saved["result"], Symbol(saved["method"]))
    return true
end

function record_search_configuration!(base, method, solver_seconds, progress_policy)
    ledger = WORKER_LEDGER[]
    ledger["plant_sha256"] = plant_fingerprint(base)
    ledger["controller_coordinates"] = "normalized"
    ledger["SOS_variable_units"] = "original"
    ledger["progress_policy"] = string(progress_policy)
    ledger["backend"] =
        method == :compositional ? "MOSEK exact aggregate clique lift" :
        "MOSEK dense monolithic SOS"
    ledger["solver_LP_limit_seconds"] = 120.0
    ledger["solver_SOS_limit_seconds"] = solver_seconds
    ledger["Julia_version"] = string(VERSION)
    ledger["JuMP_version"] = string(pkgversion(JuMP))
    ledger["MosekTools_version"] = string(pkgversion(CPC.MosekTools))
    ledger["MOSEK_version"] = string(CPC.MosekTools.Mosek.getversion())
    ledger["warmup_seed"] = 20261121
end

"Use unrelated warmup data without recycling a controller, certificate or solver model."
function warmup_worker(method)
    time_phase!("warmup") do
        result = synthesize_with_candidates(
            CPC.generate_instance(2, 20261121),
            method,
            1;
            controller_coordinates = :normalized,
        )
        @assert result["exact_proof_pass"] "independent warmup did not produce an exact certificate"
    end
end

function begin_attempt!(number)
    attempt = Dict{String, Any}(
        "attempt" => number,
        "outcome" => "running",
        "phase_seconds" => Dict{String, Any}(),
    )
    push!(WORKER_LEDGER[]["attempts"], attempt)
    save_checkpoint!()
    return attempt
end

function record_controller!(attempt, controller)
    attempt["controller_LP_count"] = controller.LP_count
    attempt["controller_offset_exact"] = fraction.(controller.a)
    attempt["controller_ki_exact"] = fraction.(controller.ki)
    attempt["controller_kv_exact"] = fraction.(controller.kv)
end

function record_problem_dimensions!(problem, method)
    ledger = WORKER_LEDGER[]
    ledger["scalar_variables"] = length(problem.variables)
    ledger["SOS_conditions"] = length(problem.sos_constraints)
    ledger["coefficient_identities"] = length(problem.coefficient_identities)
    ledger["PSD_block_orders"] = [size(block.ids, 1) for block in problem.psd_blocks]
    if method == :compositional
        @assert any(occursin("Gamma_clique", block.name) for block in problem.psd_blocks)
        @assert any(occursin("Upsilon_clique", block.name) for block in problem.psd_blocks)
    end
end

"Include repair, all exact checks and physical witness construction, excluding file I/O."
function verify_candidate_result(problem, data, method, raw; epsilon_allocation = nothing)
    repaired = repair_exact_solution(problem, raw)
    @assert verify_exact_solution(problem, repaired.u)
    tampered = copy(repaired.u)
    tampered[1] += 1
    @assert !verify_exact_solution(problem, tampered) "tampering was not detected"
    return export_certificate(
        problem,
        data,
        method,
        repaired,
        raw;
        epsilon_allocation = epsilon_allocation,
    )
end

function save_accepted_result!(base, method, progress_policy, result)
    ledger = WORKER_LEDGER[]
    saved = Dict{String, Any}(
        "N" => base.N,
        "seed" => base.seed,
        "method" => string(method),
        "plant_sha256" => ledger["plant_sha256"],
        "result" => result,
        "controller_coordinates" => "normalized",
        "backend" => ledger["backend"],
        "specification" => "F G (a_1 OR ... OR a_N)",
        "progress_policy" => string(progress_policy),
    )
    time_phase!("file_export") do
        write_toml_atomic(joinpath(WORKER_DIRECTORY[], "result.toml"), saved)
    end
end

"Run the established candidate policy; only an exact CPC witness can accept a candidate."
function run_worker_search(N, method, seed, budget, solver_seconds; progress_policy = :uniform)
    ledger = WORKER_LEDGER[]
    base = CPC.generate_instance(N, seed)
    record_search_configuration!(base, method, solver_seconds, progress_policy)
    warmup_worker(method)
    GC.gc()
    ledger["phase_seconds"] = Dict{String, Any}(phase => 0.0 for phase in TIMED_PHASES)
    ledger["timed_search"] = true
    ledger["search_started_unix"] = time()
    search_started = time_ns()
    for number in 1:budget
        attempt_started = time_ns()
        attempt = begin_attempt!(number)
        controller = time_phase!("controller_search") do
            generate_controller_candidate(base, number; coordinates = :normalized)
        end
        record_controller!(attempt, controller)
        data = try
            time_phase!("physical_reconstruction") do
                reconstruct_closed_loop(base, controller)
            end
        catch err
            if err isa LinearAlgebra.SingularException
                reject_attempt!(attempt, "unsupported singular coordinate translation", attempt_started)
                continue
            end
            rethrow()
        end
        allocation = try
            select_progress_allocation(data, progress_policy)
        catch err
            if err isa ArgumentError &&
               occursin("no strictly inside-target equilibrium", sprint(showerror, err))
                reject_attempt!(attempt, sprint(showerror, err), attempt_started)
                continue
            end
            rethrow()
        end
        if !isnothing(allocation)
            attempt["epsilon_allocation_exact"] = fraction.(allocation)
            attempt["progress_anchor"] = findfirst(>(0), allocation)
            attempt["theorem"] = "Theorem 3 allocated progress"
        end
        attempt["resolved_progress_policy"] = isnothing(allocation) ? "uniform" : "allocated"
        problem = time_phase!("SOS_build") do
            build_certificate_problem(data, method; epsilon_allocation = allocation)
        end
        set_optimizer_attribute(problem.model, "MSK_DPAR_OPTIMIZER_MAX_TIME", solver_seconds)
        record_problem_dimensions!(problem, method)
        time_phase!("SOS_solve") do
            optimize!(problem.model)
        end
        status = string(termination_status(problem.model))
        attempt["solver_status"] = status
        ledger["solver_status"] = status
        if status in ("TIME_LIMIT", "MEMORY_LIMIT")
            attempt["outcome"] = "resource_limit"
            attempt["elapsed_seconds"] = (time_ns() - attempt_started) * 1e-9
            ledger["search_elapsed_seconds"] = (time_ns() - search_started) * 1e-9
            return finish_worker!(
                status == "TIME_LIMIT" ? "TIMEOUT" : "MEMORY_LIMIT",
                "SOS returned $status",
            )
        end
        if status in (
            "INFEASIBLE",
            "ALMOST_INFEASIBLE",
            "INFEASIBLE_OR_UNBOUNDED",
            "LOCALLY_INFEASIBLE",
            "ITERATION_LIMIT",
            "SLOW_PROGRESS",
        )
            reject_attempt!(attempt, "SOS returned $status", attempt_started)
            continue
        end
        status in ("OPTIMAL", "ALMOST_OPTIMAL") || error("Unexpected SOS solver status $status")
        @assert has_values(problem.model) "successful status without primal values"
        raw = value.(problem.variables)
        result = try
            time_phase!("exact_verification") do
                verify_candidate_result(problem, data, method, raw; epsilon_allocation = allocation)
            end
        catch err
            if sprint(showerror, err) == "Exact affine projection could not preserve all SOS cones"
                reject_attempt!(
                    attempt,
                    "numerical candidate did not admit exact cone-preserving repair",
                    attempt_started,
                )
                continue
            end
            rethrow()
        end
        attempt["outcome"] = "accepted"
        attempt["elapsed_seconds"] = (time_ns() - attempt_started) * 1e-9
        ledger["search_elapsed_seconds"] = (time_ns() - search_started) * 1e-9
        ledger["timed_search"] = false
        ledger["exact_proof_pass"] = true
        save_accepted_result!(base, method, progress_policy, result)
        return finish_worker!("CERTIFIED", "$status with exact proof"; code = 0)
    end
    ledger["search_elapsed_seconds"] = (time_ns() - search_started) * 1e-9
    return finish_worker!(
        "NO_CERTIFICATE",
        "candidate budget exhausted; no nonexistence claim";
        code = 2,
    )
end

"Expand a symmetric homogeneous matrix without losing the factor two on cross terms."
function certificate_terms(H)
    issymmetric(H) || throw(ArgumentError("certificate matrix must be symmetric"))
    terms = Tuple{Rat, Int, Int}[]
    push!(terms, (H[1, 1], 0, 0))
    for i in 1:(size(H, 1) - 1)
        push!(terms, (2H[1, i + 1], 0, i))
    end
    for i in 1:(size(H, 1) - 1)
        push!(terms, (H[i + 1, i + 1], i, i))
        for j in (i + 1):(size(H, 1) - 1)
            push!(terms, (2H[i + 1, j + 1], i, j))
        end
    end
    filter(term -> !iszero(first(term)), terms)
end

function certificate_variables(N, method, name, coordinates)
    indices = method == "monolithic" ? (1:N) : (parse(Int, split(name, '_')[2]),)
    names = String[]
    for n in indices
        append!(
            names,
            coordinates == :physical ? ["i_$n", "v_$n"] :
            [
                "$(coordinates == :centered ? "xi" : "z")_$(n)_1",
                "$(coordinates == :centered ? "xi" : "z")_$(n)_2",
            ],
        )
    end
    names
end

function certificate_monomial(variables, i, j)
    i == 0 && j == 0 && return ""
    i == 0 && return variables[j]
    i == j && return variables[i] * "^2"
    variables[i] * "*" * variables[j]
end

function certificate_number(value; exact = false)
    exact && return fraction(value)
    Printf.format(Printf.Format("%.10g"), BigFloat(value))
end

"Print rounded coefficients for inspection, or the original rational coefficients exactly."
function print_certificate_polynomial(io, name, H, variables; exact = false)
    line = name * (exact ? " = " : " ≈ ")
    terms = certificate_terms(H)
    isempty(terms) && return println(io, line, "0")
    for (index, (coefficient, i, j)) in enumerate(terms)
        sign = coefficient < 0 ? "-" : "+"
        prefix = index == 1 ? (coefficient < 0 ? "-" : "") : " $sign "
        magnitude = certificate_number(abs(coefficient); exact = exact)
        monomial = certificate_monomial(variables, i, j)
        term = magnitude * (isempty(monomial) ? "" : "*" * monomial)
        if index > 1 && length(line) + length(prefix) + length(term) > 100
            println(io, line)
            line = "    $sign " * term
        else
            line *= prefix * term
        end
    end
    println(io, line)
end

"""
    print_saved_certificates(io, path; coordinates=:physical, exact=false, subsystem=nothing)

Display the saved B,V polynomials without rebuilding or solving an SOS problem.
Always start from the repaired rational coefficient matrices. Decimal rounding
is only a display choice; it must not replace the exact witness in verification.
"""
function print_saved_certificates(
    io,
    path;
    coordinates = :physical,
    exact = false,
    subsystem = nothing,
)
    fields = Dict(
        :physical => "original_physical_coefficient_matrix_exact",
        :normalized => "original_normalized_coefficient_matrix_exact",
        :centered => "centered_coefficient_matrix_exact",
    )
    haskey(fields, coordinates) || throw(ArgumentError("unknown certificate coordinates"))
    saved = TOML.parsefile(path)
    N, method, result = saved["N"], saved["method"], saved["result"]
    method in ("monolithic", "compositional") || throw(ArgumentError("unknown saved method"))
    if !isnothing(subsystem)
        method == "compositional" ||
            throw(ArgumentError("subsystem filtering is compositional only"))
        1 <= subsystem <= N || throw(ArgumentError("subsystem must lie in 1:$N"))
    end
    names = collect(keys(result["polynomials"]))
    if !isnothing(subsystem)
        filter!(name -> parse(Int, split(name, '_')[2]) == subsystem, names)
    end
    sort!(
        names;
        by = name -> (
            method == "compositional" ? parse(Int, split(name, '_')[2]) : 0,
            startswith(name, "B_") ? 0 : 1,
            endswith(name, "_bad") ? 0 : 1,
        ),
    )
    println(io, "Saved CPC: N=$N, method=$method, coordinates=$coordinates")
    println(io, "bad: shared priority 1; good: shared priority 2; V is the priority-1 rank.")
    println(
        io,
        exact ? "Exact rational coefficients." :
        "Approximate display: 10 significant digits. Use exact coefficients for verification.",
    )
    println(
        io,
        "Stored B,V are unscaled; global progress epsilon = ",
        result["epsilon_global_exact"],
    )
    println(io, "Unit decrease: divide monolithic B,V, or the sums of local B,V, by epsilon.")
    if coordinates != :physical
        println(io, "z_n is the normalized physical state defined in the plant model.")
    end
    if coordinates == :centered
        indices = isnothing(subsystem) ? (1:N) : (subsystem,)
        for n in indices
            center = [
                certificate_number(parse_fraction(value); exact = exact) for
                value in result["equilibrium_exact"][(2n - 1):2n]
            ]
            println(io, "xi_$n = z_$n - (", join(center, ", "), ")")
        end
    end
    for name in names
        rows = result["polynomials"][name][fields[coordinates]]
        all(length(row) == length(rows) for row in rows) ||
            throw(ArgumentError("certificate matrix must be square"))
        H = [parse_fraction(rows[i][j]) for i in eachindex(rows), j in eachindex(rows)]
        variables = certificate_variables(N, method, name, coordinates)
        size(H, 1) == length(variables) + 1 ||
            throw(ArgumentError("certificate dimension mismatch"))
        println(io)
        print_certificate_polynomial(io, name, H, variables; exact = exact)
    end
    nothing
end

function show_certificate_command(args)
    2 <= length(args) <= 5 || throw(
        ArgumentError(
            "usage: show result.toml [physical|normalized|centered] [decimal|exact] [subsystem|all]",
        ),
    )
    coordinates = length(args) >= 3 ? Symbol(args[3]) : :physical
    style = length(args) >= 4 ? args[4] : "decimal"
    style in ("decimal", "exact") || throw(ArgumentError("display format must be decimal or exact"))
    subsystem = length(args) == 5 && args[5] != "all" ? parse(Int, args[5]) : nothing
    print_saved_certificates(
        stdout,
        args[2];
        coordinates = coordinates,
        exact = style == "exact",
        subsystem = subsystem,
    )
    0
end

"Read-only display bypasses worker ledgers so inspecting a witness creates no run artifacts."
function benchmark_main(args = ARGS)
    mode = args[1]
    if mode == "show"
        try
            return show_certificate_command(args)
        catch err
            showerror(stderr, err)
            println(stderr)
            return 2
        end
    end
    directory = mode == "verify" ? args[3] : args[4]
    WORKER_DIRECTORY[] = directory
    mkpath(directory)
    WORKER_LEDGER[] = Dict{String, Any}(
        "mode" => mode,
        "status" => "RUNNING",
        "phase" => "startup",
        "phase_started_unix" => time(),
        "timed_search" => false,
        "phase_seconds" => Dict{String, Any}(),
        "attempts" => Dict{String, Any}[],
        "conda_environment" => get(ENV, "CONDA_DEFAULT_ENV", ""),
    )
    BLAS.set_num_threads(1)
    try
        if mode == "verify"
            elapsed = @elapsed time_phase!("reload_verification") do
                verify_result_file(args[2])
            end
            WORKER_LEDGER[]["reload_verification_seconds"] = elapsed
            WORKER_LEDGER[]["exact_proof_pass"] = true
            return finish_worker!("VERIFIED", "fresh-process reload passed"; code = 0)
        end
        @assert mode == "work"
        N = parse(Int, args[2])
        method = Symbol(args[3])
        @assert method in (:monolithic, :compositional)
        WORKER_LEDGER[]["N"] = N
        WORKER_LEDGER[]["method"] = string(method)
        policy = length(args) >= 8 ? Symbol(args[8]) : :uniform
        @assert policy in (:uniform, :allocated, :auto)
        return run_worker_search(
            N,
            method,
            parse(Int, args[5]),
            parse(Int, args[6]),
            parse(Float64, args[7]);
            progress_policy = policy,
        )
    catch err
        showerror(stderr, err, catch_backtrace())
        println(stderr)
        message = sprint(showerror, err)
        if err isa OutOfMemoryError
            return finish_worker!("MEMORY_LIMIT", message)
        elseif startswith(message, "controller LP: TIME_LIMIT")
            return finish_worker!("TIMEOUT", message)
        elseif startswith(message, "controller LP: MEMORY_LIMIT")
            return finish_worker!("MEMORY_LIMIT", message)
        elseif startswith(message, "controller LP: INFEASIBLE")
            return finish_worker!("NO_CERTIFICATE", message)
        end
        return finish_worker!("FATAL", message; code = 3)
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && exit(benchmark_main())
