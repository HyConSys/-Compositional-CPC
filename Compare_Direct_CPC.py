#!/usr/bin/env python3
"""Prepare, supervise, summarize and watch isolated direct CPC scaling runs.

Preparation starts no Julia processes. The launcher schedules the Julia
implementations; controller search and certificate computation run in Julia.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = ("compositional", "monolithic")
PHASES = (
    "controller_search",
    "physical_reconstruction",
    "SOS_build",
    "SOS_solve",
    "exact_verification",
)
RESOURCE = {"TIMEOUT", "MEMORY_LIMIT"}
TERMINAL = {"COMPLETED", "STOPPED_RESOURCE_LIMIT", "FATAL_ERROR", "STOPPED_BY_USER"}
SOURCES = (
    "Benchmark_Direct_CPC.jl",
    "Synthesis_Direct_CPC.jl",
    "Synthesis_Buck_CPC.jl",
    "Compare_Direct_CPC.py",
    "Project.toml",
    "Manifest.toml",
)


def resource_limit_reason(progress, config, now, start, peak, campaign):
    """Keep stop, memory, whole-worker and timed-search guards in their established priority order."""
    if campaign and (campaign / "STOP_REQUESTED").exists():
        return "STOPPED_BY_USER"
    if peak > config["memory_gib"] * 1024**3:
        return "MEMORY_LIMIT"
    if now - start > config["worker_timeout_seconds"]:
        return "TIMEOUT"
    if (
        progress.get("timed_search")
        and time.time() - progress["search_started_unix"]
        > config["search_timeout_seconds"]
    ):
        return "TIMEOUT"
    return None


def update_heartbeat(campaign, directory, process_id, progress, peak, now, previous):
    """Expose the current phase even when the worker is still constructing a model."""
    if campaign and now - previous >= 5:
        mark(
            campaign,
            "RUNNING",
            supervisor_pid=os.getpid(),
            worker_pid=process_id,
            active_directory=str(directory.relative_to(campaign)),
            phase=progress.get("phase", "startup"),
            N=progress.get("N"),
            method=progress.get("method"),
            peak_worker_rss_bytes=peak,
        )
        return now
    return previous


def supervision_record(process, directory, reason, guard_phase, start, peak, command):
    """Retain partial elapsed work separately when a resource guard interrupts a phase."""
    progress = read_progress(directory)
    status = reason or progress.get("status", "FATAL")
    if not reason and (
        status == "RUNNING"
        or (process.returncode != 0 and status in ("CERTIFIED", "VERIFIED"))
    ):
        status = "FATAL"
    if not reason and status not in {
        "CERTIFIED",
        "VERIFIED",
        "NO_CERTIFICATE",
        "FATAL",
        *RESOURCE,
    }:
        status = "FATAL"
    phases = dict(progress.get("phase_seconds", {}))
    partial_phase = guard_phase if guard_phase in PHASES else ""
    if reason and progress.get("timed_search") and (progress.get("phase") in PHASES):
        partial_phase = progress["phase"]
        phases[partial_phase] = phases.get(partial_phase, 0.0) + max(
            0.0, time.time() - progress["phase_started_unix"]
        )
    attempts = progress.get("attempts", [])
    record = dict(
        status=status,
        guard_reason=reason,
        returncode=process.returncode,
        wall_seconds=time.monotonic() - start,
        peak_rss_bytes=peak,
        phase_seconds=phases,
        partial_phase=partial_phase,
        attempt_count=len(attempts),
        rejected_attempt_count=sum(
            (record.get("outcome") == "rejected" for record in attempts)
        ),
        rejected_attempt_seconds=sum(
            (
                record.get("elapsed_seconds", 0.0)
                for record in attempts
                if record.get("outcome") == "rejected"
            )
        ),
        solver_status=progress.get("solver_status", ""),
        exact_proof_pass=progress.get("exact_proof_pass", False),
        plant_sha256=progress.get("plant_sha256", ""),
        search_elapsed_seconds=progress.get("search_elapsed_seconds"),
        reason=progress.get(
            "reason",
            "External resource guard" if reason else "Worker exited without an outcome",
        ),
        reload_verification_seconds=progress.get("reload_verification_seconds"),
        scalar_variables=progress.get("scalar_variables"),
        SOS_conditions=progress.get("SOS_conditions"),
        coefficient_identities=progress.get("coefficient_identities"),
        largest_PSD_block=max(progress.get("PSD_block_orders", [0])),
        resource_stage=guard_phase if reason else progress.get("resource_stage", ""),
        command=command,
    )
    if reason and progress.get("timed_search"):
        record["search_elapsed_seconds"] = max(
            0.0, time.time() - progress["search_started_unix"]
        )
    write_json(directory / "supervision.json", record)
    return record


def comparison_row(network_size, method, record):
    """Keep phase totals and rejected-attempt subtotals separate to avoid double charging."""
    phases = record.get("phase_seconds", {})
    return dict(
        N=network_size,
        method=method,
        status=record.get("status", "NOT_RUN"),
        solver_status=record.get("solver_status", ""),
        exact_proof_pass=record.get("exact_proof_pass", False),
        fresh_reload_pass=record.get("fresh_reload_pass", False),
        controller_search_s=phases.get("controller_search"),
        physical_reconstruction_s=phases.get("physical_reconstruction"),
        SOS_build_s=phases.get("SOS_build"),
        SOS_solve_s=phases.get("SOS_solve"),
        exact_verification_s=phases.get("exact_verification"),
        fresh_reload_s=record.get("reload_verification_seconds"),
        rejected_attempts=record.get("rejected_attempt_count"),
        rejected_attempts_s=record.get("rejected_attempt_seconds"),
        attempts=record.get("attempt_count"),
        search_total_s=record.get("search_elapsed_seconds"),
        orchestration_s=max(
            0.0, record["search_elapsed_seconds"] - sum(phases.values())
        )
        if record.get("search_elapsed_seconds") is not None
        else None,
        worker_wall_s=record.get("wall_seconds"),
        peak_RSS_GiB=record.get("peak_rss_bytes", 0) / 1024**3 if record else None,
        partial_phase=record.get("partial_phase", ""),
        resource_stage=record.get("resource_stage", ""),
        scalar_variables=record.get("scalar_variables"),
        largest_PSD_block=record.get("largest_PSD_block"),
        SOS_conditions=record.get("SOS_conditions"),
        coefficient_identities=record.get("coefficient_identities"),
        reason=record.get("reason", ""),
        plant_sha256=record.get("plant_sha256", ""),
    )


def format_value(value):
    """Use the report precision and missing-value marker established by the published run."""
    return (
        "—"
        if value is None
        else f"{value:.6g}"
        if isinstance(value, float)
        else str(value)
    )


def render_comparison(campaign, config, rows):
    """Render observations without treating resource limits as infeasibility or completed solves."""
    lines = [
        "# Direct controller-and-CPC scaling comparison",
        "",
        "Status: " + read_json(campaign / "status.json").get("status", "UNKNOWN"),
        "",
        "Good = at least one converter inside its target disk; bad = all converters outside.",
        "Normalized states/controller coordinates; original SOS unknown units; free degree-two B,V.",
        "MOSEK: exact clique lift for both compositional aggregate supplies; dense monolithic SOS.",
        "Progress policy: "
        + config.get("progress_policy", "uniform")
        + ". Auto retains uniform steps when all local equilibria are strictly inside their disks, otherwise uses Theorem 3 allocated progress (nonnegative local steps, positive aggregate).",
        f"Sizes {config['sizes']}; seed {config['seed']}; {config['candidates']} controller candidates; one repetition.",
        f"Limits per method: {config['memory_gib']} GiB sampled group RSS, {config['search_timeout_seconds']} s timed search, {config['worker_timeout_seconds']} s worker wall including startup/warmup; {config['solver_timeout_seconds']} s per SOS call.",
        "Each controller support LP retains the implementation's 120 s solver limit. One solver/BLAS thread.",
        "",
        "| N | Method | Outcome | Solver | Controller (s) | SOS build (s) | SOS solve (s) | Exact proof (s) | Reload (s) | Rejected count / subtotal (s) | Search total (s) | Peak GiB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in rows:

        def phase(field, name):
            return ("≥" if record["partial_phase"] == name else "") + format_value(
                record[field]
            )

        rejected = (
            "—"
            if record["rejected_attempts"] is None
            else f"{record['rejected_attempts']} / {format_value(record['rejected_attempts_s'])}"
        )
        cells = [
            record["N"],
            record["method"],
            record["status"],
            record["solver_status"] or "—",
            phase("controller_search_s", "controller_search"),
            phase("SOS_build_s", "SOS_build"),
            phase("SOS_solve_s", "SOS_solve"),
            phase("exact_verification_s", "exact_verification"),
            format_value(record["fresh_reload_s"]),
            rejected,
            format_value(record["search_total_s"]),
            format_value(record["peak_RSS_GiB"]),
        ]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    lines += [
        "",
        "Phase columns include all controller attempts. Rejected-attempt time is a separate **subset** of the total, not an extra additive cost.",
        "Exact proof includes affine rational repair, cone/identity checks, original aggregate checks and physical proof-witness construction; file writing is excluded.",
        "Search total additionally includes physical reconstruction and orchestration/checkpoint overhead (including any first-call benchmark-wrapper compilation); both are separate CSV columns. Startup, explicit independent N=2 solver/model warmup, file export, and fresh-process reload are separate.",
        "A ≥ entry is partial elapsed work at an external resource interruption; it is not a completed solve time. See resource_stage/partial_phase in CSV.",
        "TIMEOUT and MEMORY_LIMIT are censored outcomes, not solution times or evidence of infeasibility. ALMOST_OPTIMAL remains labeled as such.",
        "CERTIFIED requires an exact proof and fresh-process reload. A zero feasibility objective is not an optimal-gain claim.",
        "One-seed feasibility measurements do not establish a statistical or asymptotic complexity claim.",
        "",
        "Per-method folders preserve worker/reload logs, progress, all attempts and exact result.toml. Config and frozen sources are retained.",
    ]
    for network_size in config["sizes"]:
        pair = [record for record in rows if record["N"] == network_size]
        comp = next((record for record in pair if record["method"] == "compositional"))
        mono = next((record for record in pair if record["method"] == "monolithic"))
        if mono["status"] in RESOURCE and comp["status"] == "CERTIFIED":
            lines += [
                "",
                f"At N={network_size}, monolithic reached {mono['status']} (wall {format_value(mono['worker_wall_s'])} s, peak {format_value(mono['peak_RSS_GiB'])} GiB); compositional was certified with solver status {comp['solver_status']} in {format_value(comp['search_total_s'])} s plus reload {format_value(comp['fresh_reload_s'])} s.",
            ]
    return "\n".join(lines) + "\n"


def run_formulation(campaign, config, network_size, method):
    """Require fresh-process proof verification before publishing CERTIFIED."""
    source_check(campaign, config)
    directory = campaign / f"N{network_size}_{method}"
    command = julia_command(
        campaign,
        [
            "work",
            network_size,
            method,
            directory,
            config["seed"],
            config["candidates"],
            config["solver_timeout_seconds"],
            config.get("progress_policy", "uniform"),
        ],
    )
    record = supervise(command, directory, config, campaign)
    record.update(N=network_size, method=method, fresh_reload_pass=False)
    if record["status"] == "CERTIFIED":
        if not record["exact_proof_pass"] or not (directory / "result.toml").exists():
            raise RuntimeError("Claimed certification without exact proof/result")
        reload_dir = directory / "Reload"
        command = julia_command(
            campaign, ["verify", directory / "result.toml", reload_dir]
        )
        check = supervise(command, reload_dir, config, campaign)
        record["reload_verification_seconds"] = check["reload_verification_seconds"]
        record["peak_rss_bytes"] = max(
            record["peak_rss_bytes"], check["peak_rss_bytes"]
        )
        record["wall_seconds"] += check["wall_seconds"]
        record["fresh_reload_pass"] = (
            check["status"] == "VERIFIED" and check["exact_proof_pass"]
        )
        if not record["fresh_reload_pass"]:
            record["status"] = (
                check["status"]
                if check["status"] in RESOURCE | {"STOPPED_BY_USER"}
                else "FATAL"
            )
            record["reason"] = "Fresh reload: " + check["reason"]
            record["resource_stage"] = "fresh_reload"
    if record["status"] in RESOURCE and (not record.get("resource_stage")):
        record["resource_stage"] = read_progress(directory).get("phase", "unknown")
    write_json(directory / "run_status.json", record)
    return record


def default_results_directory():
    """Resolve the configurable output root without embedding a machine-specific path."""
    return Path(os.environ.get("CPC_RESULTS_DIR", ROOT / "Results"))


def build_argument_parser():
    """Define the public campaign CLI without starting work."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser(
        "prepare", help="Freeze sources and settings; run no experiments"
    )
    prepare_parser.add_argument(
        "--sizes", nargs="+", type=int, default=[2, 10, 50, 100]
    )
    prepare_parser.add_argument("--seed", type=int, default=20261122)
    prepare_parser.add_argument("--candidates", type=int, default=3)
    prepare_parser.add_argument("--memory-gib", type=float, default=8)
    prepare_parser.add_argument("--search-timeout", type=float, default=600)
    prepare_parser.add_argument("--worker-timeout", type=float, default=900)
    prepare_parser.add_argument("--solver-timeout", type=float, default=120)
    prepare_parser.add_argument(
        "--progress-policy", choices=["uniform", "allocated", "auto"], default="auto"
    )
    prepare_parser.add_argument("--output", type=Path)
    for action in ("run", "launch", "watch", "status", "summarize"):
        sub.add_parser(action).add_argument("campaign", type=Path)
    return parser


def write_json(path, value):
    """Replace a checkpoint atomically so a monitor never reads half a JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path, default=None):
    """Return the supplied fallback while a status file is absent or unreadable."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def read_progress(directory):
    """Parse an atomically replaced ledger; a malformed existing ledger is fatal."""
    path = Path(directory) / "progress.toml"
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text())


def digest(path):
    """Hash frozen executable sources to reject mid-campaign source changes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mark(campaign, status, **fields):
    """Publish campaign state and the corresponding terminal flag."""
    payload = dict(status=status, updated_unix=time.time(), **fields)
    write_json(campaign / "status.json", payload)
    if status != "READY":
        (campaign / "READY").unlink(missing_ok=True)
    if status in TERMINAL or status == "READY":
        write_json(campaign / status, payload)
    return payload


def prepare(args):
    """Freeze source bytes and settings without launching a Julia process."""
    if args.sizes != sorted(set(args.sizes)) or min(args.sizes) < 2:
        raise ValueError("Sizes must be increasing, unique and at least 2")
    if (
        min(
            args.memory_gib,
            args.search_timeout,
            args.worker_timeout,
            args.solver_timeout,
        )
        <= 0
        or args.candidates < 1
    ):
        raise ValueError("Resource limits and candidate budget must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    campaign = (
        args.output or default_results_directory() / f"Direct_Scaling_{stamp}"
    ).resolve()
    campaign.mkdir(parents=True, exist_ok=False)
    snapshot = campaign / "Source_Snapshot"
    snapshot.mkdir()
    hashes = {}
    for name in SOURCES:
        (snapshot / name).write_bytes((ROOT / name).read_bytes())
        hashes[name] = digest(snapshot / name)
    config = dict(
        sizes=args.sizes,
        seed=args.seed,
        candidates=args.candidates,
        memory_gib=args.memory_gib,
        search_timeout_seconds=args.search_timeout,
        worker_timeout_seconds=args.worker_timeout,
        solver_timeout_seconds=args.solver_timeout,
        controller_LP_timeout_seconds=120,
        poll_seconds=0.5,
        source_sha256=hashes,
        specification="F G (a_1 OR ... OR a_N)",
        controller_coordinates="normalized",
        SOS_variable_units="original",
        alpha="1/100",
        epsilon_local="1/1000",
        degree=2,
        solver="MOSEK",
        composition_formulation="exact aggregate clique lift",
        threads=1,
        repetitions=1,
        progress_policy=getattr(args, "progress_policy", "auto"),
        timing_note="Phase totals include every attempt. Rejected time is a subset, never added twice.",
        memory_note="Sampled process-group RSS guard, including startup and warmup; not an OS hard cap.",
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(campaign / "config.json", config)
    mark(campaign, "READY", reason="Prepared; full run has not been started")
    summarize(campaign)
    print(campaign)
    return campaign


def source_check(campaign, config):
    """Reject changed frozen sources before a worker or watcher is started."""
    for name, expected in config["source_sha256"].items():
        if digest(campaign / "Source_Snapshot" / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")


def group_rss_bytes(process_group_id):
    """Sum Linux process-group RSS, tolerating processes that exit during sampling."""
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != process_group_id:
                continue
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            pass
    return total


def stop_group(process):
    """Terminate the entire worker process group, including any remaining children."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def julia_command(campaign, arguments):
    """Run Julia with the frozen project; keep process execution shell-free."""
    project = campaign / "Source_Snapshot"
    if not (project / "Project.toml").is_file():
        project = ROOT
    return [
        "env",
        "GLIBC_TUNABLES=glibc.rtld.execstack=2",
        os.environ.get("CPC_JULIA_EXE", "julia"),
        "--startup-file=no",
        f"--project={project}",
        str(campaign / "Source_Snapshot" / "Benchmark_Direct_CPC.jl"),
        *map(str, arguments),
    ]


def supervise(command, directory, config, campaign=None):
    """Monitor a separate process group during startup, model construction, solving and verification."""
    directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    peak = 0
    reason = None
    guard_phase = ""
    progress = {}
    last_heartbeat = 0.0
    with (directory / "worker.log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=directory,
        )
        write_json(
            directory / "process.json",
            dict(pid=process.pid, command=command, started_unix=time.time()),
        )
        try:
            while process.poll() is None:
                now = time.monotonic()
                peak = max(peak, group_rss_bytes(process.pid))
                progress = read_progress(directory)
                reason = resource_limit_reason(
                    progress, config, now, start, peak, campaign
                )
                if reason:
                    guard_phase = progress.get("phase", "startup")
                    stop_group(process)
                    break
                last_heartbeat = update_heartbeat(
                    campaign,
                    directory,
                    process.pid,
                    progress,
                    peak,
                    now,
                    last_heartbeat,
                )
                time.sleep(config.get("poll_seconds", 0.5))
            process.wait()
        except BaseException:
            stop_group(process)
            raise
        finally:
            if process.poll() is None:
                stop_group(process)
    return supervision_record(
        process, directory, reason, guard_phase, start, peak, command
    )


def after_pair(records):
    """A single resource limit does not suppress the counterpart or larger network sizes."""
    if any((record["status"] == "FATAL" for record in records)):
        return "FATAL_ERROR"
    if any((record["status"] == "STOPPED_BY_USER" for record in records)):
        return "STOPPED_BY_USER"
    if len(records) == 2 and all((record["status"] in RESOURCE for record in records)):
        return "STOPPED_RESOURCE_LIMIT"
    return None


def summarize(campaign):
    """Regenerate the stable CSV and Markdown report from saved worker outcomes."""
    config = read_json(campaign / "config.json")
    rows = []
    for network_size in config["sizes"]:
        for method in METHODS:
            path = campaign / f"N{network_size}_{method}" / "run_status.json"
            record = read_json(path)
            rows.append(comparison_row(network_size, method, record))
    with (campaign / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (campaign / "comparison.md").write_text(render_comparison(campaign, config, rows))
    return rows


def run(campaign):
    """Run each prepared campaign once; preserve fatal and paired-resource stopping rules."""
    config = read_json(campaign / "config.json")
    if read_json(campaign / "status.json").get("status") != "READY":
        raise RuntimeError(
            "Only a prepared READY campaign may start; create a new campaign to rerun"
        )
    lock = campaign / "run.lock"
    lock_descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock_descriptor, str(os.getpid()).encode())
    os.close(lock_descriptor)
    previous_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)
    }

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Supervisor received signal {signum}")

    for sig in previous_handlers:
        signal.signal(sig, interrupted)
    try:
        source_check(campaign, config)
        mark(campaign, "RUNNING", supervisor_pid=os.getpid())
        for index, network_size in enumerate(config["sizes"]):
            pair = []
            order = METHODS if index % 2 == 0 else METHODS[::-1]
            for method in order:
                record = run_formulation(campaign, config, network_size, method)
                pair.append(record)
                if (
                    len(pair) == 2
                    and pair[0]["plant_sha256"]
                    and pair[1]["plant_sha256"]
                    and (pair[0]["plant_sha256"] != pair[1]["plant_sha256"])
                ):
                    raise RuntimeError(f"Plant mismatch at N={network_size}")
                summarize(campaign)
                print(
                    f"N={network_size} {method}: {record['status']}; solver={record['solver_status']}; peak={record['peak_rss_bytes'] / 1024 ** 3:.3f} GiB",
                    flush=True,
                )
                decision = after_pair(pair)
                if decision:
                    mark(
                        campaign,
                        decision,
                        N=network_size,
                        reason=record.get("reason", decision),
                    )
                    summarize(campaign)
                    return 1 if decision == "FATAL_ERROR" else 0
        mark(campaign, "COMPLETED", reason="All requested sizes attempted")
        summarize(campaign)
        return 0
    except KeyboardInterrupt as err:
        mark(campaign, "STOPPED_BY_USER", reason=str(err))
        summarize(campaign)
        return 0
    except BaseException as err:
        mark(campaign, "FATAL_ERROR", reason=f"{type(err).__name__}: {err}")
        summarize(campaign)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def watch(campaign, interval=5.0):
    """Record meaningful state changes and detect a stale supervisor heartbeat."""
    events = campaign / "notifications.jsonl"
    last = None
    while True:
        status = read_json(campaign / "status.json")
        key = (
            status.get("status"),
            status.get("active_directory"),
            status.get("phase"),
        )
        state = dict(
            watcher_pid=os.getpid(), checked_unix=time.time(), observed_status=status
        )
        if (
            status.get("status") == "RUNNING"
            and time.time() - status.get("updated_unix", 0) > 120
        ):
            state["alert"] = "Supervisor heartbeat stale"
            write_json(campaign / "ALERT", state)
        write_json(campaign / "monitor_status.json", state)
        if key != last:
            event = dict(time_utc=datetime.now(timezone.utc).isoformat(), **status)
            with events.open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            last = key
        if status.get("status") in TERMINAL:
            return 1 if status["status"] == "FATAL_ERROR" else 0
        time.sleep(interval)


def launch(campaign):
    """Start detached supervisor and watcher processes from the frozen source snapshot."""
    if read_json(campaign / "status.json").get("status") != "READY":
        raise RuntimeError("Campaign is not READY")
    script = campaign / "Source_Snapshot" / "Compare_Direct_CPC.py"
    source_check(campaign, read_json(campaign / "config.json"))
    lock_descriptor = os.open(
        campaign / "launch.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY
    )
    os.write(lock_descriptor, str(os.getpid()).encode())
    os.close(lock_descriptor)
    with (campaign / "supervisor.log").open("w") as log:
        supervisor = subprocess.Popen(
            [sys.executable, str(script), "run", str(campaign)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    with (campaign / "watcher.log").open("w") as log:
        monitor = subprocess.Popen(
            [sys.executable, str(script), "watch", str(campaign)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    write_json(
        campaign / "background_processes.json",
        dict(supervisor_pid=supervisor.pid, monitor_pid=monitor.pid),
    )
    print(
        json.dumps(
            dict(
                campaign=str(campaign),
                supervisor_pid=supervisor.pid,
                monitor_pid=monitor.pid,
            )
        )
    )


def main(argv: list[str] | None = None):
    """Dispatch the existing CLI, with optional explicit argv for repeatable tests."""
    args = build_argument_parser().parse_args(argv)
    if args.action == "prepare":
        prepare(args)
        return 0
    campaign = args.campaign.resolve()
    if args.action == "run":
        return run(campaign)
    if args.action == "launch":
        launch(campaign)
        return 0
    if args.action == "watch":
        return watch(campaign)
    if args.action == "summarize":
        summarize(campaign)
        return 0
    print(json.dumps(read_json(campaign / "status.json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
