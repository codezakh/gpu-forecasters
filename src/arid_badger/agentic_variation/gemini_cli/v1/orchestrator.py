"""Orchestrator for the Gemini-CLI-on-TriMul harness.

Exposes a single public entry point, :func:`run_experiment`, which builds
the agent container, starts the FastMCP scoring server, primes the seed
kernel's verdict, runs the agent, and returns a :class:`TrimulRunResult`
with the best-of-trajectory speedup computed from the server-side
``trajectory.jsonl`` snapshot.

Callers supply their own ``run_dir`` (the per-run output directory). The
same ``run_experiment`` is used by both single-run and repeated-run
callers — identity of a run is caller-chosen (timestamp or indexed),
the orchestrator only cares that the dir is fresh when handed in.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import docker
import docker.errors
from arid_badger.cache.file_cache import FileCache
from arid_badger.invocation_sink import code_sha256
from arid_badger.trimul.core import SuccessFeedback, TriMulKernelExecutionFeedback
from arid_badger.trimul.seed_kernel import SEED_KERNEL_CODE
from docker.models.containers import Container
from fastmcp import Client
from loguru import logger
from pydantic import TypeAdapter

from .models import BaselineFeedbackEntry, ExperimentConfig, TrimulRunResult
from .prompts import render_system_prompt, render_user_prompt
from .results import RESULT_FILENAME, RUN_DIR_PREFIX, best_record, load_trajectory


# ---------------------------------------------------------------------------
# Wire-format / naming constants — not knobs. Each has exactly one correct
# value; changing any would be a code change, not a configuration change.
# ---------------------------------------------------------------------------

IMAGE_TAG = "arid-badger-gemini-cli-trimul:v1"
_MODULE_DIR = Path(__file__).parent
# Gemini CLI namespaces MCP tools as ``mcp_<server>_<tool>``; the server
# name matches the key under ``mcpServers`` in ``settings.json``.
MCP_TOOL_NAME = "mcp_trimul_score_trimul"
# Where the rendered system prompt lands inside the container. Gemini
# CLI reads it via ``GEMINI_SYSTEM_MD``.
SYSTEM_PROMPT_IN_CONTAINER = "/etc/gemini_trimul/system_prompt.md"
# Docker label key used to bind a container to the run_dir that started
# it. The startup sweep filters on exact-match so orphan cleanup for one
# run_dir cannot touch containers from any other run_dir or caller.
RUN_DIR_LABEL = "arid_badger.run_dir"
# Written once the scoring-server subprocess is spawned; cleared when it
# exits normally. If this file is still on disk at the next run for the
# same run_dir, the process was orphaned by a hard crash.
SERVER_PID_FILENAME = ".server.pid"
# Shared across every run within one output root. Keyed by
# ``(seed_sha, gpu, triton, aggregator)`` so an N-way sweep only pays the
# baseline Modal cost once.
BASELINE_CACHE_DIRNAME = "baseline_cache"
# Module path used to spawn the scoring server as a subprocess. Pinned
# to this library version so a caller on a different harness version
# cannot accidentally reach in.
_SCORING_SERVER_MODULE = "arid_badger.agentic_variation.gemini_cli.v1.trimul_score_server"


def mcp_url_for(port: int) -> str:
    return f"http://127.0.0.1:{port}/mcp/"


# ---------------------------------------------------------------------------
# Atomic writes — any file whose presence is a durability signal (most
# importantly ``result.json``, also the server PID file) must land via
# ``os.replace`` so a crash mid-write can never leave a truncated copy
# that a resume mistakes for a complete one. Same pattern as
# :mod:`arid_badger.cache.file_cache`.
# ---------------------------------------------------------------------------


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    _ = tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode())


# ---------------------------------------------------------------------------
# Partial-run / orphan cleanup. Run identity is caller-supplied
# (``run_dir``), so resume logic is just: presence of ``result.json`` =
# done, anything else = redo. These helpers handle the "redo" side —
# reclaiming resources a previous crashed attempt for *this same*
# ``run_dir`` may have left behind.
# ---------------------------------------------------------------------------


def clear_partial_runs(output_dir: Path) -> None:
    """Remove ``run_*/`` dirs under ``output_dir`` that never produced a result.

    Presence of ``result.json`` is the terminal marker. Anything else in
    a ``run_*/`` dir is derived (trajectory log, kernel copies, agent log)
    and will be regenerated if the run is redone. Safe to call at the
    top of any caller that writes into ``output_dir``.
    """
    if not output_dir.is_dir():
        return
    for child in output_dir.glob(f"{RUN_DIR_PREFIX}*"):
        if not child.is_dir():
            continue
        if (child / RESULT_FILENAME).is_file():
            continue
        logger.info("clearing partial run dir: {p}", p=child)
        shutil.rmtree(child)


def _kill_orphaned_server(run_dir: Path) -> None:
    """Kill the scoring-server subprocess from a prior crashed attempt.

    The PID-file + cmdline match is deliberately belt-and-braces: the
    PID alone is not safe (Linux reuses PIDs), so we only signal a
    process whose ``/proc/<pid>/cmdline`` still contains
    ``trimul_score_server``. Any other state (missing file, stale PID,
    different process) is a silent no-op.
    """
    pid_file = run_dir / SERVER_PID_FILENAME
    if not pid_file.is_file():
        return
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        logger.warning("pid file {p} malformed, removing", p=pid_file)
        pid_file.unlink(missing_ok=True)
        return
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if cmdline_path.is_file():
        cmdline = cmdline_path.read_bytes().decode(errors="replace")
        if "trimul_score_server" in cmdline:
            logger.info("killing orphaned scoring-server pid={pid}", pid=pid)
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(30):
                    time.sleep(0.1)
                    if not Path(f"/proc/{pid}").exists():
                        break
                else:
                    logger.warning("pid={pid} didn't exit in 3s, SIGKILL", pid=pid)
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    pid_file.unlink(missing_ok=True)


def _sweep_orphaned_containers(
    docker_client: docker.DockerClient, run_dir: Path
) -> None:
    """Remove any container labeled with this specific ``run_dir``.

    Label filter is exact-match (Docker API), so sibling callers and
    concurrent runs for other ``run_dir`` values are guaranteed
    untouched. Only meaningful after a crash-and-restart of this same
    run index — a no-op on first attempts.
    """
    filter_value = f"{RUN_DIR_LABEL}={run_dir}"
    for c in docker_client.containers.list(
        all=True, filters={"label": filter_value}
    ):
        logger.info("removing orphaned container {cid}", cid=c.short_id)
        try:
            c.remove(force=True)
        except docker.errors.APIError as exc:
            logger.warning("orphan container remove failed: {e}", e=exc)


def _pick_free_port() -> int:
    """Ask the OS for an unused TCP port on loopback.

    Closed immediately so the scoring server can bind it. A vanishingly
    small TOCTOU window exists between release here and bind in the
    server subprocess; acceptable for coordinated runs on one host.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Run layout — bundled so ``run_agent`` doesn't take five positional Path
# args that read as interchangeable at the call site. All paths are
# absolute (docker-py's bind-mount API rejects relative paths).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunLayout:
    scratch: Path  # bind-mounted as the container's /work (agent cwd)
    agent_home: Path  # bind-mounted as /home/agent (~/.gemini/settings.json lives here)
    run_artifacts_dir: Path  # persistent output dir; survives the container
    system_prompt_host_path: Path  # file bind-mounted read-only into the container


# ---------------------------------------------------------------------------
# Log teeing
# ---------------------------------------------------------------------------


def _tee_subprocess_stream_in_background(
    stream: IO[str],
    log_file: IO[str],
    prefix: str,
) -> threading.Thread:
    """Fan a subprocess's line-buffered stdout to a text log file and loguru.

    The input ``stream`` must have been opened in text mode with line
    buffering — i.e. ``subprocess.Popen(..., bufsize=1, text=True)`` — and
    the child must be running unbuffered (e.g. ``PYTHONUNBUFFERED=1`` for
    Python children). Without both, output arrives in block-sized bursts
    rather than live.
    """

    def pump() -> None:
        for line in stream:
            stripped = line.rstrip("\n")
            _ = log_file.write(line)
            log_file.flush()
            logger.info("{prefix} {line}", prefix=prefix, line=stripped)

    thread = threading.Thread(target=pump, daemon=True, name=f"pump-{prefix}")
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Container build (docker-py)
# ---------------------------------------------------------------------------


def build_image(client: docker.DockerClient) -> None:
    logger.info("building image {tag}", tag=IMAGE_TAG)
    _, build_logs = client.images.build(
        path=str(_MODULE_DIR),
        tag=IMAGE_TAG,
        rm=True,
    )
    for chunk in build_logs:
        if not isinstance(chunk, dict):
            continue
        stream_val = chunk.get("stream")
        if isinstance(stream_val, str):
            text = stream_val.rstrip()
            if text:
                logger.info("[docker-build] {t}", t=text)
    logger.info("image ready: {tag}", tag=IMAGE_TAG)


# ---------------------------------------------------------------------------
# Scoring server lifecycle
# ---------------------------------------------------------------------------


def _wait_for_port(port: int, timeout_s: float) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(
        f"FastMCP server on port {port} didn't come up within {timeout_s}s"
    )


def _http_health_check(url: str) -> tuple[int, str]:
    """Prove that an MCP app — not just any process — is answering on the URL."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status, resp.read(200).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(200).decode("utf-8", errors="replace")


def start_scoring_server(
    config: ExperimentConfig,
    port: int,
    log_path: Path,
    scratch_root: Path,
    trajectory_log: Path,
) -> tuple[subprocess.Popen[str], IO[str]]:
    logger.info(
        "launching FastMCP scoring server (port={port}, gpu={gpu}, scratch={root})",
        port=port,
        gpu=config.gpu,
        root=scratch_root,
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # keep the child from block-buffering stdout
    log_handle = log_path.open("w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            _SCORING_SERVER_MODULE,
            "--port",
            str(port),
            "--gpu",
            config.gpu,
            "--aggregator",
            config.aggregator,
            "--scratch-root",
            str(scratch_root),
            "--trajectory-log",
            str(trajectory_log),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    _ = _tee_subprocess_stream_in_background(
        proc.stdout, log_handle, prefix="[server]"
    )

    _wait_for_port(port, timeout_s=180.0)
    url = mcp_url_for(port)
    status, body_preview = _http_health_check(url)
    logger.info(
        "scoring server answering at {url}: status={status}, body[:80]={preview!r}",
        url=url,
        status=status,
        preview=body_preview[:80],
    )
    return proc, log_handle


# ---------------------------------------------------------------------------
# Baseline priming via FastMCP client
# ---------------------------------------------------------------------------


_FEEDBACK_ADAPTER: TypeAdapter[TriMulKernelExecutionFeedback] = TypeAdapter(
    TriMulKernelExecutionFeedback
)


async def _prime_baseline_async(
    mcp_url: str, relative_path: str
) -> dict[str, Any]:
    client: Client[Any] = Client(mcp_url)
    async with client:
        result = await client.call_tool("score_trimul", {"path": relative_path})
    data = result.data
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected score_trimul result shape: {data!r}")
    return data


def prime_baseline(
    mcp_url: str, relative_path: str
) -> TriMulKernelExecutionFeedback:
    """Score the seed kernel once up front so the agent doesn't burn a turn on it."""
    logger.info("priming baseline: scoring {p} via MCP client", p=relative_path)
    start = time.monotonic()
    raw = asyncio.run(_prime_baseline_async(mcp_url, relative_path))
    elapsed = time.monotonic() - start
    feedback = _FEEDBACK_ADAPTER.validate_python(raw)
    match feedback:
        case SuccessFeedback(aggregated_speedup=speedup):
            logger.info(
                "baseline: success aggregated_speedup={sp:.4f}x (in {el:.1f}s)",
                sp=speedup,
                el=elapsed,
            )
        case _:
            logger.warning(
                "baseline: non-success kind={kind} (in {el:.1f}s)",
                kind=feedback.kind,
                el=elapsed,
            )
    return feedback


# ---------------------------------------------------------------------------
# Agent event logging
# ---------------------------------------------------------------------------


def _summarize_agent_event(line: str) -> str:
    """Turn one NDJSON stream-json line into a short human-readable summary."""
    try:
        obj: Any = json.loads(line)
    except json.JSONDecodeError:
        return f"non-json: {line[:200]}"
    if not isinstance(obj, dict):
        return f"non-dict: {str(obj)[:200]}"
    d: dict[str, Any] = obj
    kind = d.get("type") or d.get("event") or d.get("kind") or "?"
    name = d.get("tool_name") or d.get("name") or d.get("tool") or ""
    suffix = f" name={name}" if name else ""
    for key in ("message", "summary", "content"):
        val = d.get(key)
        if isinstance(val, str) and 0 < len(val) < 200:
            suffix += f" {key}={val!r}"
            break
    return f"event={kind}{suffix}"


def _stream_container_logs(container: Container, raw_log_path: Path) -> Iterator[str]:
    """Yield decoded lines from docker-py's interleaved-bytes log iterator."""
    log_iter: Any = container.logs(stream=True, follow=True, stdout=True, stderr=True)
    with raw_log_path.open("wb") as raw:
        buf = bytearray()
        for chunk in log_iter:
            if not chunk:
                continue
            _ = raw.write(chunk)
            raw.flush()
            buf.extend(chunk)
            while True:
                newline = buf.find(b"\n")
                if newline < 0:
                    break
                line_bytes = bytes(buf[:newline])
                del buf[: newline + 1]
                yield line_bytes.decode("utf-8", errors="replace")
        if buf:
            yield bytes(buf).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Agent run
# ---------------------------------------------------------------------------


def run_agent(
    docker_client: docker.DockerClient,
    config: ExperimentConfig,
    layout: RunLayout,
    api_key: str,
    user_prompt: str,
) -> tuple[int, float]:
    uid = os.getuid()
    gid = os.getgid()
    container_cmd: list[str] = [
        "gemini",
        "-p",
        user_prompt,
        "-m",
        config.model_slug,
        "--yolo",
        "--output-format",
        "stream-json",
    ]

    logger.info("starting gemini container (detached)")
    start = time.monotonic()
    container: Container = docker_client.containers.run(
        IMAGE_TAG,
        command=container_cmd,
        detach=True,
        network_mode="host",
        user=f"{uid}:{gid}",
        volumes={
            str(layout.scratch): {"bind": "/work", "mode": "rw"},
            str(layout.agent_home): {"bind": "/home/agent", "mode": "rw"},
            str(layout.system_prompt_host_path): {
                "bind": SYSTEM_PROMPT_IN_CONTAINER,
                "mode": "ro",
            },
        },
        working_dir="/work",
        environment={
            "HOME": "/home/agent",
            "GEMINI_API_KEY": api_key,
            "GEMINI_SYSTEM_MD": SYSTEM_PROMPT_IN_CONTAINER,
        },
        # Label keyed on the run_dir so the startup sweep can reclaim
        # *only* this run's orphans after a crash, never a sibling's.
        labels={RUN_DIR_LABEL: str(layout.run_artifacts_dir)},
    )
    logger.info("container id={cid}", cid=container.short_id)

    raw_log_path = layout.run_artifacts_dir / "agent_raw.log"
    try:
        for line in _stream_container_logs(container, raw_log_path):
            try:
                _ = json.loads(line)
                logger.info("[agent] {s}", s=_summarize_agent_event(line))
            except json.JSONDecodeError:
                logger.info("[agent] {line}", line=line[:400])
        wait_result = container.wait()
        exit_code = int(wait_result["StatusCode"])
    finally:
        try:
            container.remove(force=True)
        except docker.errors.APIError as exc:
            logger.warning("container remove failed: {e}", e=exc)

    elapsed = time.monotonic() - start
    logger.info(
        "gemini exit={code} elapsed={el:.1f}s",
        code=exit_code,
        el=elapsed,
    )
    return exit_code, elapsed


# ---------------------------------------------------------------------------
# Gemini settings.json construction
# ---------------------------------------------------------------------------


# Gemini CLI's classifier silently rewrites ``gemini-3-pro-preview`` to
# ``gemini-3.1-pro-preview`` at request time (seen in the
# ``[Routing] Selected model`` stderr trace with ``--debug``). A single
# ``modelConfigs.overrides`` entry matched on the CLI-arg id still takes
# effect, but matching on the resolved id reduces thoughts tokens further.
# Emitting both entries makes the override robust regardless of which id
# the CLI's resolver uses when applying overrides.
_GEMINI_MODEL_ALIASES: dict[str, list[str]] = {
    "gemini-3-pro-preview": ["gemini-3-pro-preview", "gemini-3.1-pro-preview"],
}


def _thinking_override_model_ids(model_slug: str) -> list[str]:
    return _GEMINI_MODEL_ALIASES.get(model_slug, [model_slug])


def _build_gemini_settings(config: ExperimentConfig, mcp_url: str) -> dict[str, Any]:
    """Build the ``settings.json`` payload for Gemini CLI's agent home.

    ``model.maxSessionTurns`` caps the session; ``mcpServers.trimul``
    publishes the benchmarking tool. When ``thinking_level`` is set we
    additionally override ``thinkingConfig.thinkingLevel`` via
    ``modelConfigs.overrides``, emitting one entry per known resolver
    target for the configured model.
    """
    settings: dict[str, Any] = {
        "model": {"maxSessionTurns": config.max_session_turns},
        "mcpServers": {
            "trimul": {
                "httpUrl": mcp_url,
                "trust": True,
                "description": "TriMul kernel benchmarking via Modal.",
            }
        },
    }
    if config.thinking_level is not None:
        override_entries: list[dict[str, Any]] = [
            {
                "match": {"model": model_id},
                "modelConfig": {
                    "generateContentConfig": {
                        "thinkingConfig": {"thinkingLevel": config.thinking_level}
                    }
                },
            }
            for model_id in _thinking_override_model_ids(config.model_slug)
        ]
        settings["modelConfigs"] = {"overrides": override_entries}
    return settings


# ---------------------------------------------------------------------------
# Kernel-file archiving
# ---------------------------------------------------------------------------


def _baseline_cache_key(config: ExperimentConfig) -> str:
    """Key a cached baseline on everything the verdict actually depends on.

    Underscores (not slashes) so the key stays a single filename rather
    than being interpreted by :class:`FileCache` as subdirectory
    namespacing.
    """
    return (
        f"{code_sha256(SEED_KERNEL_CODE)}"
        f"_{config.gpu}_{config.triton_version}_{config.aggregator}"
    )


def _get_or_prime_baseline(
    mcp_url: str,
    relative_path: str,
    config: ExperimentConfig,
    cache_dir: Path,
) -> TriMulKernelExecutionFeedback:
    """Return the cached baseline verdict, or score once and populate the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache: FileCache[BaselineFeedbackEntry] = FileCache(
        root=cache_dir, value_type=BaselineFeedbackEntry
    )
    key = _baseline_cache_key(config)
    hit = cache.get(key)
    if hit is not None:
        logger.info("baseline cache hit: {k}", k=key)
        return hit.feedback
    logger.info("baseline cache miss: {k} — priming", k=key)
    feedback = prime_baseline(mcp_url, relative_path)
    cache.put(key, BaselineFeedbackEntry(feedback=feedback))
    return feedback


def _copy_kernel_files(scratch: Path, run_artifacts_dir: Path) -> None:
    """Copy every ``kernel*.py`` from the scratch dir into artifacts.

    The versioned-file discipline (``kernel.py``, ``kernel_v1.py``, ...)
    lets a human browse candidates without parsing JSONL. The server-side
    trajectory log is still the source of truth; this is a convenience.
    """
    for src in sorted(scratch.glob("kernel*.py")):
        _ = shutil.copy2(src, run_artifacts_dir / src.name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_experiment(config: ExperimentConfig, run_dir: Path) -> TrimulRunResult:
    """Build the image, start the scoring server, run the agent, return the result.

    ``run_dir`` is the caller-supplied, already-created output dir for
    this one run. Identity is the caller's responsibility — repeated-run
    callers use indexed names (``run_00``, ``run_01``, ...) so the
    presence of ``run_dir/result.json`` is a deterministic "this one is
    done" signal for resume logic; single-run callers may use a
    timestamp so each invocation adds a new dir.

    Must be handed a freshly-created dir (any partial contents from a
    prior crashed attempt should be cleared by the caller before calling
    this — see :func:`clear_partial_runs`). We still perform an
    orphan-container / orphan-server sweep here so a caller that neglected
    to clear the filesystem won't leak Docker / Modal resources.
    """
    # docker bind-mount sources must be absolute; resolve defensively
    # rather than trusting the caller.
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set — expected from 15-arid-badger/.env via `just run`"
        )

    docker_client = docker.from_env()
    # Reclaim resources any prior crashed attempt for *this* run_dir may
    # have left behind. Both sweeps are exact-match on run_dir so no
    # sibling run or caller can be touched.
    _sweep_orphaned_containers(docker_client, run_dir)
    _kill_orphaned_server(run_dir)

    mcp_port = _pick_free_port()
    mcp_url = mcp_url_for(mcp_port)

    system_prompt_text = render_system_prompt(
        mcp_tool_name=MCP_TOOL_NAME,
        benchmark_budget=config.max_session_turns,
    )
    system_prompt_path = run_dir / "system_prompt.md"
    _ = system_prompt_path.write_text(system_prompt_text)

    build_image(docker_client)

    trajectory_log = run_dir / "trajectory.jsonl"
    # Pre-create so the first append doesn't race with a reader; empty
    # file is a valid "no calls yet" state for ``load_trajectory``.
    trajectory_log.touch()

    # Baseline cache lives one level up — shared across every run within
    # one output root (same seed, gpu, triton, aggregator).
    baseline_cache_dir = run_dir.parent / BASELINE_CACHE_DIRNAME

    with tempfile.TemporaryDirectory(prefix="gemini_trimul_") as tmp:
        host_root = Path(tmp)
        scratch = host_root / "work"
        agent_home = host_root / "home"
        gemini_config_dir = agent_home / ".gemini"
        scratch.mkdir()
        gemini_config_dir.mkdir(parents=True)

        layout = RunLayout(
            scratch=scratch,
            agent_home=agent_home,
            run_artifacts_dir=run_dir,
            system_prompt_host_path=system_prompt_path,
        )

        _ = (scratch / "kernel.py").write_text(SEED_KERNEL_CODE)
        _ = (run_dir / "seed_kernel.py").write_text(SEED_KERNEL_CODE)

        gemini_settings = _build_gemini_settings(config, mcp_url)
        _ = (gemini_config_dir / "settings.json").write_text(
            json.dumps(gemini_settings, indent=2)
        )

        server_log = run_dir / "server.log"
        server_proc, server_log_handle = start_scoring_server(
            config, mcp_port, server_log, scratch, trajectory_log
        )
        # Written atomically *after* Popen succeeds so a crash between
        # spawn and PID-file write at worst leaves an orphan that
        # eventually dies with its parent (it's a child of this proc),
        # which is fine. On successful shutdown below we remove the file.
        pid_file = run_dir / SERVER_PID_FILENAME
        _atomic_write_text(pid_file, str(server_proc.pid))
        try:
            baseline_feedback = _get_or_prime_baseline(
                mcp_url, "kernel.py", config, baseline_cache_dir
            )
            _ = (run_dir / "baseline_feedback.json").write_text(
                baseline_feedback.model_dump_json(indent=2)
            )
            user_prompt_text = render_user_prompt(
                gpu_name=config.gpu,
                triton_version=config.triton_version,
                seed_source=SEED_KERNEL_CODE,
                seed_feedback=baseline_feedback,
                benchmark_budget=config.max_session_turns,
            )
            _ = (run_dir / "user_prompt.md").write_text(user_prompt_text)
            exit_code, elapsed = run_agent(
                docker_client=docker_client,
                config=config,
                layout=layout,
                api_key=api_key,
                user_prompt=user_prompt_text,
            )
        finally:
            logger.info("terminating scoring server")
            server_proc.terminate()
            try:
                _ = server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning("server didn't exit in 15s, killing")
                server_proc.kill()
                _ = server_proc.wait(timeout=5)
            server_log_handle.close()
            # Clean shutdown — the PID file has served its purpose.
            pid_file.unlink(missing_ok=True)

        _copy_kernel_files(scratch, run_dir)
        kernel_path = scratch / "kernel.py"
        final_source = kernel_path.read_text() if kernel_path.exists() else None

    records = load_trajectory(run_dir)
    best = best_record(records)
    best_speedup: float | None = None
    best_sha: str | None = None
    if best is not None and isinstance(best.feedback, SuccessFeedback):
        best_speedup = best.feedback.aggregated_speedup
        best_sha = best.sha256

    logger.info(
        "trajectory: {n} calls, best_speedup={bs}, best_sha={sha}",
        n=len(records),
        bs=best_speedup,
        sha=best_sha,
    )

    result = TrimulRunResult(
        config=config,
        exit_code=exit_code,
        elapsed_s=elapsed,
        final_kernel_source=final_source,
        best_speedup=best_speedup,
        best_kernel_sha256=best_sha,
    )
    # Terminal marker — written atomically so a crash mid-write can never
    # leave a truncated ``result.json`` that a resume mistakes for done.
    result_path = run_dir / RESULT_FILENAME
    _atomic_write_text(result_path, result.model_dump_json(indent=2))
    logger.info("wrote {p}", p=result_path)
    return result
