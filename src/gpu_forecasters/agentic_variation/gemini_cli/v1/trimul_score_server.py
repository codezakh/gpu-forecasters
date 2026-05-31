"""FastMCP scoring server for the Gemini CLI TriMul harness.

Host-side process that exposes a single MCP tool, ``score_trimul``, which
benchmarks a candidate TriMul kernel on Modal using the existing
:class:`TriMulModalProvider`. The server runs on the host (where modal and
gpu_forecasters are already installed); the Gemini CLI runs inside a minimal
container and reaches the server via streamable HTTP over ``--network=host``,
so the container stays free of Python/modal/gpu_forecasters dependencies.

Design note — path-based tool: the agent passes a *path* relative to its
working directory rather than the full kernel source. The server reads the
file directly off the host-side bind mount of the container's scratch dir.
This avoids making the agent re-emit the entire kernel as tool-call tokens
on every iteration, which is the dominant wall-clock cost per turn for
large kernels.

Lifecycle: one Modal scoring session per server lifetime. The orchestrator
launches this as a subprocess and terminates it when the agent run finishes.

This script is run by the orchestrator; not intended for standalone use.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import FastMCP
from loguru import logger

from gpu_forecasters.experiment_utils import install_loguru_intercept
from gpu_forecasters.hill_climbing.scoring_providers.trimul_modal import (
    AggregationMethod,
    TriMulModalProvider,
)
from gpu_forecasters.invocation_sink import code_sha256
from gpu_forecasters.trimul.cases import BENCHMARK_CASES

from .prompts import format_feedback_summary


DEFAULT_PATH = "/mcp/"


def build_server(
    provider: TriMulModalProvider,
    scratch_root: Path,
    trajectory_log: Path | None,
) -> FastMCP:
    mcp: FastMCP = FastMCP("trimul-score")
    resolved_root = scratch_root.resolve()

    @mcp.tool
    def score_trimul(path: str) -> dict[str, object]:
        """Benchmark a candidate TriMul kernel file on Modal.

        Arguments:
          path: Path to the kernel source file, relative to your working
            directory (e.g. ``"kernel.py"``). The file is read by the
            scoring service directly; you do NOT need to pass the kernel
            contents as an argument. Write/edit the file normally, then
            call this tool with the path.

        The tool returns the raw feedback fields (``kind`` discriminator
        plus kind-specific data — ``aggregated_speedup`` and
        ``per_case_speedups`` on success, error / traceback fields on
        failure) alongside a ``summary`` string: a natural-language
        rendering of the same information, matching the per-iteration
        feedback format used by the non-agentic TriMul mutation
        operator. Read ``summary`` for the actionable signal; ``kind``
        and its siblings are there if you want to branch programmatically.
        """
        candidate_path = (resolved_root / path).resolve()
        # Keep the agent inside its scratch dir — scoring arbitrary files
        # on the host would be both surprising and a footgun.
        try:
            _ = candidate_path.relative_to(resolved_root)
        except ValueError:
            raise ValueError(
                f"path {path!r} escapes the scratch directory; must be relative to cwd"
            )
        if not candidate_path.is_file():
            raise FileNotFoundError(f"no such file: {path!r}")

        source = candidate_path.read_text()
        evaluation = provider.evaluate(source)
        feedback = evaluation.observation.feedback
        logger.info(
            "score_trimul(path={path}): kind={kind}, reward={reward}",
            path=path,
            kind=feedback.kind,
            reward=evaluation.reward,
        )
        # Infrastructure failures bubble up as exceptions (the MCP client
        # sees a tool error, which is the right signal — we don't want the
        # agent trying to "fix" our scoring pipeline). Only the four
        # kernel-level verdicts reach the summary formatter.
        if feedback.kind == "infrastructure_failure":
            raise RuntimeError(
                f"scoring infrastructure failure: {feedback.reason}"
            )
        if trajectory_log is not None:
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "path": path,
                "sha256": code_sha256(source),
                "kernel_source": source,
                "feedback": feedback.model_dump(),
            }
            # Append one JSON record per call. Open/close each time so a
            # crash mid-run leaves a well-formed prefix rather than losing
            # buffered writes.
            with trajectory_log.open("a") as f:
                _ = f.write(json.dumps(record) + "\n")
        response = feedback.model_dump()
        response["summary"] = format_feedback_summary(feedback)
        return response

    return mcp


def main() -> None:
    install_loguru_intercept()
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--port", type=int, required=True)
    _ = parser.add_argument("--gpu", required=True)
    _ = parser.add_argument("--scratch-root", type=Path, required=True)
    _ = parser.add_argument("--aggregator", default="geomean")
    _ = parser.add_argument("--trajectory-log", type=Path, default=None)
    args = parser.parse_args()

    aggregator: AggregationMethod = args.aggregator
    scratch_root: Path = args.scratch_root
    trajectory_log: Path | None = args.trajectory_log

    if not scratch_root.is_dir():
        raise RuntimeError(f"--scratch-root {scratch_root} is not a directory")

    logger.info(
        "opening Modal scoring session (gpu={gpu}, aggregator={agg}, scratch={root})",
        gpu=args.gpu,
        agg=aggregator,
        root=scratch_root,
    )
    with TriMulModalProvider(
        test_cases=BENCHMARK_CASES,
        gpu=args.gpu,
        aggregator=aggregator,
    ) as provider:
        mcp = build_server(provider, scratch_root, trajectory_log)
        logger.info(
            "FastMCP listening on http://127.0.0.1:{port}{path}",
            port=args.port,
            path=DEFAULT_PATH,
        )
        mcp.run(transport="http", host="127.0.0.1", port=args.port, path=DEFAULT_PATH)


if __name__ == "__main__":
    main()
