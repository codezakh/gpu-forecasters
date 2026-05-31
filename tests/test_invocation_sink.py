"""Tests for gpu_forecasters.invocation_sink."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

from pydantic import BaseModel
from ulid import ULID

from gpu_forecasters.greedy_search.domain import MutationContext
from gpu_forecasters.greedy_search.feedback_mutation import (
    FeedbackMutationRecord,
    KernelBenchExecutionFeedbackMutationFunction,
)
from gpu_forecasters.greedy_search.kernelbench_prompt_mutation import (
    KernelBenchPromptMutationFunction,
    PromptMutationRecord,
)
from gpu_forecasters.invocation_sink import (
    FilesystemInvocationSink,
    InvocationSink,
    code_sha256,
)
from gpu_forecasters.landscape_map.v1.domain import (
    KernelTaskInfo,
    LikertConfidence,
    LlmCallUsage,
    KernelRuntimeEstimate,
    SpeedupBin,
)
from gpu_forecasters.landscape_map.v1.training_free_evaluation_provider import (
    KernelWorldModelEvaluationProvider,
    KwmEvaluationRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleRecord(BaseModel, frozen=True):
    kind: Literal["simple"] = "simple"
    value: int


def _make_litellm_response(
    content: str = "```python\ndef kernel(): pass\n```",
    input_tokens: int = 80,
    output_tokens: int = 40,
) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    response.usage.prompt_tokens = input_tokens
    response.usage.completion_tokens = output_tokens
    return response


def _make_kwm_estimate() -> KernelRuntimeEstimate:
    return KernelRuntimeEstimate(
        predicted_bin=SpeedupBin.MINOR_SPEEDUP,
        bin_confidences={b: LikertConfidence.MODERATE for b in SpeedupBin if b != SpeedupBin.FAILURE},
        reasoning="looks fast",
    )


def _make_task_info() -> KernelTaskInfo:
    return KernelTaskInfo(op_name="test_op", level_id=1, task_id=1)


def _make_mutation_context() -> MutationContext:
    return MutationContext(
        reference_kernel_code="def ref(): pass",
        previous_kernel_code="def prev(): pass",
        num_mutations=1,
    )


# ---------------------------------------------------------------------------
# FilesystemInvocationSink
# ---------------------------------------------------------------------------


class TestFilesystemInvocationSink:
    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        subdir = tmp_path / "deep" / "nested"
        assert not subdir.exists()
        FilesystemInvocationSink(subdir)
        assert subdir.is_dir()

    def test_record_writes_single_json_file(self, tmp_path: Path) -> None:
        sink = FilesystemInvocationSink(tmp_path)
        sink.record(_SimpleRecord(value=42))
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".json"

    def test_record_json_deserializes_correctly(self, tmp_path: Path) -> None:
        sink = FilesystemInvocationSink(tmp_path)
        sink.record(_SimpleRecord(value=99))
        file = next(tmp_path.iterdir())
        data = json.loads(file.read_text())
        assert data["kind"] == "simple"
        assert data["value"] == 99

    def test_filename_is_valid_ulid(self, tmp_path: Path) -> None:
        sink = FilesystemInvocationSink(tmp_path)
        sink.record(_SimpleRecord(value=1))
        file = next(tmp_path.iterdir())
        name = file.stem
        parsed = ULID.from_str(name)
        assert str(parsed) == name

    def test_multiple_records_produce_distinct_files(self, tmp_path: Path) -> None:
        sink = FilesystemInvocationSink(tmp_path)
        for i in range(5):
            sink.record(_SimpleRecord(value=i))
        files = list(tmp_path.iterdir())
        assert len(files) == 5
        assert len({f.stem for f in files}) == 5

    def test_satisfies_invocation_sink_protocol(self, tmp_path: Path) -> None:
        sink = FilesystemInvocationSink(tmp_path)
        assert isinstance(sink, InvocationSink)

    def test_record_does_not_raise_on_write_failure(self, tmp_path: Path) -> None:
        sink = FilesystemInvocationSink(tmp_path)
        with patch("gpu_forecasters.invocation_sink.Path.write_text", side_effect=OSError("disk full")):
            sink.record(_SimpleRecord(value=1))  # must not raise

    def test_record_writes_to_nested_nonexistent_path(self, tmp_path: Path) -> None:
        subdir = tmp_path / "deep" / "nested"
        assert not subdir.exists()
        sink = FilesystemInvocationSink(subdir)
        sink.record(_SimpleRecord(value=7))
        files = list(subdir.iterdir())
        assert len(files) == 1


# ---------------------------------------------------------------------------
# code_sha256
# ---------------------------------------------------------------------------


class TestCodeSha256:
    def test_deterministic(self) -> None:
        assert code_sha256("hello") == code_sha256("hello")

    def test_distinct_inputs_produce_distinct_hashes(self) -> None:
        assert code_sha256("foo") != code_sha256("bar")

    def test_returns_64_char_hex_string(self) -> None:
        digest = code_sha256("some code")
        assert len(digest) == 64
        int(digest, 16)  # raises if not valid hex


# ---------------------------------------------------------------------------
# KernelBenchPromptMutationFunction sink
# ---------------------------------------------------------------------------


class TestPromptMutationFunctionSink:
    def test_records_when_sink_present(self) -> None:
        mock_sink = MagicMock(spec=InvocationSink)
        fn = KernelBenchPromptMutationFunction(
            model_slug="test-model",
            invocation_sink=mock_sink,
        )
        context = _make_mutation_context()
        response = _make_litellm_response(input_tokens=80, output_tokens=40)

        with patch(
            "gpu_forecasters.greedy_search.kernelbench_prompt_mutation.completion",
            return_value=response,
        ):
            fn(context)

        mock_sink.record.assert_called_once()
        record = mock_sink.record.call_args[0][0]
        assert isinstance(record, PromptMutationRecord)
        assert record.model_slug == "test-model"
        assert record.input_tokens == 80
        assert record.output_tokens == 40
        assert record.parent_code_sha256 == code_sha256("def prev(): pass")
        assert record.child_code_sha256 == code_sha256("def kernel(): pass")

    def test_does_not_record_when_sink_is_none(self) -> None:
        fn = KernelBenchPromptMutationFunction(model_slug="test-model")
        context = _make_mutation_context()
        response = _make_litellm_response()

        with patch(
            "gpu_forecasters.greedy_search.kernelbench_prompt_mutation.completion",
            return_value=response,
        ):
            fn(context)  # should not raise


# ---------------------------------------------------------------------------
# KernelWorldModelEvaluationProvider sink
# ---------------------------------------------------------------------------


class TestKernelWorldModelEvaluationProviderSink:
    def test_records_when_sink_and_usage_present(self) -> None:
        estimate = _make_kwm_estimate()
        usage = LlmCallUsage(input_tokens=100, output_tokens=50)
        mock_estimator = MagicMock()
        mock_estimator.estimate.return_value = (estimate, usage)

        mock_sink = MagicMock(spec=InvocationSink)
        provider = KernelWorldModelEvaluationProvider(
            reference_kernel_code="def ref(): pass",
            task_info=_make_task_info(),
            estimator=mock_estimator,
            model_slug="test-model",
            invocation_sink=mock_sink,
        )

        program_code = "def kernel(): pass"
        provider.evaluate(program_code)

        mock_sink.record.assert_called_once()
        record = mock_sink.record.call_args[0][0]
        assert isinstance(record, KwmEvaluationRecord)
        assert record.model_slug == "test-model"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.code_sha256 == code_sha256(program_code)
        assert record.predicted_bin == int(SpeedupBin.MINOR_SPEEDUP)

    def test_does_not_record_when_sink_is_none(self) -> None:
        estimate = _make_kwm_estimate()
        usage = LlmCallUsage(input_tokens=100, output_tokens=50)
        mock_estimator = MagicMock()
        mock_estimator.estimate.return_value = (estimate, usage)

        provider = KernelWorldModelEvaluationProvider(
            reference_kernel_code="def ref(): pass",
            task_info=_make_task_info(),
            estimator=mock_estimator,
            invocation_sink=None,
        )
        provider.evaluate("def kernel(): pass")  # should not raise

    def test_does_not_record_when_usage_is_none(self) -> None:
        estimate = _make_kwm_estimate()
        mock_estimator = MagicMock()
        mock_estimator.estimate.return_value = (estimate, None)

        mock_sink = MagicMock(spec=InvocationSink)
        provider = KernelWorldModelEvaluationProvider(
            reference_kernel_code="def ref(): pass",
            task_info=_make_task_info(),
            estimator=mock_estimator,
            model_slug="test-model",
            invocation_sink=mock_sink,
        )
        provider.evaluate("def kernel(): pass")
        mock_sink.record.assert_not_called()


# ---------------------------------------------------------------------------
# FilesystemInvocationSink round-trip with a real provider record type
# ---------------------------------------------------------------------------


class TestFilesystemSinkRoundTrip:
    def test_prompt_mutation_record_round_trip(self, tmp_path: Path) -> None:
        sink = FilesystemInvocationSink(tmp_path)
        record = PromptMutationRecord(
            parent_code_sha256="aabbcc",
            child_code_sha256="ddeeff",
            model_slug="gemini/test",
            input_tokens=80,
            output_tokens=40,
            wall_clock_seconds=1.23,
            timestamp_utc="2026-01-01T00:00:00+00:00",
        )
        sink.record(record)

        file = next(tmp_path.iterdir())
        data = json.loads(file.read_text())
        assert data["kind"] == "prompt_mutation"
        assert data["model_slug"] == "gemini/test"
        assert data["input_tokens"] == 80
