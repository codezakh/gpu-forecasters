from unittest.mock import MagicMock

from arid_badger.greedy_search.domain import (
    MutatedKernel,
    MutationContext,
    MutationFunction,
)
from arid_badger.greedy_search.mutation_provider import SerialMutationProvider
from arid_badger.greedy_search.trace import MutationFailure, MutationSuccess


def _make_context(*, num_mutations: int = 2) -> MutationContext:
    return MutationContext(
        reference_kernel_code="def ref(): pass",
        previous_kernel_code="def prev(): pass",
        num_mutations=num_mutations,
    )


class TestGenerateMutations:
    def test_all_succeed(self):
        mutation1 = MutatedKernel(kernel_code="code_a", ancestor_ulid=None)
        mutation2 = MutatedKernel(kernel_code="code_b", ancestor_ulid=None)

        mock_fn = MagicMock(spec=MutationFunction)
        mock_fn.side_effect = [mutation1, mutation2]

        provider = SerialMutationProvider(mutation_function=mock_fn)
        context = _make_context(num_mutations=2)

        attempts, generated = provider.generate_mutations(context)

        assert len(attempts) == 2
        assert all(isinstance(a, MutationSuccess) for a in attempts)
        assert len(generated) == 2
        assert generated[0].code == "code_a"
        assert generated[1].code == "code_b"
        assert attempts[0].attempt_idx == 0
        assert attempts[1].attempt_idx == 1

    def test_some_raise_produces_mutation_failure(self):
        mutation1 = MutatedKernel(kernel_code="code_a", ancestor_ulid=None)

        mock_fn = MagicMock(spec=MutationFunction)
        mock_fn.side_effect = [
            mutation1,
            RuntimeError("LLM error"),
            MutatedKernel(kernel_code="code_c", ancestor_ulid=None),
        ]

        provider = SerialMutationProvider(mutation_function=mock_fn)
        context = _make_context(num_mutations=3)

        attempts, generated = provider.generate_mutations(context)

        assert len(attempts) == 3
        assert isinstance(attempts[0], MutationSuccess)
        assert isinstance(attempts[1], MutationFailure)
        assert "LLM error" in attempts[1].error.exception_repr
        assert isinstance(attempts[2], MutationSuccess)
        assert len(generated) == 2

    def test_all_raise(self):
        mock_fn = MagicMock(spec=MutationFunction)
        mock_fn.side_effect = [
            ValueError("error1"),
            ValueError("error2"),
        ]

        provider = SerialMutationProvider(mutation_function=mock_fn)
        context = _make_context(num_mutations=2)

        attempts, generated = provider.generate_mutations(context)

        assert len(attempts) == 2
        assert all(isinstance(a, MutationFailure) for a in attempts)
        assert len(generated) == 0
