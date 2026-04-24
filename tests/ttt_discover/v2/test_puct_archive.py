from pathlib import Path

from arid_badger.trimul.core import CaseSpeedup, IncorrectFeedback, SuccessFeedback
from arid_badger.ttt_discover.v2.archive.puct import (
    PUCTCandidateArchive,
    build_candidate,
)


def _success(runtime_ns: float) -> SuccessFeedback:
    return SuccessFeedback(
        aggregated_speedup=1.0,
        aggregation_method="geomean",
        per_case_speedups=[
            CaseSpeedup(
                seqlen=256,
                bs=2,
                dim=128,
                hiddendim=128,
                nomask=True,
                distribution="normal",
                speedup=1.0,
                runtime_ns=runtime_ns,
                ref_runtime_ns=runtime_ns,
            )
        ],
    )


def test_seed_root_is_sampleable(tmp_path: Path) -> None:
    archive = PUCTCandidateArchive(directory=tmp_path)
    picked = archive.sample(n=1)
    assert len(picked) == 1
    assert picked[0].code == ""
    assert picked[0].outcome is None


def test_insert_child_then_sample_prefers_higher_reward(tmp_path: Path) -> None:
    archive = PUCTCandidateArchive(directory=tmp_path)
    root = archive.sample(n=1)[0]

    low = build_candidate(
        code="low",
        timestep=0,
        parent_id=root.id,
        outcome=IncorrectFeedback(error_message="nope"),
        reward=0.0,
    )
    high = build_candidate(
        code="high",
        timestep=0,
        parent_id=root.id,
        outcome=_success(runtime_ns=1_000_000.0),
        reward=2.5,
    )
    archive.credit_rollout(parent=root, child=low)
    archive.credit_rollout(parent=root, child=high)

    # Sample enough to see both — but lineage blocking ensures we only get
    # the root OR one non-overlapping descendant. n=1 should prefer the
    # highest-score candidate. The "high" child and the root share a
    # lineage (root → high), so sampling picks exactly one.
    picked = archive.sample(n=1)
    assert len(picked) == 1
    # With puct_c=1 and identical n across candidates, the highest reward
    # wins. That is the ``high`` child; the root's reward is 0.
    assert picked[0].code == "high"


def test_snapshot_round_trip(tmp_path: Path) -> None:
    archive = PUCTCandidateArchive(directory=tmp_path)
    root = archive.sample(n=1)[0]
    child = build_candidate(
        code="c1",
        timestep=0,
        parent_id=root.id,
        outcome=_success(runtime_ns=2_500_000.0),
        reward=1.0,
    )
    archive.credit_rollout(parent=root, child=child)

    archive.snapshot(step=7)
    snap_path = tmp_path / "puct_archive_step_000007.json"
    assert snap_path.exists()

    reloaded = PUCTCandidateArchive.from_snapshot(
        directory=tmp_path, snapshot_path=snap_path
    )
    picked = reloaded.sample(n=2)
    codes = {c.code for c in picked}
    # Root + child may both appear if lineage blocking doesn't preclude.
    # At minimum, the inserted child is recoverable.
    assert "c1" in codes or picked[0].code == "c1"


def test_credit_rollout_without_child_increments_visits(tmp_path: Path) -> None:
    archive = PUCTCandidateArchive(directory=tmp_path)
    root = archive.sample(n=1)[0]
    archive.credit_rollout(parent=root, child=None)
    archive.credit_rollout(parent=root, child=None)
    archive.snapshot(step=1)

    snap_path = tmp_path / "puct_archive_step_000001.json"
    text = snap_path.read_text()
    # Root id should show up in visits with count 2.
    assert root.id in text
    assert '"total_visits": 2' in text


def test_credit_rollout_with_and_without_child_bump_visits_identically(
    tmp_path: Path,
) -> None:
    """Load-bearing invariant: ``credit_rollout(child=candidate_with_reward_0)``
    and ``credit_rollout(child=None)`` produce the same visit-count /
    total-visits deltas on the parent's subtree. This is what lets the
    admission policy be a pure ``AdmitChild | CreditOnly`` choice
    without concern about double-counting or under-counting rollouts."""
    archive_a = PUCTCandidateArchive(directory=tmp_path / "a")
    archive_b = PUCTCandidateArchive(directory=tmp_path / "b")
    root_a = archive_a.sample(n=1)[0]
    root_b = archive_b.sample(n=1)[0]

    failed = build_candidate(
        code="bad",
        timestep=0,
        parent_id=root_a.id,
        outcome=IncorrectFeedback(error_message="wrong"),
        reward=0.0,
    )
    archive_a.credit_rollout(parent=root_a, child=failed)
    archive_b.credit_rollout(parent=root_b, child=None)

    archive_a.snapshot(step=0)
    archive_b.snapshot(step=0)
    snap_a = (tmp_path / "a" / "puct_archive_step_000000.json").read_text()
    snap_b = (tmp_path / "b" / "puct_archive_step_000000.json").read_text()

    # Visit bookkeeping on the root is identical across the two paths.
    assert f'"{root_a.id}": 1' in snap_a
    assert f'"{root_b.id}": 1' in snap_b
    assert '"total_visits": 1' in snap_a
    assert '"total_visits": 1' in snap_b
