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
    archive.insert(low, root)
    archive.insert(high, root)

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
    archive.insert(child, root)

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


def test_record_failed_attempt_increments_visits(tmp_path: Path) -> None:
    archive = PUCTCandidateArchive(directory=tmp_path)
    root = archive.sample(n=1)[0]
    archive.record_failed_attempt(root)
    archive.record_failed_attempt(root)
    archive.snapshot(step=1)

    snap_path = tmp_path / "puct_archive_step_000001.json"
    text = snap_path.read_text()
    # Root id should show up in visits with count 2.
    assert root.id in text
    assert '"total_visits": 2' in text
