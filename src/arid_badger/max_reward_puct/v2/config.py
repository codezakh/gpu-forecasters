"""Configuration objects for v2 max-reward PUCT.

A caller constructs one ``SearchConfig``, hands it to the driver, and
doesn't worry about individual knobs at the call site. Group knobs that
belong together into sub-objects once there's enough of them to warrant it.
"""

from pydantic import BaseModel, ConfigDict


class SearchConfig(BaseModel):
    """All parameters that shape the search itself. Providers and the event
    log are injected separately — they are infrastructure, not config."""

    model_config = ConfigDict(frozen=True)

    total_budget_steps: int
    batch_size: int
    samples_per_parent: int
    k_per_parent: int
    archive_capacity: int = 1000
    c_puct: float = 1.0
