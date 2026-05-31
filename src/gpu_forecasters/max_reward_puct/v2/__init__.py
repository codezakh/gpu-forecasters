"""Event-sourced v2 of Max-Reward PUCT search.

The state of a search is derived from an append-only log of typed events.
Durability, resumability, and idempotency fall out of that single design
choice rather than being bolted on as separate concerns. See
``docs/specs/gh054-v2-plan-v2.md`` for the philosophy.
"""
