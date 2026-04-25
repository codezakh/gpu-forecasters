"""Prompt templates for the theory-builder LLM.

Two prompts: one for proposing a hypothesis, one for explaining a
result. Both use the let-me-think-aloud-then-emit-tags pattern: the
model is free to reason in prose, then must put each output field in a
named tagged span. This is robust on both frontier and open models.

The four-section world-model convention (Established Beliefs / Working
Hypotheses / Open Questions / Anomalies) is taught to the model in
this prompt — the world model is just a string; the structure isn't
enforced by code.
"""

from __future__ import annotations

from arid_badger.theory_builder.v1.domain import Hypothesis


_TAG_FORMAT_RULES = """\
Inside your final answer, every load-bearing field MUST be wrapped in a
matching XML-like tag. Tags must not nest; if you need to mention a tag
literally inside a field, paraphrase rather than quote. Stray angle
brackets in code references etc. are fine — only the named tags below
are extracted by the parser.
"""


_WORLD_MODEL_STRUCTURE_GUIDE = """\
The world model is a single Markdown document. By convention it has
four sections, in this order:

1. **Established beliefs** — claims tested and confirmed, with the
   experiments that supported them.
2. **Working hypotheses** — partially supported claims; what would
   resolve them.
3. **Open questions** — known unknowns; prompts for future diagnostic
   experiments.
4. **Anomalies** — observations that don't fit current beliefs.

Each entry is a bullet that begins with a stable identifier (e.g.
`H-2025-04-25-01:` or `Q-`) and a one-sentence claim, optionally
followed by sub-bullets with detail. When closing a working hypothesis
or resolving an open question, edit the entry's `Status:` line and
move it to the appropriate section by emitting a SEARCH/REPLACE diff
that deletes it from the old section and another that inserts it into
the new section.
"""


_HYPOTHESIS_RULES = """\
A hypothesis MUST have three load-bearing parts:

* `<bottleneck>` — a specific claim about what is currently
  bottlenecking the kernel. Reference concrete code: tile sizes,
  access patterns, line numbers, specific operations. Generic claims
  like "memory access is bad" are unfalsifiable and useless.
* `<intervention>` — the change you want made + the mechanism by
  which it should help. State the mechanism even when it feels
  obvious; that's what an explanation can disagree with.
* `<prediction>` — a quantitative or directional prediction the
  benchmark will reveal if the claim is right. e.g. "geomean speedup
  rises from 1.2x to >=1.5x", "the seqlen=1024 case improves more
  than seqlen=256", "compilation succeeds where it currently fails".

Optional:

* `<code_references>` — a wrapper containing one or more
  `<reference>...</reference>` items. Each reference is a free-form
  string anchoring you to specific code (line numbers, identifiers,
  tile sizes). Empty / omitted is allowed when there's nothing
  specific to anchor to yet.

The prediction is the load-bearing part. Without it there is nothing
to be surprised by, and the resulting explanation will be vacuous.
"""


HYPOTHESIS_SYSTEM_PROMPT = f"""\
You are a research assistant building a natural-language world model
of a GPU kernel under iterative optimization. You read the current
world model + a description of the kernel, and propose ONE hypothesis
for a focused inner search to test.

{_TAG_FORMAT_RULES}

{_WORLD_MODEL_STRUCTURE_GUIDE}

{_HYPOTHESIS_RULES}

Reason aloud in prose first if it helps; only the trailing tagged
spans are extracted. Be specific. Reference the kernel code, not
generalities about GPUs.
"""


_EXPLANATION_RULES = """\
An explanation MUST have three load-bearing parts:

* `<gap>` — where the prediction and the observation diverged (or
  converged). One short paragraph. Quote concrete numbers from the
  experiment summary.
* `<mechanism>` — a proposed mechanism for the gap. If the prediction
  was confirmed, state *why* the intervention worked (and what that
  rules out). If it failed, propose what the bottleneck actually was.
* `<belief_update>` — the specific update to the world model. This is
  the prose version; you also emit one or more SEARCH/REPLACE diffs
  below to apply the update structurally.

Then emit one or more SEARCH/REPLACE blocks to update the world model.
Format:

    <<<<<<< SEARCH
    ...exact text from the current world model to find...
    =======
    ...replacement text...
    >>>>>>> REPLACE

Rules for diffs:

* The SEARCH side must match exactly ONE location in the current
  world model. Choose enough surrounding context to make it unique.
* Use an empty SEARCH side to APPEND a new section/bullet to the
  end of the document. (This is how you seed a section that doesn't
  exist yet.)
* Multiple diffs are allowed and applied in order.
* You MAY emit zero diffs if the experiment confirmed the existing
  world model with no update needed.
* Always update the source hypothesis's `Status:` line — to
  `closed` (rejected by evidence), `established` (confirmed by
  evidence), or leave it as `under_investigation` if more evidence
  is needed.

Explanations that just restate the result ("the kernel got 2.3x
because the optimization worked") add nothing. Identify where the
prior reasoning was wrong or incomplete and commit to an update that
will constrain future hypotheses.
"""


EXPLANATION_SYSTEM_PROMPT = f"""\
You are a research assistant maintaining a natural-language world
model of a GPU kernel under iterative optimization. You just ran an
inner search to test a hypothesis. Read the experiment summary and
emit (a) a structured explanation, (b) SEARCH/REPLACE diffs to update
the world model.

{_TAG_FORMAT_RULES}

{_WORLD_MODEL_STRUCTURE_GUIDE}

{_EXPLANATION_RULES}
"""


def hypothesis_user_prompt(world_model_section: str) -> str:
    return f"""\
{world_model_section}

---

Propose ONE hypothesis for the next inner-search iteration. Use the
guidance above. End your response with the required tagged spans.
"""


def explanation_user_prompt(
    world_model_section: str,
    hypothesis: Hypothesis,
    result_section: str,
) -> str:
    code_refs = (
        "\n".join(f"  - {r}" for r in hypothesis.code_references)
        if hypothesis.code_references
        else "  (none)"
    )
    return f"""\
{world_model_section}

---

## Hypothesis under investigation (id={hypothesis.id})

**Status:** {hypothesis.status}

**Bottleneck claim:**
{hypothesis.bottleneck}

**Intervention:**
{hypothesis.intervention}

**Prediction:**
{hypothesis.prediction}

**Code references:**
{code_refs}

---

## Experiment result

{result_section}

---

Read the experiment result and emit (a) the structured explanation in
tagged spans, (b) SEARCH/REPLACE diffs to update the world model
appropriately. Update the hypothesis's `Status:` to either `closed` or
`established`, or leave as `under_investigation` if the evidence is
inconclusive.
"""


__all__ = [
    "HYPOTHESIS_SYSTEM_PROMPT",
    "EXPLANATION_SYSTEM_PROMPT",
    "hypothesis_user_prompt",
    "explanation_user_prompt",
]
