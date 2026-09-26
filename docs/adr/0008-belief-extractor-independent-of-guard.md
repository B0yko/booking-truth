# 8. The belief extractor is independent of the agent's guard

Status: accepted

## Context

The guarded agent blocks replies that claim a booking it cannot verify. If the harness measured "what the prospect
was told" with the same detector, a phrasing the detector misses would be missed twice: the guard would let it
through and the grader would not see it. The measurement would be circular.

## Decision

The harness extracts the prospect's belief with its own components:

- an LLM extractor with structured output that reads the agent's messages, with the persona's true zone as context;
- a deterministic lexicon extractor in `harness/` with its own pattern list. A test enforces that no harness module
  imports `agent/guards`.

Both run on every trial when a key is configured; under offline grading only the lexicon extractor runs. The report
gives their disagreement rate separately for naive and guarded trials and lists every naive `false_success` where
they disagree. Guarded confirmations are code-rendered and easier to parse, so agreement is expected to be higher on
guarded trials; the split keeps that from flattering the result.

## Consequences

- The extractor is itself an LLM with a measured accuracy on a held-out labelled set (`booking-truth eval
  extractor`), and the limitations section states it.
