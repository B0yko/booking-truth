"""``booking-truth eval``: component evaluations of the timezone resolver and the belief extractors, on
the held-out test split of their respective datasets. See ``datasets/README.md`` for the scoring rules and
the split policy, and ``docs/adr/0008-belief-extractor-independent-of-guard.md`` for why the extractor eval
is not circular. Results land under ``results/<run-id>/{tz-eval,extractor-eval}.{json,md}``.
"""

from __future__ import annotations
