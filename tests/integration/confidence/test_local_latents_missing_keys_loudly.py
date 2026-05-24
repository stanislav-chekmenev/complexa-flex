"""Sidecar must fail loudly on a missing `local_latents` intermediate.

Contract: `ConfidenceDistillationModule._forward` reads
`inter["local_latents"]` directly (subscript, not `.get`). When the
frozen trunk forgets to emit `local_latents` -- a real risk during
trunk-side churn -- the sidecar must raise `KeyError` rather than
silently substitute zeros (which would let a buggy trunk silently
train heads without the fusion residual).

The check is structural: scan the sidecar source for the read pattern
to ensure no future refactor introduces `inter.get("local_latents", ...)`
without an accompanying explicit error.
"""

from __future__ import annotations

import inspect
import re

from proteinfoundation.confidence import lightning_module


def test_lightning_module_reads_local_latents_with_subscript_not_get() -> None:
    src = inspect.getsource(lightning_module)
    assert 'inter["local_latents"]' in src or "inter['local_latents']" in src, (
        "ConfidenceDistillationModule._forward must read "
        "`inter[\"local_latents\"]` (subscript). A missing key should "
        "raise KeyError so a buggy frozen trunk fails loudly."
    )

    get_patterns = [
        re.compile(r'inter\.get\(\s*[\'"]local_latents[\'"]'),
        re.compile(r'inter\[\s*[\'"]local_latents[\'"]\s*\]\s*if\s+'),
    ]
    for pat in get_patterns:
        assert not pat.search(src), (
            f"local_latents must be read with a hard subscript so a missing "
            f"key raises KeyError, not silently substituted. Found pattern "
            f"{pat.pattern!r} in lightning_module.py."
        )
