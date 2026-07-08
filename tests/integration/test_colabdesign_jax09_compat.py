"""Compat guard: vendored colabdesign must use jax 0.9 APIs, not removed ones.

The venv pins ``jax[cuda13]==0.9.2`` (env/build_uv_env.sh). jax 0.9 removed a
set of top-level aliases that the vendored ``community_models/colabdesign``
tree still calls, which crashed the AF2 binder-evaluate stage (job 85858) at
``clear_mem`` with ``module 'jax.lib' has no attribute 'xla_bridge'``.

The failures reproduce on CPU (``JAX_PLATFORMS=cpu``), no GPU needed. These
tests pin the three removed-API families so the compat does not silently
regress on a future colabdesign re-vendor:

  - ``jax.lib.xla_bridge``      -> ``jax.extend.backend`` (clear_mem)
  - ``jax.tree_map``            -> ``jax.tree_util.tree_map``
  - ``jax.tree_flatten`` /
    ``jax.tree_unflatten``      -> ``jax.tree_util.tree_{flatten,unflatten}``
  - ``jax.util.wraps``          -> ``jax._src.util.wraps`` (mapping.py; keeps
                                   the ``docstr=`` kwarg functools.wraps lacks)
  - ``jax.tree_leaves``         -> ``jax.tree_util.tree_leaves`` (tr/model.py)

The replacements are pure namespace relocations with identical semantics; no
colabdesign numerics change (community-parity rule preserved).

``test_all_jax_tokens_resolve_under_jax09`` is the self-maintaining guard: it
greps every distinct ``jax.<attr>`` token used across the vendored tree and
fails cheaply on CPU (no GPU forward pass) the moment a future re-vendor
reintroduces a removed-in-0.9 attribute.
"""

import importlib
import os
import pathlib
import re

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
COLABDESIGN_ROOT = REPO_ROOT / "community_models" / "colabdesign"

# jax attributes removed in 0.9 that must not appear as ``jax.<name>`` /
# ``jax.lib.<name>`` in the vendored source.
REMOVED_PATTERNS = {
    "jax.lib.xla_bridge": re.compile(r"\bjax\.lib\.xla_bridge\b"),
    "jax.tree_map": re.compile(r"\bjax\.tree_map\b"),
    "jax.tree_flatten": re.compile(r"\bjax\.tree_flatten\b"),
    "jax.tree_unflatten": re.compile(r"\bjax\.tree_unflatten\b"),
    "jax.tree_leaves": re.compile(r"\bjax\.tree_leaves\b"),
    "jax.util": re.compile(r"\bjax\.util\b"),
}

# Matches a top-level ``jax.<attr>`` reference (first component after ``jax.``).
# ``jax._src`` is captured as ``_src`` so nested private refs like
# ``jax._src.util.wraps`` are grouped under the importable ``jax._src`` root.
_JAX_TOKEN = re.compile(r"\bjax\.([A-Za-z_][A-Za-z0-9_]*)")


def _iter_py_files():
    return sorted(COLABDESIGN_ROOT.rglob("*.py"))


def _distinct_jax_tokens():
    """Every distinct first-level ``jax.<attr>`` component in the vendored tree."""
    tokens = set()
    for path in _iter_py_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in _JAX_TOKEN.finditer(text):
            tokens.add(m.group(1))
    return sorted(tokens)


def test_no_removed_jax_apis_in_colabdesign_source():
    offenders = {}
    for path in _iter_py_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pat in REMOVED_PATTERNS.items():
            hits = [i + 1 for i, line in enumerate(text.splitlines()) if pat.search(line)]
            if hits:
                offenders.setdefault(str(path.relative_to(REPO_ROOT)), {})[name] = hits
    assert not offenders, (
        "Removed jax-0.9 APIs still present in vendored colabdesign:\n"
        + "\n".join(f"  {f}: {v}" for f, v in offenders.items())
    )


def test_clear_mem_runs_under_jax09():
    """The exact job-85858 crash site: clear_mem must execute on jax 0.9."""
    from community_models.colabdesign.shared.utils import clear_mem

    clear_mem()  # raised AttributeError(jax.lib has no xla_bridge) before the fix


def test_tree_flatten_call_sites_work():
    """Exercise a real jax.tree_flatten/unflatten round-trip like mapping.py does."""
    import jax
    import jax.numpy as jnp

    leaves, treedef = jax.tree_util.tree_flatten({"a": jnp.zeros(2), "b": jnp.ones(3)})
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert set(rebuilt) == {"a", "b"}


def test_all_jax_tokens_resolve_under_jax09():
    """Self-maintaining guard: every ``jax.<attr>`` in the tree must resolve.

    A token passes if it is either a live attribute on the imported ``jax``
    module OR an importable submodule (e.g. ``jax.extend``, ``jax._src``).
    Submodules are NOT plain attributes of a bare ``import jax``, so they are
    checked via ``importlib.import_module`` to avoid false MISSING reports.
    Any future removed-in-jax colabdesign token fails here on CPU without a
    GPU forward pass.
    """
    import jax

    missing = []
    for token in _distinct_jax_tokens():
        if hasattr(jax, token):
            continue
        try:
            importlib.import_module(f"jax.{token}")
        except Exception:  # noqa: BLE001 - any import failure means unresolvable
            missing.append(token)

    assert not missing, (
        "colabdesign references jax attrs unresolvable under this jax "
        f"({jax.__version__}): " + ", ".join(f"jax.{t}" for t in missing)
    )


def test_wraps_and_tree_leaves_call_sites_work():
    """Exercise the two relocated APIs exactly as the fixed sites use them.

    - ``jax._src.util.wraps(fun, docstr=...)`` (mapping.py:123): the decorator
      must preserve ``__name__``/``__doc__``. ``functools.wraps`` would reject
      the ``docstr=`` kwarg, which is why the port keeps ``jax._src.util``.
    - ``jax.tree_util.tree_leaves`` (tr/model.py:85,312): summed over a loss
      pytree to fold nested weighted losses into a scalar.
    """
    import jax
    import jax.numpy as jnp
    from jax._src import util as _jax_util

    def fun(x):
        """orig docstring."""
        return x

    @_jax_util.wraps(fun, docstr="wrapped docstring.")
    def wrapped(x):
        return x

    assert wrapped.__name__ == "fun"
    assert wrapped.__doc__ == "wrapped docstring."

    losses = {"a": jnp.asarray(1.0), "b": {"c": jnp.asarray(2.0), "d": jnp.asarray(3.0)}}
    assert float(sum(jax.tree_util.tree_leaves(losses))) == 6.0
