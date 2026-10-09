"""Guard the documented agent model routing in ``docs/claude/agents.md``."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"
SETTINGS = REPO_ROOT / ".claude" / "settings.json"
LOCAL_SETTINGS = REPO_ROOT / ".claude" / "settings.local.json"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
ROUTING_DOC = REPO_ROOT / "docs" / "claude" / "agents.md"

MAIN_MODEL = "gpt-6-sol"
MAIN_AGENT = "orchestrator"
JUDGEMENT_MODEL = "z-ai/glm-5.3-flash"
IMPLEMENTATION_MODEL = "gpt-6-luna"
SESSION_EFFORT = "max"
JUDGEMENT_EFFORT = "max"
COLLABORATOR_SUBAGENT_MODEL = "claude-opus-5"
MAIN_LANE_LABEL = "Main session"

LANES = {
    "inherit": {MAIN_AGENT},
    JUDGEMENT_MODEL: {
        "code-review-debug-complexity-expert",
        "software-planning-architect",
        "generative-protein-scientist",
        "generative-flow-stochastic-math-expert",
        "structural-biology-binder-expert",
        "structural-biology-idp-smallmol-expert",
        "physics-statmech-md-dft-expert",
        "xray-crystallography-binder-ml",
        "Explore",
        "Plan",
        "general-purpose",
    },
    IMPLEMENTATION_MODEL: {
        "ml-protein-architect",
        "ml-software-pytorch-jax-expert",
    },
}
NON_ROSTER = {MAIN_AGENT, "Explore", "Plan", "general-purpose"}
ALIAS_KEYS = {
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
}
GATEWAY_MODELS = {
    MAIN_MODEL,
    JUDGEMENT_MODEL,
    IMPLEMENTATION_MODEL,
    "claude-haiku-4-5",
}


def _frontmatter(path: Path) -> dict[str, str]:
    """Parse fields from the leading frontmatter block only."""
    text = path.read_text()
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match is not None, f"{path} has no frontmatter block"
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if not separator or key != key.strip() or not re.fullmatch(r"[\w-]+", key):
            continue
        fields[key] = re.sub(r"\s+#.*$", "", value).strip().strip("\"'")
    return fields


def _agent_models() -> dict[str, str | None]:
    return {
        path.stem: _frontmatter(path).get("model")
        for path in sorted(AGENTS_DIR.rglob("*.md"))
    }


def _settings() -> dict:
    return json.loads(SETTINGS.read_text())


def test_every_agent_pins_a_model():
    unpinned = sorted(name for name, model in _agent_models().items() if not model)
    assert unpinned == [], f"agents without a model pin: {unpinned}"


def test_every_agent_name_matches_its_filename():
    mismatched = {
        path.stem: _frontmatter(path).get("name")
        for path in sorted(AGENTS_DIR.rglob("*.md"))
        if _frontmatter(path).get("name") != path.stem
    }
    assert mismatched == {}, f"frontmatter name disagrees with filename: {mismatched}"


def test_the_roster_on_disk_matches_the_lane_table():
    expected = {name for names in LANES.values() for name in names}
    assert set(_agent_models()) == expected


@pytest.mark.parametrize(
    ("agent", "model"),
    sorted((name, lane) for lane, names in LANES.items() for name in names),
)
def test_agent_is_on_its_documented_lane(agent: str, model: str):
    assert _agent_models().get(agent) == model


def test_no_alias_resolves_to_the_implementation_lane():
    settings = _settings()
    assert ALIAS_KEYS <= set(settings["env"])
    alias_values = {key: settings["env"][key] for key in ALIAS_KEYS}
    reachable = sorted(key for key, value in alias_values.items() if value == IMPLEMENTATION_MODEL)
    assert reachable == [], f"family aliases can resolve to the implementation lane: {reachable}"
    assert settings["env"].get("ANTHROPIC_SMALL_FAST_MODEL") == "claude-haiku-4-5"
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in settings["env"]
    unresolvable = sorted(
        value
        for key, value in settings["env"].items()
        if key.endswith("_MODEL") and value not in GATEWAY_MODELS
    )
    assert unresolvable == [], f"settings pin models absent from GATEWAY_MODELS: {unresolvable}"


def test_no_exported_alias_reaches_the_implementation_lane():
    checked = sorted(ALIAS_KEYS | {"ANTHROPIC_SMALL_FAST_MODEL"})
    reachable = sorted(key for key in checked if os.environ.get(key) == IMPLEMENTATION_MODEL)
    assert reachable == [], f"exported aliases point at the implementation lane: {reachable}"


def test_every_lane_model_is_one_the_gateway_serves():
    pinned = {model for model in LANES if model != "inherit"} | {MAIN_MODEL}
    assert pinned <= GATEWAY_MODELS, f"models missing from GATEWAY_MODELS: {sorted(pinned - GATEWAY_MODELS)}"


def test_any_subagent_model_override_is_the_collaborator_profile():
    exported = os.environ.get("CLAUDE_CODE_SUBAGENT_MODEL")
    assert exported in (None, COLLABORATOR_SUBAGENT_MODEL)


def test_local_settings_do_not_override_routing():
    if not LOCAL_SETTINGS.is_file():
        pytest.skip("no settings.local.json in this worktree")
    local = json.loads(LOCAL_SETTINGS.read_text())
    clashing = sorted({"agent", "model", "availableModels"} & set(local))
    local_env = local.get("env") or {}
    clashing.extend(
        sorted(
            {
                "ANTHROPIC_MODEL",
                "ANTHROPIC_SMALL_FAST_MODEL",
                "CLAUDE_CODE_SUBAGENT_MODEL",
                *ALIAS_KEYS,
            }
            & set(local_env)
        )
    )
    assert clashing == [], f"settings.local.json overrides routing keys: {clashing}"


def test_the_orchestrator_keeps_the_full_tool_set():
    fields = _frontmatter(AGENTS_DIR / f"{MAIN_AGENT}.md")
    assert "tools" not in fields
    assert fields.get("model") == "inherit"
    assert fields.get("effort") == SESSION_EFFORT


def test_every_session_starts_as_the_orchestrator():
    settings = _settings()
    assert settings.get("agent") == MAIN_AGENT
    assert (AGENTS_DIR / f"{MAIN_AGENT}.md").is_file()


def test_the_session_pins_the_max_effort_level():
    assert _settings().get("effortLevel") == SESSION_EFFORT


@pytest.mark.parametrize("agent", sorted(LANES[JUDGEMENT_MODEL]))
def test_judgement_lane_agents_pin_the_high_effort(agent: str):
    assert _frontmatter(AGENTS_DIR / f"{agent}.md").get("effort") == JUDGEMENT_EFFORT


@pytest.mark.parametrize("agent", sorted(LANES[IMPLEMENTATION_MODEL]))
def test_implementation_lane_agents_pin_no_effort(agent: str):
    assert "effort" not in _frontmatter(AGENTS_DIR / f"{agent}.md")


def test_the_session_model_is_pinned_in_both_places():
    settings = _settings()
    assert settings.get("model") == MAIN_MODEL
    assert settings.get("env", {}).get("ANTHROPIC_MODEL") == MAIN_MODEL


def test_the_claude_md_roster_table_matches_the_frontmatter():
    actual = _agent_models()
    rows = re.findall(
        r"^\| \[([\w.-]+)\]\([^)]*\) \| `([\w./-]+)` \|",
        CLAUDE_MD.read_text(),
        re.MULTILINE,
    )
    expected_names = set(actual) - NON_ROSTER
    assert {name for name, _ in rows} == expected_names
    for name, model in rows:
        assert actual[name] == model, f"CLAUDE.md assigns {name} to the wrong model"


def test_the_routing_doc_lane_table_matches_the_frontmatter():
    actual = _agent_models()
    doc = ROUTING_DOC.read_text()
    table = re.search(r"^\| Lane \| Model \| Who \|\n(?:\|.*\n)+", doc, re.MULTILINE)
    assert table is not None, "lane table not found in routing doc"
    rows = re.findall(
        r"^\| ([^|\n]+?) +\| `([\w./-]+)` \|([^\n]*)\|",
        table.group(0),
        re.MULTILINE,
    )
    assert len(rows) == len(LANES)
    by_label = {label: (model, who) for label, model, who in rows}
    assert len(by_label) == len(rows)
    assert MAIN_LANE_LABEL in by_label
    main_model, main_who = by_label[MAIN_LANE_LABEL]
    assert main_model == MAIN_MODEL
    assert MAIN_AGENT in main_who
    documented = {
        model: {name for name in re.findall(r"`([\w.-]+)`", who) if name in actual}
        for label, (model, who) in by_label.items()
        if label != MAIN_LANE_LABEL
    }
    expected = {model: names for model, names in LANES.items() if model != "inherit"}
    assert documented == expected
