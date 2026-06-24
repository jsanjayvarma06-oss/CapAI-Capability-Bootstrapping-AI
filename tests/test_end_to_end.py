import pytest

from capai import CapAI, config
from capai.exceptions import CapabilityAcquisitionError


@pytest.fixture(autouse=True)
def isolated_capai_home(tmp_path, monkeypatch):
    """Every test gets its own throwaway .capai directory so MCP servers
    and registry state never leak between tests."""
    home = tmp_path / ".capai"
    monkeypatch.setattr(config, "CAPAI_HOME", home)
    monkeypatch.setattr(config, "MCP_SERVERS_DIR", home / "mcp_servers")
    monkeypatch.setattr(config, "REGISTRY_PATH", home / "registry.json")
    config.MCP_SERVERS_DIR.mkdir(parents=True, exist_ok=True)
    yield


def test_acquires_and_registers_a_new_capability():
    events = []
    ai = CapAI(on_event=events.append)

    result = ai.run("celsius_to_fahrenheit", "Convert a Celsius temperature to Fahrenheit", 100)

    assert result == 212.0
    assert ai.has_capability("celsius_to_fahrenheit")
    stages = [e.stage for e in events]
    assert "registry_miss" in stages
    assert "mcp_created" in stages
    assert "diagnosing" in stages
    assert "writing_code" in stages
    assert "testing" in stages
    assert "promoted" in stages


def test_second_call_is_served_instantly_from_the_registry():
    ai = CapAI()
    ai.run("celsius_to_fahrenheit", "Convert a Celsius temperature to Fahrenheit", 100)

    events = []
    ai2 = CapAI(on_event=events.append, registry=ai.registry)
    result = ai2.run("celsius_to_fahrenheit", "Convert a Celsius temperature to Fahrenheit", 0)

    assert result == 32.0
    stages = [e.stage for e in events]
    assert "registry_hit" in stages
    assert "diagnosing" not in stages  # no re-acquisition needed


def test_a_second_distinct_capability_is_acquired_independently():
    ai = CapAI()
    ai.run("celsius_to_fahrenheit", "Convert a Celsius temperature to Fahrenheit", 0)
    result = ai.run("is_prime", "Check whether an integer is a prime number", 97)

    assert result is True
    assert set(ai.capabilities()) == {"celsius_to_fahrenheit", "is_prime"}


def test_slugify_is_acquired_and_handles_empty_input():
    ai = CapAI()

    result = ai.run(
        "slugify",
        "Convert text into a URL-friendly slug",
        "Hello CapAI Prototype",
    )

    assert result == "hello-capai-prototype"
    assert ai.run("slugify", "Convert text into a URL-friendly slug", "") == ""
    assert ai.has_capability("slugify")


def test_unrecognised_capability_raises_after_max_attempts():
    """The honest failure mode: the offline heuristic Code Writer doesn't
    know this task, so its generic fallback should fail the Diagnostic
    Agent's own edge-case tests and the loop should raise rather than
    silently registering something wrong."""
    ai = CapAI()
    with pytest.raises(CapabilityAcquisitionError):
        ai.run("launch_a_rocket_to_mars", "Safely land a rocket on Mars", "Starship")
