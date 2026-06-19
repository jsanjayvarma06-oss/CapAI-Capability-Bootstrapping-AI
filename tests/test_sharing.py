import pytest

from capai import config
from capai.manager_agent import ManagerAgent
from capai.models import Capability, CapabilitySpec, VerificationResult
from capai.registry import CapabilityRegistry
from capai.sharing import CapabilityExchange, TrustedImporter


@pytest.fixture(autouse=True)
def isolated_mcp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MCP_SERVERS_DIR", tmp_path / "mcp_servers")
    config.MCP_SERVERS_DIR.mkdir(parents=True, exist_ok=True)
    yield


def _seed_capability_on(registry: CapabilityRegistry, name: str = "double_it") -> None:
    spec = CapabilitySpec(
        name=name,
        description="Doubles a number",
        signature=f"def {name}(x):",
        example_inputs=[[4]],
        expected_behavior="Returns x multiplied by 2.",
    )
    source = (
        f"def {name}(x):\n"
        f"    if not isinstance(x, (int, float)) or isinstance(x, bool):\n"
        f"        raise TypeError('x must be numeric')\n"
        f"    return x * 2\n"
    )
    capability = Capability(name=name, description=spec.description, source_code=source,
                             spec=spec, mcp_id="owner-mcp-seed")
    verification = VerificationResult(passed=True, layer_results={"seed": True}, details=["seed capability"])
    promoted = ManagerAgent(registry).review_and_promote(capability, verification)
    assert promoted is True


def test_owner_can_list_what_it_offers(tmp_path):
    owner_registry = CapabilityRegistry(path=tmp_path / "owner.json")
    _seed_capability_on(owner_registry)

    exchange = CapabilityExchange(owner_registry, owner_id="agent-A")
    offered = exchange.list_offered()

    assert len(offered) == 1
    assert offered[0]["name"] == "double_it"
    assert offered[0]["owner_id"] == "agent-A"


def test_receiver_independently_reverifies_before_trusting(tmp_path):
    owner_registry = CapabilityRegistry(path=tmp_path / "owner.json")
    _seed_capability_on(owner_registry)
    exchange = CapabilityExchange(owner_registry, owner_id="agent-A")

    receiver_registry = CapabilityRegistry(path=tmp_path / "receiver.json")
    importer = TrustedImporter(receiver_registry)

    imported = importer.import_from(exchange, "double_it")

    assert imported is True
    assert receiver_registry.has("double_it")
    imported_cap = receiver_registry.get("double_it")
    assert imported_cap.verification["imported_from"] == "agent-A"


def test_importing_a_capability_that_does_not_exist_fails_safely(tmp_path):
    owner_registry = CapabilityRegistry(path=tmp_path / "owner.json")
    exchange = CapabilityExchange(owner_registry, owner_id="agent-A")

    receiver_registry = CapabilityRegistry(path=tmp_path / "receiver.json")
    importer = TrustedImporter(receiver_registry)

    assert importer.import_from(exchange, "nonexistent_capability") is False
