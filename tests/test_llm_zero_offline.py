"""Unit tests enforcing strict Zero Offline Mode requirements.

Verifies:
1. --offline CLI flag rejection in mission_runner (raises SystemExit code 1).
2. Absence of offline fallback generators (OfflineRuleBasedClient).
3. Immediate mission abort and sys.exit(1) on LLM network failure or timeout.
4. FleetCoordinator rejection of empty/dummy plans.
"""
from __future__ import annotations

import argparse
from typing import Optional
from unittest import mock

import pytest

from agents.llm_client import LLMClient, LLMConnectionError, SverkLLMClient
from common.schema import PaintTask, Plan
from mission_runner import check_no_offline_flag, get_cli_parser, main, run_mission
from swarm.fleet_coordinator import FleetCoordinator


def test_cli_rejects_offline_flags() -> None:
    """Verifies that check_no_offline_flag and CLI parser reject any offline flags."""
    with pytest.raises(SystemExit) as exc_info:
        check_no_offline_flag(["--offline"])
    assert exc_info.value.code == 1

    with pytest.raises(SystemExit) as exc_info:
        check_no_offline_flag(["--sim", "-o"])
    assert exc_info.value.code == 1

    with pytest.raises(SystemExit) as exc_info:
        check_no_offline_flag(["offline=true"])
    assert exc_info.value.code == 1

    # Verify parser does not accept --offline flag
    parser = get_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--prompt", "Test prompt", "--offline"])

    # Verify main() returns exit code 1 when --offline is passed
    assert main(["--offline"]) == 1


def test_no_offline_fallback_client_exists() -> None:
    """Verifies that no offline fallback client or dummy generator exists in agents package."""
    import agents.llm_client as llm_mod
    assert not hasattr(llm_mod, "OfflineRuleBasedClient")
    assert not hasattr(llm_mod, "DummyOfflineClient")


def test_llm_connection_error_raises_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that missing API key or network failure raises LLMConnectionError without fallback."""
    monkeypatch.delenv("SVERK_LLM_API_KEY", raising=False)
    client = SverkLLMClient(api_key="")
    client.api_key = None

    with pytest.raises(LLMConnectionError, match="Missing API key"):
        client.chat("system", "user")


class FailingNetworkLLMClient(LLMClient):
    """Network failure mock client."""
    def chat(self, system_prompt: str, user_prompt: str, timeout_s: Optional[float] = None) -> str:
        raise LLMConnectionError("Sverk AI network API endpoint unreachable. Zero Offline Mode enforced.")


def test_mission_runner_aborts_on_network_failure(tmp_path: Path) -> None:
    """Verifies that run_mission exits immediately with code 1 on network failure."""
    log_dir = tmp_path / "logs_fail"
    parser = get_cli_parser()
    args = parser.parse_args(["--prompt", "Test network abort", "--sim", "--log-dir", str(log_dir)])

    failing_llm = FailingNetworkLLMClient()

    with pytest.raises(SystemExit) as exc_info:
        run_mission(args, custom_llm=failing_llm)

    assert exc_info.value.code == 1
    assert not (log_dir / "mission_report.json").exists()


def test_fleet_coordinator_rejects_empty_plan() -> None:
    """Verifies that FleetCoordinator rejects empty/dummy plans."""
    coordinator = FleetCoordinator()
    empty_plan = Plan(prompt="Empty test prompt")

    with pytest.raises(RuntimeError) as exc_info:
        coordinator.execute_plan(empty_plan)
    assert "aborted" in str(exc_info.value).lower() or "offline" in str(exc_info.value).lower()
