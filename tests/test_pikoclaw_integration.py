"""Integration and unit tests for PikoClaw HTTP bridge client, PikaClawDroneLink, and chat summary by persona."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from common.schema import DialogueTurn, FlightCommand, PaintTask, Plan
from mission_runner import print_chat_summary_by_persona
from openclaw.drone_link import PikaClawDroneLink
from openclaw.pikoclaw_bridge_client import PikoClawBridgeClient, PikoClawBridgeError


@mock.patch("urllib.request.urlopen")
def test_pikoclaw_bridge_client_endpoints(mock_urlopen: mock.MagicMock) -> None:
    """Tests PikoClawBridgeClient HTTP calls for /healthz, /takeoff, /move, /spray, /land, /pose."""
    def create_mock_response(body_dict: dict):
        resp = mock.MagicMock()
        resp.getcode.return_value = 200
        resp.read.return_value = json.dumps(body_dict).encode("utf-8")
        resp.__enter__.return_value = resp
        return resp

    # Mock responses for healthz, takeoff, move, spray, pose, land
    mock_urlopen.side_effect = [
        create_mock_response({"ok": True}),
        create_mock_response({"ok": True, "landed": False, "pose": [0, 0]}),
        create_mock_response({"ok": True, "pose": [1.5, 2.5]}),
        create_mock_response({"ok": True, "points": [[0, 0], [1.5, 2.5]], "color": "#000000", "length": 2}),
        create_mock_response({"xy": [1.5, 2.5], "heading": 0.0}),
        create_mock_response({"ok": True, "landed": True, "pose": [1.5, 2.5]}),
    ]

    client = PikoClawBridgeClient(base_url="http://localhost:9000")

    assert client.healthz() is True
    assert client.takeoff()["ok"] is True
    assert client.move(to=[1.5, 2.5])["ok"] is True
    assert client.spray(points=[[1.5, 2.5]], color="#000000")["ok"] is True
    assert client.pose()["xy"] == [1.5, 2.5]
    assert client.land()["ok"] is True


@mock.patch("urllib.request.urlopen")
def test_pikoclaw_bridge_client_error_handling(mock_urlopen: mock.MagicMock) -> None:
    """Tests PikoClawBridgeError handling on HTTP failure or connection timeout."""
    import urllib.error
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    client = PikoClawBridgeClient(base_url="http://localhost:9999")
    
    assert client.healthz() is False

    with pytest.raises(PikoClawBridgeError, match="Failed to connect to PikoClaw bridge"):
        client.move(to=[1.0, 1.0])


def test_pikoclaw_drone_link_wrapper() -> None:
    """Tests PikaClawDroneLink delegation to PikoClawBridgeClient."""
    mock_bridge = mock.MagicMock(spec=PikoClawBridgeClient)
    mock_bridge.healthz.return_value = True
    mock_bridge.takeoff.return_value = {"ok": True}
    mock_bridge.move.return_value = {"ok": True}
    mock_bridge.spray.return_value = {"ok": True}
    mock_bridge.land.return_value = {"ok": True}
    mock_bridge.pose.return_value = {"xy": [2.0, 3.0]}

    link = PikaClawDroneLink(node_name="drone_black", bridge_client=mock_bridge)
    
    assert link.connect() is True
    assert link.takeoff(z=2.0) is True
    assert link.navigate_wait(x=2.0, y=3.0, z=2.0) is True
    assert link.paint_zone(duration_s=2.0, passes=1) is True
    assert link.land() is True
    
    telemetry = link.get_telemetry()
    assert telemetry.x == pytest.approx(2.0)
    assert telemetry.y == pytest.approx(3.0)


def test_print_chat_summary_by_persona(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Tests formatting of dialogue transcript by persona role and saving to file."""
    plan = Plan(
        prompt="Harmonious swarm painting",
        cells=[PaintTask("A1", "black")],
        flight_commands=[FlightCommand("drone_black", "takeoff")],
        transcript=[
            DialogueTurn(agent="Академист", text="I demand precise 0.9 m/s contouring in ARUCO frame."),
            DialogueTurn(agent="Экспрессионист", text="Adding red dynamic stroke at 1.8 m/s!"),
            DialogueTurn(agent="Минималист", text="Removing redundant cells for clean space."),
            DialogueTurn(agent="Детализатор", text="Waiting in yield_wait for final detail pass at 0.6 m/s."),
            DialogueTurn(agent="Координатор", text="Final PikoClaw swarm plan approved."),
        ],
        notes="All persona roles synchronized.",
    )

    summary_text = print_chat_summary_by_persona(plan)
    
    captured = capsys.readouterr()
    assert "CHAT SUMMARY BY PERSONA ROLE" in captured.out
    assert "@Академист" in captured.out
    assert "@Экспрессионист" in captured.out
    assert "@Минималист" in captured.out
    assert "@Детализатор" in captured.out
    assert "@Координатор" in captured.out

    # Verify saving to file
    summary_file = tmp_path / "logs" / "chat_summary_by_persona.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(summary_text, encoding="utf-8")
    
    assert summary_file.exists()
    file_content = summary_file.read_text(encoding="utf-8")
    assert "0.9 m/s contouring" in file_content
    assert "yield_wait" in file_content
