"""Multi-persona LLM agent system for Sverk PikoClaw Swarm."""
from agents.personas import (
    COLOR_TO_AGENT,
    AGENT_TO_COLOR,
    AGENT_TO_DRONE,
    PERSONA_SPEEDS,
    get_agent_prompt,
    get_coordinator_prompt,
)

__all__ = [
    "COLOR_TO_AGENT",
    "AGENT_TO_COLOR",
    "AGENT_TO_DRONE",
    "PERSONA_SPEEDS",
    "get_agent_prompt",
    "get_coordinator_prompt",
]
