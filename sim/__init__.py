"""
Island Simulation Package.

Provides various simulation architectures:
- scalable_sim: Efficient location-based simulation with parallel processing
- sim_concordia: Full Concordia prefab-based simulation
- agents: Agent configurations and state management
- world: Island world model with locations and travel
"""

from .scalable_sim import (
    ScalableSimulation,
    SimpleAgent,
    Encounter,
    StepResult,
    create_scalable_simulation,
)
from .world import Island, Place, tick_to_time_str, tick_to_day_and_time
from .agents import AgentConfig, DEFAULT_ISLAND_AGENTS

__all__ = [
    # Scalable simulation
    "ScalableSimulation",
    "SimpleAgent",
    "Encounter",
    "StepResult",
    "create_scalable_simulation",
    # World
    "Island",
    "Place",
    "tick_to_time_str",
    "tick_to_day_and_time",
    # Agents
    "AgentConfig",
    "DEFAULT_ISLAND_AGENTS",
]
