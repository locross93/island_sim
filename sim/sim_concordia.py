"""
Concordia-based simulation system for the island using proper prefab patterns.

This module provides:
1. Proper Concordia Config/Role-based configuration
2. Single simulation for manageable agent counts (e.g., 10 agents)
3. Multi-simulation coordinator for large agent counts (e.g., 100 agents)
"""

import dataclasses
from pathlib import Path
from typing import Any, Callable
import numpy as np

from concordia.associative_memory import basic_associative_memory
from concordia.environment import engine as engine_lib
from concordia.environment.engines import sequential
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib
from concordia.typing import simulation as simulation_lib

from .prefabs_entity import IslandEntity
from .prefabs_gm import IslandGameMaster
from .agents import AgentConfig
from .world import Island


# === CONCORDIA CONFIG TYPES ===

Config = prefab_lib.Config
Role = prefab_lib.Role
InstanceConfig = prefab_lib.InstanceConfig


def create_island_config(
    agents: list[AgentConfig],
    island: Island,
    gm_name: str = "Island Game Master",
    start_tick: int = 16,
) -> Config:
    """
    Create a Concordia Config for the island simulation.

    Args:
        agents: List of agent configurations
        island: The Island world model
        gm_name: Name for the game master
        start_tick: Starting tick (default 16 = 8:00 AM)

    Returns:
        A Concordia Config object
    """
    # Define prefabs
    prefabs = {
        'island_entity': IslandEntity(),
        'island_gm': IslandGameMaster(),
    }

    # Create entity instances from agent configs
    instances = []

    for agent_config in agents:
        # Build formative memories
        formative_memories = [
            f'{agent_config.name} lives at {agent_config.home_place}.',
        ]
        if agent_config.work_place:
            formative_memories.append(
                f'{agent_config.name} works at {agent_config.work_place}.'
            )
        formative_memories.append(
            f'{agent_config.name} has the following backstory: {agent_config.backstory}'
        )

        # Add initial relationships as memories
        if agent_config.initial_relationships:
            for other_name, affinity in agent_config.initial_relationships.items():
                sentiment = 'likes' if affinity > 20 else 'knows' if affinity > -20 else 'dislikes'
                formative_memories.append(
                    f'{agent_config.name} {sentiment} {other_name}.'
                )

        # Create entity instance config
        entity_instance = InstanceConfig(
            role=Role.ENTITY,
            prefab='island_entity',
            params={
                'name': agent_config.name,
                'home_place': agent_config.home_place,
                'work_place': agent_config.work_place,
                'personality': agent_config.personality,
                'backstory': agent_config.backstory,
                'schedule_bias': agent_config.schedule_bias.value,
                'formative_memories': formative_memories,
                'goal': f'Live according to {agent_config.name}\'s personality and fulfill basic needs while building relationships on the island.',
            }
        )
        instances.append(entity_instance)

    # Create location descriptions from island
    location_descriptions = _build_location_descriptions(island)

    # Calculate current time string
    day, time_str = _tick_to_time(start_tick)
    start_time = f'Day {day}, {time_str} (tick {start_tick})'

    # Create Game Master instance
    gm_instance = InstanceConfig(
        role=Role.GAME_MASTER,
        prefab='island_gm',
        params={
            'name': gm_name,
            'start_time': start_time,
            'locations': location_descriptions,
        }
    )
    instances.append(gm_instance)

    # Create config
    config = Config(
        prefabs=prefabs,
        instances=instances,
    )

    return config


def _build_location_descriptions(island: Island) -> str:
    """Build location descriptions from Island model."""
    desc_parts = [
        "The island has the following locations:\n"
    ]

    for place in island.places.values():
        open_start, open_end = place.open
        desc_parts.append(
            f"- {place.name} ({place.id}): {place.type}, "
            f"open from tick {open_start} to {open_end}"
        )

    desc_parts.append("\nLocations are connected by paths with travel times.")

    return '\n'.join(desc_parts)


def _tick_to_time(tick: int) -> tuple[int, str]:
    """Convert tick to (day, time_string)."""
    TICKS_PER_DAY = 48
    day = (tick // TICKS_PER_DAY) + 1
    tick_of_day = tick % TICKS_PER_DAY
    hour = tick_of_day // 2  # 2 ticks per hour
    minute = (tick_of_day % 2) * 30
    time_str = f"{hour:02d}:{minute:02d}"
    return day, time_str


# === SIMULATION CLASS ===

class IslandSimulationConcordia(simulation_lib.Simulation):
    """
    Island simulation using proper Concordia architecture.

    This wraps the Concordia generic simulation with island-specific
    functionality while maintaining proper prefab patterns.
    """

    def __init__(
        self,
        config: Config,
        model: language_model.LanguageModel,
        embedder: Callable[[str], np.ndarray],
        island: Island,
        engine: engine_lib.Engine | None = None,
    ):
        """Initialize the island simulation.

        Args:
            config: The Concordia config
            model: The language model
            embedder: The sentence embedder
            island: The Island world model
            engine: The simulation engine (defaults to Sequential)
        """
        self._config = config
        self._model = model
        self._embedder = embedder
        self._island = island
        self._engine = engine or sequential.Sequential()

        self.game_masters = []
        self.entities = []

        # Shared game master memory bank
        self.game_master_memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=embedder,
        )

        # Separate instances by role
        all_data = self._config.instances
        gm_configs = [
            cfg for cfg in all_data if cfg.role == Role.GAME_MASTER
        ]
        entity_configs = [
            cfg for cfg in all_data if cfg.role == Role.ENTITY
        ]

        # Build entities first
        for entity_config in entity_configs:
            self._add_entity(entity_config)

        # Build game masters (with references to entities)
        for gm_config in gm_configs:
            self._add_game_master(gm_config)

    def _add_entity(self, instance_config: InstanceConfig):
        """Add an entity to the simulation."""
        entity_prefab = self._config.prefabs[instance_config.prefab]
        entity_prefab.params = instance_config.params

        # Create personal memory bank
        memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=self._embedder,
        )

        # Build entity
        entity = entity_prefab.build(model=self._model, memory_bank=memory_bank)
        self.entities.append(entity)

    def _add_game_master(self, instance_config: InstanceConfig):
        """Add a game master to the simulation."""
        gm_prefab = self._config.prefabs[instance_config.prefab]
        gm_prefab.params = instance_config.params
        gm_prefab.entities = self.entities  # Give GM references to entities

        # Build game master
        game_master = gm_prefab.build(
            model=self._model,
            memory_bank=self.game_master_memory_bank
        )
        self.game_masters.append(game_master)

    def get_game_masters(self) -> list[entity_lib.Entity]:
        """Get the game masters."""
        return list(self.game_masters)

    def get_entities(self) -> list[entity_lib.Entity]:
        """Get the entities."""
        return list(self.entities)

    def get_island(self) -> Island:
        """Get the Island world model."""
        return self._island

    def step(self, action_spec: Any = None) -> str:
        """Run one step of the simulation.

        Args:
            action_spec: Optional action specification

        Returns:
            Result string from the step
        """
        # Use the engine to run one step
        # This will invoke game master and entity act() methods
        result = self._engine.step(
            entities=self.entities + self.game_masters,
            action_spec=action_spec,
        )
        return result

    def run(self, num_steps: int) -> list[str]:
        """Run the simulation for multiple steps.

        Args:
            num_steps: Number of steps to run

        Returns:
            List of result strings
        """
        results = []
        for _ in range(num_steps):
            result = self.step()
            results.append(result)
        return results


# === MULTI-SIMULATION COORDINATOR ===

@dataclasses.dataclass
class NeighborhoodSimulation:
    """A sub-simulation for a neighborhood on the island."""
    name: str
    simulation: IslandSimulationConcordia
    agent_names: list[str]


class MultiIslandSimulation:
    """
    Coordinator for running multiple island simulations in parallel.

    This allows handling 100+ agents by splitting them across neighborhoods
    or economic zones, running each as a separate Concordia simulation.
    Agents can swap between simulations when they travel to other areas.
    """

    def __init__(
        self,
        neighborhoods: list[NeighborhoodSimulation],
    ):
        """Initialize the multi-simulation coordinator.

        Args:
            neighborhoods: List of neighborhood simulations
        """
        self.neighborhoods = neighborhoods
        self._current_tick = 0

    def step_all(self) -> dict[str, list[str]]:
        """
        Step all neighborhood simulations in parallel.

        Returns:
            Dictionary mapping neighborhood name to results
        """
        results = {}
        for neighborhood in self.neighborhoods:
            result = neighborhood.simulation.step()
            results[neighborhood.name] = [result]

        self._current_tick += 1
        return results

    def run_all(self, num_steps: int) -> dict[str, list[str]]:
        """
        Run all simulations for multiple steps.

        Args:
            num_steps: Number of steps to run

        Returns:
            Dictionary mapping neighborhood name to results
        """
        all_results = {n.name: [] for n in self.neighborhoods}

        for _ in range(num_steps):
            step_results = self.step_all()
            for name, results in step_results.items():
                all_results[name].extend(results)

        return all_results

    def get_agent_location(self, agent_name: str) -> tuple[str, Any]:
        """
        Find which neighborhood simulation contains an agent.

        Args:
            agent_name: The agent's name

        Returns:
            Tuple of (neighborhood_name, entity)
        """
        for neighborhood in self.neighborhoods:
            for entity in neighborhood.simulation.get_entities():
                if entity.name == agent_name:
                    return neighborhood.name, entity

        return None, None

    def swap_agent(
        self,
        agent_name: str,
        from_neighborhood: str,
        to_neighborhood: str
    ) -> bool:
        """
        Swap an agent from one simulation to another.

        This would be called when an agent travels to a different area.

        Args:
            agent_name: The agent to swap
            from_neighborhood: Source neighborhood
            to_neighborhood: Destination neighborhood

        Returns:
            True if swap was successful
        """
        # This is a placeholder for cross-simulation agent movement
        # Full implementation would require saving/loading agent state
        # and updating both simulations
        raise NotImplementedError(
            "Cross-simulation agent swapping not yet implemented. "
            "For now, keep agent counts per simulation manageable."
        )


def create_neighborhood_simulations(
    all_agents: list[AgentConfig],
    island: Island,
    model: language_model.LanguageModel,
    embedder: Callable[[str], np.ndarray],
    max_agents_per_sim: int = 25,
) -> list[NeighborhoodSimulation]:
    """
    Create multiple neighborhood simulations by splitting agents.

    Args:
        all_agents: All agent configurations
        island: The island world model
        model: The language model
        embedder: The sentence embedder
        max_agents_per_sim: Maximum agents per simulation

    Returns:
        List of NeighborhoodSimulation objects
    """
    neighborhoods = []

    # Group agents by home location (neighborhood)
    agents_by_home = {}
    for agent in all_agents:
        home = agent.home_place
        if home not in agents_by_home:
            agents_by_home[home] = []
        agents_by_home[home].append(agent)

    # Create simulations, combining neighborhoods if needed
    current_batch = []
    current_batch_name_parts = []

    for home_place, agents in agents_by_home.items():
        if len(current_batch) + len(agents) > max_agents_per_sim and current_batch:
            # Create simulation for current batch
            batch_name = '_'.join(current_batch_name_parts[:3])  # Limit name length
            config = create_island_config(current_batch, island, gm_name=f"GM_{batch_name}")
            sim = IslandSimulationConcordia(
                config=config,
                model=model,
                embedder=embedder,
                island=island,
            )
            neighborhoods.append(NeighborhoodSimulation(
                name=batch_name,
                simulation=sim,
                agent_names=[a.name for a in current_batch],
            ))

            # Start new batch
            current_batch = agents
            current_batch_name_parts = [home_place]
        else:
            current_batch.extend(agents)
            current_batch_name_parts.append(home_place)

    # Create final batch
    if current_batch:
        batch_name = '_'.join(current_batch_name_parts[:3])
        config = create_island_config(current_batch, island, gm_name=f"GM_{batch_name}")
        sim = IslandSimulationConcordia(
            config=config,
            model=model,
            embedder=embedder,
            island=island,
        )
        neighborhoods.append(NeighborhoodSimulation(
            name=batch_name,
            simulation=sim,
            agent_names=[a.name for a in current_batch],
        ))

    return neighborhoods
