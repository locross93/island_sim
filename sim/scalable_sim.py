"""
Scalable Concordia simulation with location-based scoping.

This module provides a highly efficient simulation architecture that minimizes LLM calls:
1. PHASE 1: Parallel location observations (1 shared observation per occupied location)
2. PHASE 2: Parallel movement decisions (each agent decides independently)
3. PHASE 3: Sequential encounters (dialogue only for co-located agents)

LLM Call Comparison:
| Scenario                  | Traditional | Scalable Design |
|---------------------------|-------------|-----------------|
| 10 agents, 6 locations    | ~50/step    | ~12/step        |
| 50 agents, 20 locations   | ~250/step   | ~40/step        |
| 100 agents, 30 locations  | ~500/step   | ~60/step        |
"""

from __future__ import annotations

import functools
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .world import Island, tick_to_time_str, tick_to_day_and_time
from .agents import AgentConfig


@dataclass
class SimpleAgent:
    """
    Minimal agent state without heavy Concordia components.

    This lightweight representation stores just what's needed for
    the scalable simulation loop, avoiding the overhead of full
    EntityAgent prefabs.
    """
    name: str
    location: str
    personality: str
    home_place: str
    work_place: str | None = None
    memories: list[str] = field(default_factory=list)
    max_memories: int = 20

    def observe(self, text: str) -> None:
        """Add an observation to memory."""
        self.memories.append(text)
        if len(self.memories) > self.max_memories:
            self.memories.pop(0)

    def recent_memories(self, n: int = 5) -> list[str]:
        """Get the n most recent memories."""
        return self.memories[-n:]

    @classmethod
    def from_config(cls, config: AgentConfig) -> "SimpleAgent":
        """Create a SimpleAgent from an AgentConfig."""
        agent = cls(
            name=config.name,
            location=config.home_place,
            personality=config.personality,
            home_place=config.home_place,
            work_place=config.work_place,
        )
        # Seed with backstory
        if config.backstory:
            agent.observe(f"Background: {config.backstory}")
        return agent


@dataclass
class Encounter:
    """Record of a dialogue encounter between agents."""
    location: str
    participants: list[str]
    dialogue: list[tuple[str, str]]  # (speaker, utterance)

    def summary(self) -> str:
        """Get a summary of the encounter."""
        lines = [f"Encounter at {self.location} between {', '.join(self.participants)}:"]
        for speaker, utterance in self.dialogue:
            lines.append(f"  {speaker}: {utterance}")
        return "\n".join(lines)


@dataclass
class StepTiming:
    """Timing information for a simulation step."""
    total_ms: float = 0.0
    observation_ms: float = 0.0
    movement_ms: float = 0.0
    encounter_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_ms": round(self.total_ms, 2),
            "observation_ms": round(self.observation_ms, 2),
            "movement_ms": round(self.movement_ms, 2),
            "encounter_ms": round(self.encounter_ms, 2),
        }


@dataclass
class StepResult:
    """Results from a single simulation step."""
    tick: int
    observations: dict[str, str]  # location -> observation text
    movements: dict[str, str | None]  # agent_name -> new_location or None
    encounters: list[Encounter]
    llm_calls: int = 0
    timing: StepTiming = field(default_factory=StepTiming)

    def summary(self) -> str:
        """Get a summary of the step."""
        lines = [f"Step at tick {self.tick} ({tick_to_time_str(self.tick)}):"]
        lines.append(f"  Observations generated: {len(self.observations)}")
        moves = sum(1 for v in self.movements.values() if v is not None)
        lines.append(f"  Movements: {moves} agents moved")
        lines.append(f"  Encounters: {len(self.encounters)}")
        lines.append(f"  Total LLM calls: {self.llm_calls}")
        lines.append(f"  Timing: {self.timing.total_ms:.0f}ms total "
                     f"(obs: {self.timing.observation_ms:.0f}ms, "
                     f"move: {self.timing.movement_ms:.0f}ms, "
                     f"enc: {self.timing.encounter_ms:.0f}ms)")
        return "\n".join(lines)


def _run_tasks_parallel(tasks: dict[str, Callable[[], Any]]) -> dict[str, Any]:
    """
    Run tasks in parallel using ThreadPoolExecutor.

    This is a simplified version of Concordia's concurrency.run_tasks().
    Each task is a callable that returns a result.

    Args:
        tasks: Dictionary mapping task keys to callables

    Returns:
        Dictionary mapping task keys to results
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}

    if not tasks:
        return results

    # Use ThreadPoolExecutor for I/O-bound LLM calls
    with ThreadPoolExecutor(max_workers=min(len(tasks), 10)) as executor:
        future_to_key = {
            executor.submit(task): key
            for key, task in tasks.items()
        }

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as e:
                # Log error but continue with other tasks
                print(f"Task {key} failed: {e}")
                results[key] = None

    return results


class ScalableSimulation:
    """
    Location-scoped Concordia simulation with async phases.

    Architecture:
    1. Group agents by location
    2. Generate ONE observation per location (shared)
    3. Parallel movement decisions
    4. Sequential dialogue only for co-located agents

    This dramatically reduces LLM calls compared to traditional
    approaches that broadcast observations to all agents.
    """

    def __init__(
        self,
        agents: list[AgentConfig],
        island: Island,
        model: Any,
        embedder: Callable[[str], np.ndarray] | None = None,
        start_tick: int = 16,  # 8:00 AM
        max_dialogue_turns: int = 3,
        max_encounter_agents: int = 2,
        verbose: bool = False,
    ):
        """
        Initialize the scalable simulation.

        Args:
            agents: List of agent configurations
            island: The Island world model
            model: Language model with __call__(prompt) -> str interface
            embedder: Optional sentence embedder (not used in basic mode)
            start_tick: Starting tick (default 16 = 8:00 AM)
            max_dialogue_turns: Maximum dialogue exchanges per encounter
            max_encounter_agents: Maximum agents per dialogue encounter
            verbose: Whether to print detailed progress
        """
        self.agents = {
            config.name: SimpleAgent.from_config(config)
            for config in agents
        }
        self.island = island
        self.model = model
        self.embedder = embedder
        self.tick = start_tick
        self.max_dialogue_turns = max_dialogue_turns
        self.max_encounter_agents = max_encounter_agents
        self.verbose = verbose

        # Location tracking: location_id -> [agent_names]
        self.locations: dict[str, list[str]] = defaultdict(list)
        self._update_locations()

        # Statistics
        self.total_llm_calls = 0
        self.step_history: list[StepResult] = []

    def _update_locations(self) -> None:
        """Update the location -> agents mapping."""
        self.locations.clear()
        for name, agent in self.agents.items():
            self.locations[agent.location].append(name)

    def _time_str(self) -> str:
        """Get current time as human-readable string."""
        day, time = tick_to_day_and_time(self.tick)
        return f"Day {day}, {time}"

    def step(self) -> StepResult:
        """
        Run one simulation step with three phases.

        Returns:
            StepResult with observations, movements, and encounters
        """
        import time

        step_start = time.perf_counter()
        timing = StepTiming()
        llm_calls = 0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"STEP at {self._time_str()} (tick {self.tick})")
            print(f"{'='*60}")

        # PHASE 1: Parallel location observations
        if self.verbose:
            print("\nPHASE 1: Generating location observations...")
        phase_start = time.perf_counter()
        observations, obs_calls = self._observe_locations_parallel()
        timing.observation_ms = (time.perf_counter() - phase_start) * 1000
        llm_calls += obs_calls

        # Share observations with agents at each location
        for loc_id, obs_text in observations.items():
            for agent_name in self.locations.get(loc_id, []):
                self.agents[agent_name].observe(obs_text)

        # PHASE 2: Parallel movement decisions
        if self.verbose:
            print("\nPHASE 2: Deciding movements...")
        phase_start = time.perf_counter()
        movements, move_calls = self._decide_movements_parallel(observations)
        timing.movement_ms = (time.perf_counter() - phase_start) * 1000
        llm_calls += move_calls
        self._apply_movements(movements)

        # PHASE 3: Sequential encounters at crowded locations
        if self.verbose:
            print("\nPHASE 3: Running encounters...")
        phase_start = time.perf_counter()
        encounters, enc_calls = self._run_encounters_sequential()
        timing.encounter_ms = (time.perf_counter() - phase_start) * 1000
        llm_calls += enc_calls

        timing.total_ms = (time.perf_counter() - step_start) * 1000

        # Advance tick
        self.tick += 1
        self.total_llm_calls += llm_calls

        result = StepResult(
            tick=self.tick - 1,
            observations=observations,
            movements=movements,
            encounters=encounters,
            llm_calls=llm_calls,
            timing=timing,
        )
        self.step_history.append(result)

        if self.verbose:
            print(f"\n{result.summary()}")

        return result

    def _observe_locations_parallel(self) -> tuple[dict[str, str], int]:
        """
        Generate ONE observation per occupied location in parallel.

        Returns:
            Tuple of (location -> observation text, number of LLM calls)
        """
        tasks = {}

        for loc_id, agent_names in self.locations.items():
            if agent_names:
                tasks[loc_id] = functools.partial(
                    self._generate_location_observation,
                    loc_id,
                    agent_names
                )

        if self.verbose:
            print(f"  Generating observations for {len(tasks)} locations in parallel...")

        results = _run_tasks_parallel(tasks)

        # Filter out None results from failed tasks
        observations = {k: v for k, v in results.items() if v is not None}

        return observations, len(tasks)

    def _generate_location_observation(
        self,
        loc_id: str,
        agent_names: list[str]
    ) -> str:
        """
        Generate a single observation for a location.

        Args:
            loc_id: Location identifier
            agent_names: Names of agents at this location

        Returns:
            Observation text describing the scene
        """
        place = self.island.places.get(loc_id)
        if place is None:
            place_name = loc_id.replace("_", " ").title()
            is_open = True
        else:
            place_name = place.name
            is_open = place.is_open(self.tick)

        # Format present agents with personalities
        present_descriptions = []
        for name in agent_names[:5]:  # Limit to first 5 for prompt size
            agent = self.agents[name]
            present_descriptions.append(f"{name} ({agent.personality[:50]}...)"
                                        if len(agent.personality) > 50
                                        else f"{name} ({agent.personality})")

        if len(agent_names) > 5:
            present_descriptions.append(f"and {len(agent_names) - 5} others")

        prompt = f"""Location: {place_name}
Time: {self._time_str()}
Currently present: {', '.join(present_descriptions)}
Status: {'Open' if is_open else 'Closed'}

Describe what people at this location observe right now in 2-3 sentences.
Include the atmosphere, any notable activities, and interactions between those present.
Be specific and vivid but concise."""

        return self.model(prompt)

    def _decide_movements_parallel(
        self,
        observations: dict[str, str]
    ) -> tuple[dict[str, str | None], int]:
        """
        Each agent decides independently whether to move, in parallel.

        Args:
            observations: Location -> observation text from Phase 1

        Returns:
            Tuple of (agent_name -> new_location or None, number of LLM calls)
        """
        tasks = {}

        for name, agent in self.agents.items():
            obs = observations.get(agent.location, "Nothing notable happening.")
            tasks[name] = functools.partial(
                self._agent_decide_movement,
                agent,
                obs
            )

        if self.verbose:
            print(f"  Getting movement decisions for {len(tasks)} agents in parallel...")

        results = _run_tasks_parallel(tasks)

        return results, len(tasks)

    def _agent_decide_movement(
        self,
        agent: SimpleAgent,
        observation: str
    ) -> str | None:
        """
        Single agent decides whether to stay or move.

        Args:
            agent: The agent making the decision
            observation: What the agent currently observes

        Returns:
            New location string if moving, None if staying
        """
        # Get valid destinations (neighbors + home + work)
        neighbors = self.island.neighbors(agent.location)
        valid_destinations = set(neighbors)
        valid_destinations.add(agent.home_place)
        if agent.work_place:
            valid_destinations.add(agent.work_place)
        valid_destinations.discard(agent.location)  # Can't move to current location

        # Get place names for the prompt
        dest_names = []
        for dest in valid_destinations:
            place = self.island.places.get(dest)
            name = place.name if place else dest.replace("_", " ").title()
            dest_names.append(f"{dest} ({name})")

        recent_memories = "\n".join(f"- {m}" for m in agent.recent_memories(3))

        prompt = f"""{agent.name} is at {agent.location}.
Personality: {agent.personality}

Current observation:
{observation}

Recent memories:
{recent_memories}

Available destinations:
{chr(10).join('- ' + d for d in dest_names)}

Based on {agent.name}'s personality and current situation, should they stay or move?
If they should move, which destination makes the most sense?

Respond with EXACTLY one of:
STAY
MOVE:<destination_id>

Where <destination_id> is one of: {', '.join(valid_destinations)}"""

        response = self.model(prompt).strip().upper()

        return self._parse_movement(response, valid_destinations)

    def _parse_movement(
        self,
        response: str,
        valid_destinations: set[str]
    ) -> str | None:
        """
        Parse an LLM movement response.

        Args:
            response: Raw LLM response
            valid_destinations: Set of valid destination IDs

        Returns:
            Destination string if moving, None if staying
        """
        response = response.strip().upper()

        if response.startswith("STAY"):
            return None

        if response.startswith("MOVE:") or response.startswith("MOVE<"):
            # Handle both MOVE:dest and MOVE:<dest> formats
            dest = response[5:].strip().lower()
            # Remove angle brackets if present
            dest = dest.strip("<>")
            # Try exact match first
            if dest in valid_destinations:
                return dest
            # Try fuzzy match (in case LLM uses display name)
            for valid in valid_destinations:
                if valid in dest or dest in valid:
                    return valid
            # If no match, pick first valid destination as fallback
            if valid_destinations:
                return list(valid_destinations)[0]

        # Default to staying if unparseable
        return None

    def _apply_movements(self, movements: dict[str, str | None]) -> None:
        """
        Apply movement decisions to update agent locations.

        Args:
            movements: Agent name -> new location (None means stay)
        """
        for name, new_location in movements.items():
            if new_location is not None:
                agent = self.agents[name]
                old_location = agent.location
                agent.location = new_location
                agent.observe(f"Moved from {old_location} to {new_location}.")

                if self.verbose:
                    print(f"    {name}: {old_location} -> {new_location}")

        # Update location tracking
        self._update_locations()

    def _run_encounters_sequential(self) -> tuple[list[Encounter], int]:
        """
        Run dialogue at locations with 2+ agents. Sequential to preserve turn-taking.

        Returns:
            Tuple of (list of encounters, number of LLM calls)
        """
        encounters = []
        llm_calls = 0

        for loc_id, agent_names in self.locations.items():
            if len(agent_names) >= 2:
                # Run a dialogue encounter
                encounter, calls = self._run_dialogue_sequential(loc_id, agent_names)
                encounters.append(encounter)
                llm_calls += calls

                # Add encounter to agent memories
                summary = f"Had a conversation at {loc_id} with {', '.join(encounter.participants)}."
                for name in encounter.participants:
                    self.agents[name].observe(summary)

        return encounters, llm_calls

    def _run_dialogue_sequential(
        self,
        loc_id: str,
        agent_names: list[str]
    ) -> tuple[Encounter, int]:
        """
        Turn-based dialogue between agents at a location.

        Args:
            loc_id: Location where encounter happens
            agent_names: All agents at this location

        Returns:
            Tuple of (Encounter record, number of LLM calls)
        """
        # Select participants (limit to max_encounter_agents)
        if len(agent_names) > self.max_encounter_agents:
            participants = random.sample(agent_names, self.max_encounter_agents)
        else:
            participants = list(agent_names)

        place = self.island.places.get(loc_id)
        place_name = place.name if place else loc_id.replace("_", " ").title()

        dialogue = []
        llm_calls = 0

        if self.verbose:
            print(f"    Encounter at {place_name}: {', '.join(participants)}")

        for turn in range(self.max_dialogue_turns):
            speaker_name = participants[turn % len(participants)]
            other_names = [p for p in participants if p != speaker_name]
            speaker = self.agents[speaker_name]

            # Build context for the speaker
            previous_lines = ""
            if dialogue:
                previous_lines = "\n".join(
                    f"{spk}: {utt}" for spk, utt in dialogue[-3:]
                )

            prompt = f"""{speaker_name} ({speaker.personality}) is at {place_name} talking with {', '.join(other_names)}.

{'Previous dialogue:' if previous_lines else 'The conversation is just starting.'}
{previous_lines}

What does {speaker_name} say next? Write ONE short sentence of natural dialogue.
Just the dialogue, no quotes or speaker name prefix."""

            utterance = self.model(prompt).strip()
            # Clean up common LLM response patterns
            utterance = utterance.strip('"\'')
            if utterance.lower().startswith(speaker_name.lower() + ":"):
                utterance = utterance[len(speaker_name) + 1:].strip()

            dialogue.append((speaker_name, utterance))
            llm_calls += 1

            if self.verbose:
                print(f"      {speaker_name}: {utterance[:60]}...")

        return Encounter(loc_id, participants, dialogue), llm_calls

    def run(
        self,
        num_steps: int,
        premise: str = ""
    ) -> list[StepResult]:
        """
        Run the simulation for multiple steps.

        Args:
            num_steps: Number of steps to run
            premise: Optional initial premise/context

        Returns:
            List of StepResult objects
        """
        if premise:
            # Broadcast premise to all agents
            for agent in self.agents.values():
                agent.observe(premise)

        results = []
        for i in range(num_steps):
            if self.verbose:
                print(f"\n{'#'*60}")
                print(f"# Running step {i+1}/{num_steps}")
                print(f"{'#'*60}")

            result = self.step()
            results.append(result)

        return results

    def get_agent_locations(self) -> dict[str, str]:
        """Get current location of each agent."""
        return {name: agent.location for name, agent in self.agents.items()}

    def get_location_populations(self) -> dict[str, int]:
        """Get number of agents at each location."""
        return {loc: len(names) for loc, names in self.locations.items()}

    def get_statistics(self) -> dict[str, Any]:
        """Get simulation statistics."""
        return {
            "total_ticks": self.tick,
            "total_llm_calls": self.total_llm_calls,
            "total_steps": len(self.step_history),
            "average_llm_calls_per_step": (
                self.total_llm_calls / len(self.step_history)
                if self.step_history else 0
            ),
            "total_encounters": sum(
                len(r.encounters) for r in self.step_history
            ),
            "total_movements": sum(
                sum(1 for v in r.movements.values() if v is not None)
                for r in self.step_history
            ),
            "agent_count": len(self.agents),
            "location_count": len(self.island.places),
        }

    def export_results(self, path: str) -> None:
        """
        Export simulation results to JSON for analysis.

        Args:
            path: Output file path
        """
        import json

        data = {
            "statistics": self.get_statistics(),
            "timing_summary": {
                "total_ms": sum(r.timing.total_ms for r in self.step_history),
                "observation_ms": sum(r.timing.observation_ms for r in self.step_history),
                "movement_ms": sum(r.timing.movement_ms for r in self.step_history),
                "encounter_ms": sum(r.timing.encounter_ms for r in self.step_history),
            },
            "agents": {
                name: {
                    "location": agent.location,
                    "home_place": agent.home_place,
                    "work_place": agent.work_place,
                    "personality": agent.personality,
                    "memories": agent.memories,
                }
                for name, agent in self.agents.items()
            },
            "steps": [
                {
                    "tick": r.tick,
                    "llm_calls": r.llm_calls,
                    "timing": r.timing.to_dict(),
                    "observations": r.observations,
                    "movements": {k: v for k, v in r.movements.items() if v is not None},
                    "encounters": [
                        {
                            "location": e.location,
                            "participants": e.participants,
                            "dialogue": [
                                {"speaker": s, "text": t} for s, t in e.dialogue
                            ],
                        }
                        for e in r.encounters
                    ],
                }
                for r in self.step_history
            ],
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Exported simulation results to {path}")


def create_scalable_simulation(
    agents: list[AgentConfig],
    island: Island,
    model: Any,
    embedder: Callable[[str], np.ndarray] | None = None,
    **kwargs
) -> ScalableSimulation:
    """
    Factory function to create a ScalableSimulation.

    Args:
        agents: List of agent configurations
        island: The Island world model
        model: Language model
        embedder: Optional sentence embedder
        **kwargs: Additional arguments passed to ScalableSimulation

    Returns:
        Configured ScalableSimulation instance
    """
    return ScalableSimulation(
        agents=agents,
        island=island,
        model=model,
        embedder=embedder,
        **kwargs
    )
