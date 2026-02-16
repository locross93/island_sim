# AsyncLocationEngine Implementation Plan

## Overview

A new Concordia engine that supports location-scoped parallel execution with:
- **Async phases** for observations and actions
- **Lock-protected location state**
- **Two agent modes**: Isolated (solo observation/action) vs Social (dialogue)
- **Parallel dialogues** using dialogic GM pattern
- **Time synchronization** across all agents

## Architecture

### Core Insight

Instead of "one entity acts at a time" (Sequential) or "all entities act and resolve together" (Simultaneous), we need:

```
STEP
│
├── PHASE 1: Parallel Observations (async)
│   └── All agents receive location-scoped observations in parallel
│
├── PHASE 2: Mode Classification
│   └── GM classifies each agent: ISOLATED or SOCIAL
│       - ISOLATED: Alone at location OR no interaction triggered
│       - SOCIAL: 2+ agents at same location AND interaction triggered
│
├── PHASE 3: Parallel Actions (async)
│   ├── ISOLATED agents: Get action choices, decide, resolve
│   └── SOCIAL agents: Run dialogue loops in PARALLEL (multiple dialogues concurrent)
│
└── PHASE 4: State Update (locked)
    └── Apply all movements atomically with location lock
```

## File Structure

```
concordia/
└── environment/
    └── engines/
        └── async_location.py    # New engine (contribute upstream)

sim/
├── engines/
│   └── async_location.py        # Local copy for development
├── components/
│   ├── location_state.py        # Thread-safe location tracking
│   ├── mode_classifier.py       # Isolated vs Social classification
│   └── dialogic_gm.py           # Dialogue GM for social mode
└── prefabs/
    └── async_island_gm.py       # GM prefab using new components
```

## Component 1: LocationState (GM Component)

Thread-safe location tracking with atomic updates.

```python
# sim/components/location_state.py

import threading
from collections import defaultdict
from concordia.typing import entity_component

class LocationState(entity_component.ContextComponent):
    """
    Thread-safe location state manager.

    Tracks which agents are at which locations with proper locking
    for concurrent access during async phases.
    """

    def __init__(self, initial_locations: dict[str, str]):
        """
        Args:
            initial_locations: agent_name -> location_id mapping
        """
        self._lock = threading.RLock()
        self._agent_locations: dict[str, str] = dict(initial_locations)
        self._location_agents: dict[str, set[str]] = defaultdict(set)

        # Build reverse index
        for agent, loc in self._agent_locations.items():
            self._location_agents[loc].add(agent)

    def get_location(self, agent_name: str) -> str:
        """Get agent's current location (thread-safe read)."""
        with self._lock:
            return self._agent_locations.get(agent_name, "unknown")

    def get_agents_at(self, location: str) -> frozenset[str]:
        """Get all agents at a location (thread-safe read)."""
        with self._lock:
            return frozenset(self._location_agents.get(location, set()))

    def get_colocated_agents(self, agent_name: str) -> frozenset[str]:
        """Get other agents at same location as agent."""
        with self._lock:
            loc = self._agent_locations.get(agent_name)
            if loc is None:
                return frozenset()
            others = self._location_agents[loc] - {agent_name}
            return frozenset(others)

    def move_agent(self, agent_name: str, new_location: str) -> str:
        """
        Atomically move agent to new location.

        Returns:
            Previous location
        """
        with self._lock:
            old_location = self._agent_locations.get(agent_name)
            if old_location == new_location:
                return old_location

            # Remove from old location
            if old_location and agent_name in self._location_agents[old_location]:
                self._location_agents[old_location].remove(agent_name)

            # Add to new location
            self._agent_locations[agent_name] = new_location
            self._location_agents[new_location].add(agent_name)

            return old_location

    def batch_move(self, movements: dict[str, str]) -> dict[str, str]:
        """
        Atomically apply multiple movements.

        Args:
            movements: agent_name -> new_location mapping

        Returns:
            agent_name -> old_location mapping
        """
        with self._lock:
            old_locations = {}
            for agent, new_loc in movements.items():
                old_locations[agent] = self.move_agent(agent, new_loc)
            return old_locations

    def get_occupied_locations(self) -> dict[str, frozenset[str]]:
        """Get all locations with agents."""
        with self._lock:
            return {
                loc: frozenset(agents)
                for loc, agents in self._location_agents.items()
                if agents
            }

    def pre_act(self, action_spec) -> str:
        """Concordia component interface: provide location context."""
        with self._lock:
            lines = ["Current agent locations:"]
            for loc, agents in self._location_agents.items():
                if agents:
                    lines.append(f"  {loc}: {', '.join(sorted(agents))}")
            return '\n'.join(lines)
```

## Component 2: ModeClassifier (GM Component)

Determines which agents are in isolated vs social mode.

```python
# sim/components/mode_classifier.py

from enum import Enum
from dataclasses import dataclass
from concordia.typing import entity_component

class AgentMode(Enum):
    ISOLATED = "isolated"
    SOCIAL = "social"

@dataclass
class ModeClassification:
    """Result of mode classification for a step."""
    isolated_agents: frozenset[str]
    social_groups: list[frozenset[str]]  # Groups of agents in dialogue

class ModeClassifier(entity_component.ContextComponent):
    """
    Classifies agents into ISOLATED or SOCIAL mode based on:
    1. Location co-presence (2+ agents at same location)
    2. Interaction triggers (optional LLM check)
    """

    def __init__(
        self,
        location_state: 'LocationState',
        model: 'LanguageModel',
        min_agents_for_social: int = 2,
        max_dialogue_group_size: int = 4,
        use_llm_trigger: bool = False,
    ):
        self._location_state = location_state
        self._model = model
        self._min_agents = min_agents_for_social
        self._max_group_size = max_dialogue_group_size
        self._use_llm_trigger = use_llm_trigger

    def classify(self, active_agents: frozenset[str]) -> ModeClassification:
        """
        Classify all active agents into modes.

        Args:
            active_agents: Set of agent names participating this step

        Returns:
            ModeClassification with isolated agents and social groups
        """
        occupied = self._location_state.get_occupied_locations()

        social_groups = []
        social_agents = set()

        for loc, agents_at_loc in occupied.items():
            # Filter to active agents only
            active_at_loc = agents_at_loc & active_agents

            if len(active_at_loc) >= self._min_agents:
                # This location has enough agents for social interaction
                if self._use_llm_trigger:
                    # Optional: Ask LLM if interaction should occur
                    should_interact = self._check_interaction_trigger(loc, active_at_loc)
                    if not should_interact:
                        continue

                # Split into groups if too large
                agents_list = list(active_at_loc)
                for i in range(0, len(agents_list), self._max_group_size):
                    group = frozenset(agents_list[i:i + self._max_group_size])
                    if len(group) >= self._min_agents:
                        social_groups.append(group)
                        social_agents.update(group)

        isolated_agents = active_agents - social_agents

        return ModeClassification(
            isolated_agents=frozenset(isolated_agents),
            social_groups=social_groups,
        )

    def _check_interaction_trigger(
        self,
        location: str,
        agents: frozenset[str]
    ) -> bool:
        """Optional LLM check for whether interaction should occur."""
        prompt = f"""At {location}, these agents are present: {', '.join(agents)}.

Should a social interaction (conversation) occur between them right now?
Consider: time of day, location type, agent relationships.

Answer YES or NO."""

        response = self._model(prompt).strip().upper()
        return response.startswith("YES")
```

## Component 3: DialogicGM (Dialogue Game Master)

Runs a dialogue loop between a group of agents.

```python
# sim/components/dialogic_gm.py

from dataclasses import dataclass, field
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

@dataclass
class DialogueResult:
    """Result of a dialogue session."""
    location: str
    participants: list[str]
    exchanges: list[tuple[str, str]]  # (speaker, utterance)
    summary: str = ""

class DialogicGM(entity_component.ContextComponent):
    """
    Runs dialogue sessions between groups of agents.

    Each dialogue is a turn-based conversation where agents
    speak in sequence, building on previous utterances.
    """

    def __init__(
        self,
        model: 'LanguageModel',
        location_state: 'LocationState',
        max_turns: int = 6,
        turns_per_agent: int = 2,
    ):
        self._model = model
        self._location_state = location_state
        self._max_turns = max_turns
        self._turns_per_agent = turns_per_agent

    def run_dialogue(
        self,
        participants: list['Entity'],
        location: str,
    ) -> DialogueResult:
        """
        Run a dialogue session between participants.

        This method is THREAD-SAFE and can be called in parallel
        for multiple dialogue groups.

        Args:
            participants: List of entity agents in the dialogue
            location: Where the dialogue occurs

        Returns:
            DialogueResult with full transcript
        """
        exchanges = []
        participant_names = [p.name for p in participants]
        num_turns = min(self._max_turns, len(participants) * self._turns_per_agent)

        for turn in range(num_turns):
            speaker_idx = turn % len(participants)
            speaker = participants[speaker_idx]
            others = [p.name for p in participants if p != speaker]

            # Build context for speaker
            recent_dialogue = self._format_recent_dialogue(exchanges[-6:])

            # Get speaker's utterance via their entity act
            action_spec = entity_lib.ActionSpec(
                call_to_action=self._build_dialogue_prompt(
                    speaker.name,
                    others,
                    location,
                    recent_dialogue,
                ),
                output_type=entity_lib.OutputType.FREE,
            )

            utterance = speaker.act(action_spec)
            utterance = self._clean_utterance(utterance, speaker.name)
            exchanges.append((speaker.name, utterance))

            # Observe the utterance for all participants
            observation = f"{speaker.name} says: \"{utterance}\""
            for p in participants:
                p.observe(observation)

        # Generate summary
        summary = self._summarize_dialogue(location, participant_names, exchanges)

        return DialogueResult(
            location=location,
            participants=participant_names,
            exchanges=exchanges,
            summary=summary,
        )

    def _build_dialogue_prompt(
        self,
        speaker: str,
        others: list[str],
        location: str,
        recent_dialogue: str,
    ) -> str:
        return f"""{speaker} is at {location} talking with {', '.join(others)}.

{recent_dialogue if recent_dialogue else 'The conversation is just starting.'}

What does {speaker} say next? Respond with just the dialogue (one or two sentences)."""

    def _format_recent_dialogue(self, exchanges: list[tuple[str, str]]) -> str:
        if not exchanges:
            return ""
        lines = [f"{speaker}: \"{text}\"" for speaker, text in exchanges]
        return "Recent dialogue:\n" + "\n".join(lines)

    def _clean_utterance(self, utterance: str, speaker: str) -> str:
        """Clean up LLM response to just the dialogue text."""
        utterance = utterance.strip().strip('"\'')
        # Remove speaker prefix if present
        if utterance.lower().startswith(speaker.lower() + ":"):
            utterance = utterance[len(speaker) + 1:].strip()
        return utterance

    def _summarize_dialogue(
        self,
        location: str,
        participants: list[str],
        exchanges: list[tuple[str, str]],
    ) -> str:
        """Generate a brief summary of the dialogue."""
        dialogue_text = "\n".join(f"{s}: {t}" for s, t in exchanges)
        prompt = f"""Summarize this conversation at {location} between {', '.join(participants)} in one sentence:

{dialogue_text}

Summary:"""
        return self._model(prompt).strip()
```

## Component 4: AsyncLocationEngine

The main engine that orchestrates everything.

```python
# sim/engines/async_location.py

import functools
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
from concordia.utils import concurrency

from ..components.location_state import LocationState
from ..components.mode_classifier import ModeClassifier, ModeClassification
from ..components.dialogic_gm import DialogicGM, DialogueResult

class AsyncLocationEngine(engine_lib.Engine):
    """
    Async engine with location-scoped parallel execution.

    Phases:
    1. Parallel observations (all agents observe their location)
    2. Mode classification (isolated vs social)
    3. Parallel actions:
       - Isolated agents: action choice + resolution
       - Social agents: parallel dialogues
    4. Locked state update (movements applied atomically)
    """

    def __init__(
        self,
        location_state: LocationState,
        mode_classifier: ModeClassifier,
        dialogic_gm: DialogicGM,
        call_to_make_observation: str = "What does {name} observe at their current location?",
        call_to_isolated_action: str = "What does {name} do next?",
        call_to_resolve: str = "What happens as a result?",
        call_to_check_termination: str = "Is the simulation finished?",
        isolated_action_choices: list[str] | None = None,
    ):
        self._location_state = location_state
        self._mode_classifier = mode_classifier
        self._dialogic_gm = dialogic_gm

        self._call_to_make_observation = call_to_make_observation
        self._call_to_isolated_action = call_to_isolated_action
        self._call_to_resolve = call_to_resolve
        self._call_to_check_termination = call_to_check_termination

        # Default isolated action choices
        self._isolated_choices = isolated_action_choices or [
            "explore the area",
            "rest and reflect",
            "move to a nearby location",
            "wait and observe",
        ]

    def run_loop(
        self,
        game_masters: Sequence[entity_lib.Entity],
        entities: Sequence[entity_lib.Entity],
        premise: str = '',
        max_steps: int = 100,
        verbose: bool = False,
        log: list[Mapping[str, Any]] | None = None,
        checkpoint_callback: Callable[[int], None] | None = None,
    ):
        """Run the async location-based game loop."""
        if not game_masters:
            raise ValueError("No game masters provided.")

        game_master = game_masters[0]
        entities_by_name = {e.name: e for e in entities}

        if premise:
            game_master.observe(premise)
            for entity in entities:
                entity.observe(premise)

        for step in range(max_steps):
            if self.terminate(game_master, verbose):
                break

            if verbose:
                print(f"\n{'='*60}")
                print(f"STEP {step + 1}")
                print(f"{'='*60}")

            # === PHASE 1: Parallel Observations ===
            if verbose:
                print("\nPhase 1: Parallel Observations")

            observations = self._parallel_observations(
                game_master, entities, verbose
            )

            # Deliver observations to agents
            for name, obs in observations.items():
                entities_by_name[name].observe(obs)

            # === PHASE 2: Mode Classification ===
            if verbose:
                print("\nPhase 2: Mode Classification")

            active_agents = frozenset(e.name for e in entities)
            classification = self._mode_classifier.classify(active_agents)

            if verbose:
                print(f"  Isolated: {classification.isolated_agents}")
                print(f"  Social groups: {classification.social_groups}")

            # === PHASE 3: Parallel Actions ===
            if verbose:
                print("\nPhase 3: Parallel Actions")

            # Run isolated actions and dialogues in parallel
            results = self._parallel_actions(
                game_master,
                entities_by_name,
                classification,
                verbose,
            )

            # === PHASE 4: State Update (locked) ===
            if verbose:
                print("\nPhase 4: State Update")

            self._apply_state_updates(results, verbose)

            if checkpoint_callback:
                checkpoint_callback(step)

    def _parallel_observations(
        self,
        game_master: entity_lib.Entity,
        entities: Sequence[entity_lib.Entity],
        verbose: bool,
    ) -> dict[str, str]:
        """Generate observations for all agents in parallel."""

        def make_obs(entity: entity_lib.Entity) -> str:
            return self.make_observation(game_master, entity)

        tasks = {
            entity.name: functools.partial(make_obs, entity)
            for entity in entities
        }

        return concurrency.run_tasks(tasks)

    def _parallel_actions(
        self,
        game_master: entity_lib.Entity,
        entities_by_name: dict[str, entity_lib.Entity],
        classification: ModeClassification,
        verbose: bool,
    ) -> dict[str, Any]:
        """
        Run all actions in parallel:
        - Isolated agents do solo actions
        - Social groups run dialogues
        """
        tasks = {}

        # Tasks for isolated agents
        for agent_name in classification.isolated_agents:
            entity = entities_by_name[agent_name]
            tasks[f"isolated_{agent_name}"] = functools.partial(
                self._isolated_agent_action,
                game_master,
                entity,
                verbose,
            )

        # Tasks for social groups (one dialogue per group)
        for i, group in enumerate(classification.social_groups):
            participants = [entities_by_name[name] for name in group]
            location = self._location_state.get_location(list(group)[0])
            tasks[f"dialogue_{i}"] = functools.partial(
                self._run_dialogue_group,
                participants,
                location,
                verbose,
            )

        return concurrency.run_tasks(tasks)

    def _isolated_agent_action(
        self,
        game_master: entity_lib.Entity,
        entity: entity_lib.Entity,
        verbose: bool,
    ) -> dict[str, Any]:
        """Handle an isolated agent's action."""
        location = self._location_state.get_location(entity.name)
        neighbors = self._get_neighbor_locations(location)

        # Build action choices including movement options
        choices = list(self._isolated_choices)
        for neighbor in neighbors[:3]:  # Limit movement options
            choices.append(f"move to {neighbor}")

        # Get agent's choice
        action_spec = entity_lib.ActionSpec(
            call_to_action=self._call_to_isolated_action.format(name=entity.name),
            output_type=entity_lib.OutputType.CHOICE,
            options=tuple(choices),
        )

        action = entity.act(action_spec)

        # Parse for movement
        movement = None
        if action.startswith("move to "):
            dest = action[8:].strip()
            if dest in neighbors:
                movement = dest

        if verbose:
            print(f"  {entity.name} (isolated): {action}")

        return {
            "agent": entity.name,
            "action": action,
            "movement": movement,
        }

    def _run_dialogue_group(
        self,
        participants: list[entity_lib.Entity],
        location: str,
        verbose: bool,
    ) -> DialogueResult:
        """Run a dialogue for a social group."""
        if verbose:
            names = [p.name for p in participants]
            print(f"  Dialogue at {location}: {names}")

        result = self._dialogic_gm.run_dialogue(participants, location)

        if verbose:
            for speaker, text in result.exchanges[:2]:
                print(f"    {speaker}: \"{text[:50]}...\"")

        return result

    def _apply_state_updates(
        self,
        results: dict[str, Any],
        verbose: bool,
    ) -> None:
        """Apply all state updates with location lock."""
        movements = {}

        for key, result in results.items():
            if key.startswith("isolated_") and isinstance(result, dict):
                if result.get("movement"):
                    movements[result["agent"]] = result["movement"]

        if movements:
            old_locs = self._location_state.batch_move(movements)
            if verbose:
                for agent, new_loc in movements.items():
                    print(f"  {agent}: {old_locs[agent]} -> {new_loc}")

    def _get_neighbor_locations(self, location: str) -> list[str]:
        """Get neighboring locations (override for specific world graph)."""
        # Default: return empty, override in subclass or inject
        return []

    # === Required Engine interface methods ===

    def make_observation(
        self,
        game_master: entity_lib.Entity,
        entity: entity_lib.Entity,
    ) -> str:
        """Make an observation for an entity."""
        return game_master.act(
            action_spec=entity_lib.ActionSpec(
                call_to_action=self._call_to_make_observation.format(name=entity.name),
                output_type=entity_lib.OutputType.MAKE_OBSERVATION,
            )
        )

    def next_acting(
        self,
        game_master: entity_lib.Entity,
        entities: Sequence[entity_lib.Entity],
    ) -> tuple[entity_lib.Entity, entity_lib.ActionSpec]:
        """Not used in this engine - we handle all entities in parallel."""
        raise NotImplementedError("AsyncLocationEngine handles all entities in parallel")

    def resolve(
        self,
        game_master: entity_lib.Entity,
        event: str,
    ) -> None:
        """Resolve an event through the game master."""
        game_master.observe(event)
        game_master.act(
            action_spec=entity_lib.ActionSpec(
                call_to_action=self._call_to_resolve,
                output_type=entity_lib.OutputType.RESOLVE,
            )
        )

    def terminate(
        self,
        game_master: entity_lib.Entity,
        verbose: bool = False,
    ) -> bool:
        """Check if simulation should terminate."""
        result = game_master.act(
            action_spec=entity_lib.ActionSpec(
                call_to_action=self._call_to_check_termination,
                output_type=entity_lib.OutputType.TERMINATE,
                options=tuple(entity_lib.BINARY_OPTIONS.values()),
            )
        )
        return result == entity_lib.BINARY_OPTIONS['affirmative']

    def next_game_master(
        self,
        game_master: entity_lib.Entity,
        game_masters: Sequence[entity_lib.Entity],
    ) -> entity_lib.Entity:
        """Select next game master (default to first)."""
        return game_masters[0]
```

## GM Prefab: AsyncIslandGM

Assembles all components into a proper Concordia GM.

```python
# sim/prefabs/async_island_gm.py

import dataclasses
from concordia.prefabs import prefab as prefab_lib
from concordia.agents import entity_agent_with_logging

from ..components.location_state import LocationState
from ..components.mode_classifier import ModeClassifier
from ..components.dialogic_gm import DialogicGM

@dataclasses.dataclass
class AsyncIslandGM(prefab_lib.Prefab):
    """
    GM prefab for async location-based simulation.

    Includes:
    - LocationState: Thread-safe agent location tracking
    - ModeClassifier: Isolated vs Social mode assignment
    - DialogicGM: Parallel dialogue runner
    - Standard GM components (memory, observation, etc.)
    """

    description: str = "Async island game master with location-scoped execution"

    def build(
        self,
        model,
        memory_bank,
        initial_locations: dict[str, str],
        world_graph: 'Island',  # For neighbor lookups
        **kwargs,
    ):
        # Build location state
        location_state = LocationState(initial_locations)

        # Build mode classifier
        mode_classifier = ModeClassifier(
            location_state=location_state,
            model=model,
            min_agents_for_social=2,
            max_dialogue_group_size=4,
        )

        # Build dialogic GM
        dialogic_gm = DialogicGM(
            model=model,
            location_state=location_state,
            max_turns=6,
        )

        # Assemble standard GM components
        components = self._build_gm_components(
            model=model,
            memory_bank=memory_bank,
            location_state=location_state,
        )

        # Build entity with all components
        gm = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name="Island GM",
            model=model,
            components=components,
            component_order=list(components.keys()),
        )

        # Attach extra components as attributes for engine access
        gm.location_state = location_state
        gm.mode_classifier = mode_classifier
        gm.dialogic_gm = dialogic_gm
        gm.world_graph = world_graph

        return gm

    def _build_gm_components(self, model, memory_bank, location_state):
        """Build standard GM components."""
        from concordia.components import agent as agent_components
        from concordia.components.game_master import make_observation

        components = {}

        # Instructions
        components['instructions'] = agent_components.constant.Constant(
            state="You are the game master for an island simulation...",
            pre_act_label="Instructions",
        )

        # Location state (provides context about who is where)
        components['location_state'] = location_state

        # Memory
        components['memory'] = agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank,
            num_memories_to_retrieve=25,
        )

        # Observation maker
        components['make_observation'] = make_observation.MakeObservation(
            model=model,
            components=components,
        )

        return components
```

## Integration: run_async.py

```python
# run_async.py

from sim.engines.async_location import AsyncLocationEngine
from sim.prefabs.async_island_gm import AsyncIslandGM
from sim.prefabs_entity import IslandEntity
from sim.world import Island
from sim.agents import DEFAULT_ISLAND_AGENTS

def main():
    # Setup
    island = Island.load("data/island.yaml")
    model = create_model()
    embedder = create_embedder()

    # Build entities
    entities = []
    initial_locations = {}
    for config in DEFAULT_ISLAND_AGENTS[:10]:
        entity = IslandEntity().build(model, create_memory_bank(), config)
        entities.append(entity)
        initial_locations[config.name] = config.home_place

    # Build GM with async components
    gm = AsyncIslandGM().build(
        model=model,
        memory_bank=create_memory_bank(),
        initial_locations=initial_locations,
        world_graph=island,
    )

    # Build engine
    engine = AsyncLocationEngine(
        location_state=gm.location_state,
        mode_classifier=gm.mode_classifier,
        dialogic_gm=gm.dialogic_gm,
    )

    # Inject world graph for neighbor lookups
    engine._get_neighbor_locations = lambda loc: island.neighbors(loc)

    # Run
    engine.run_loop(
        game_masters=[gm],
        entities=entities,
        max_steps=48,
        verbose=True,
    )
```

## Key Design Decisions

### 1. Thread-Safe Location State
- Uses `threading.RLock()` for all location reads/writes
- `batch_move()` applies multiple movements atomically
- Prevents race conditions during parallel action phase

### 2. Mode Classification
- Happens AFTER observations, BEFORE actions
- Simple heuristic: 2+ agents at location = SOCIAL
- Optional LLM trigger for more nuanced classification

### 3. Parallel Dialogues
- Each dialogue group runs independently
- Uses `concurrency.run_tasks()` from Concordia
- Dialogic GM is thread-safe (no shared mutable state)

### 4. Time Synchronization
- All agents observe at same "time"
- All actions resolve before next step
- Movements applied atomically at end of step

### 5. Isolated vs Social Mode
- **Isolated**: Agent gets observation + action choices (including move)
- **Social**: Agent participates in dialogue, no solo action

## LLM Call Analysis

For 10 agents, 6 locations, per step:

| Phase | Calls | Notes |
|-------|-------|-------|
| Observations | 10 | One per agent (could optimize to 1 per location) |
| Mode Classification | 0-3 | Optional LLM trigger checks |
| Isolated Actions | 4-8 | Depends on classification |
| Dialogues | 6-12 | 2-4 groups × 3 turns each |
| **Total** | ~20-30 | vs ~50 with naive approach |

## Next Steps

1. Implement and test LocationState component
2. Implement and test ModeClassifier component
3. Implement and test DialogicGM component
4. Implement AsyncLocationEngine
5. Create AsyncIslandGM prefab
6. Integration test with Island simulation
7. Benchmark against Sequential engine
8. Optimize observation phase (share per location)
9. Consider contributing engine upstream to Concordia
