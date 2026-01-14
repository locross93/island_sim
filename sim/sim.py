"""
Main simulation loop for the Island Time-and-Place Concordia Simulation.

This module ties together the world, agents, and game master into a
runnable simulation with save/load capabilities.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from .agents import (
    AgentConfig,
    IslandAgent,
    IslandAgentState,
    Needs,
    ScheduleBias,
    DEFAULT_ISLAND_AGENTS,
)
from .gm import GameMaster, TickResult, EncounterRecord
from .world import Island, tick_to_day_and_time, TICKS_PER_DAY


class SimulationState(BaseModel):
    """Serializable simulation state for save/load."""
    current_tick: int
    agents: list[IslandAgentState]
    encounter_history: list[EncounterRecord]
    created_at: str
    last_saved: str


class SimulationConfig(BaseModel):
    """Configuration for running the simulation."""
    island_yaml_path: str = "data/island.yaml"
    use_llm: bool = True
    llm_provider: str = "openai"  # "openai" or "anthropic"
    llm_model: str = "gpt-4.1-nano"  # Default to GPT-4.1 nano
    ticks_per_day: int = TICKS_PER_DAY
    start_tick: int = 16  # 8:00 AM
    max_ticks: int | None = None  # None = run forever
    auto_save_interval: int = 48  # Save every day


@dataclass
class IslandSimulation:
    """
    The main simulation controller.

    Manages the world, agents, and game master, and provides
    methods for running, saving, and loading the simulation.
    """
    config: SimulationConfig
    island: Island = field(init=False)
    agents: list[IslandAgent] = field(default_factory=list)
    gm: GameMaster = field(init=False)
    model: Any = None
    embedder: Any = None
    _tick_callbacks: list[Callable[[TickResult], None]] = field(default_factory=list)
    _running: bool = False
    _created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # Load the island
        self.island = Island.load(Path(self.config.island_yaml_path))

        # Initialize LLM if enabled
        if self.config.use_llm:
            self._init_llm()

    def _init_llm(self) -> None:
        """Initialize LLM model and embedder for Concordia."""
        if self.config.llm_provider == "openai":
            try:
                from openai import OpenAI
                self.model = OpenAI()
                print(f"Initialized OpenAI client (model: {self.config.llm_model})")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
                self.model = None
        else:
            try:
                from anthropic import Anthropic
                self.model = Anthropic()
                print(f"Initialized Anthropic client (model: {self.config.llm_model})")
            except Exception as e:
                print(f"Warning: Could not initialize Anthropic client: {e}")
                self.model = None

        # Create a simple embedder using sentence-transformers if available
        try:
            from sentence_transformers import SentenceTransformer
            _embedder_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedder = lambda text: _embedder_model.encode(text)
        except ImportError:
            # Fallback: create a dummy embedder
            import hashlib
            def dummy_embedder(text: str) -> list[float]:
                # Create a deterministic but non-semantic embedding
                h = hashlib.sha256(text.encode()).hexdigest()
                return [int(h[i:i+2], 16) / 255.0 for i in range(0, 64, 2)]
            self.embedder = dummy_embedder
            print("Note: Using dummy embedder. Install sentence-transformers for semantic memory.")

    def initialize_agents(
        self,
        configs: list[AgentConfig] | None = None,
    ) -> None:
        """
        Initialize agents for the simulation.

        Args:
            configs: Agent configurations. Defaults to the 10 pre-defined agents.
        """
        configs = configs or DEFAULT_ISLAND_AGENTS

        # Create mock model wrapper for Concordia if needed
        model_wrapper = self._create_model_wrapper()

        self.agents = []
        for config in configs:
            # Create personal memory bank for each agent
            try:
                from concordia.associative_memory import basic_associative_memory
                memory_bank = basic_associative_memory.AssociativeMemoryBank(
                    sentence_embedder=self.embedder,
                )
            except Exception:
                memory_bank = None

            if memory_bank and model_wrapper:
                agent = IslandAgent(
                    config=config,
                    model=model_wrapper,
                    memory_bank=memory_bank,
                )
            else:
                # Create simplified agent without full Concordia
                agent = self._create_simple_agent(config)

            self.agents.append(agent)

        # Initialize the game master
        self.gm = GameMaster(
            island=self.island,
            agents=self.agents,
            model=self.model,
            current_tick=self.config.start_tick,
        )

        print(f"Initialized {len(self.agents)} agents on the island")

    def _create_model_wrapper(self) -> Any:
        """Create a model wrapper compatible with Concordia components."""
        if not self.model:
            return None

        if self.config.llm_provider == "openai":
            class OpenAIModelWrapper:
                """Wrapper to make OpenAI client compatible with Concordia."""
                def __init__(self, client, model_name: str):
                    self.client = client
                    self.model_name = model_name

                def __call__(self, prompt: str) -> str:
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            max_tokens=300,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        return response.choices[0].message.content
                    except Exception as e:
                        return f"[Error: {e}]"

            return OpenAIModelWrapper(self.model, self.config.llm_model)
        else:
            class AnthropicModelWrapper:
                """Wrapper to make Anthropic client compatible with Concordia."""
                def __init__(self, client, model_name: str):
                    self.client = client
                    self.model_name = model_name

                def __call__(self, prompt: str) -> str:
                    try:
                        response = self.client.messages.create(
                            model=self.model_name,
                            max_tokens=300,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        return response.content[0].text
                    except Exception as e:
                        return f"[Error: {e}]"

            return AnthropicModelWrapper(self.model, self.config.llm_model)

    def _create_simple_agent(self, config: AgentConfig) -> IslandAgent:
        """Create a simplified agent without full Concordia integration."""
        # Create a minimal agent state
        class SimpleAgent:
            def __init__(self, cfg: AgentConfig):
                self.name = cfg.name
                self.home_place = cfg.home_place
                self.work_place = cfg.work_place
                self.current_location = cfg.home_place
                self.in_transit_to = None
                self.in_transit_ticks = 0
                self.needs = Needs()
                self.relationships = dict(cfg.initial_relationships)
                self.schedule_bias = cfg.schedule_bias
                self.personality = cfg.personality
                self.backstory = cfg.backstory

            def observe(self, observation: str) -> None:
                pass  # Simple agents don't process observations

            def generate_intent(self, tick: int, context: str):
                # Will use GM's simple intent generator
                from .agents import Intent, IntentType
                return Intent(type=IntentType.STAY, reason="Waiting")

            def tick_update(self) -> None:
                self.needs.tick_decay()
                if self.in_transit_ticks > 0:
                    self.in_transit_ticks -= 1
                    if self.in_transit_ticks == 0 and self.in_transit_to:
                        self.current_location = self.in_transit_to
                        self.in_transit_to = None

            def start_travel(self, destination: str, travel_ticks: int) -> None:
                self.in_transit_to = destination
                self.in_transit_ticks = travel_ticks

            def update_relationship(self, other_name: str, delta: float) -> None:
                current = self.relationships.get(other_name, 0.0)
                self.relationships[other_name] = max(-100.0, min(100.0, current + delta))

            def is_available(self) -> bool:
                return self.in_transit_ticks == 0

            @property
            def location_display(self) -> str:
                if self.in_transit_to:
                    return f"traveling to {self.in_transit_to}"
                return self.current_location

            def get_state(self) -> IslandAgentState:
                return IslandAgentState(
                    name=self.name,
                    home_place=self.home_place,
                    work_place=self.work_place,
                    current_location=self.current_location,
                    in_transit_to=self.in_transit_to,
                    in_transit_ticks=self.in_transit_ticks,
                    needs=self.needs,
                    relationships=self.relationships,
                    schedule_bias=self.schedule_bias,
                    personality=self.personality,
                    backstory=self.backstory,
                )

            def get_last_log(self) -> dict:
                """Simple agents don't have Concordia component logs."""
                return {}

        return SimpleAgent(config)

    def on_tick(self, callback: Callable[[TickResult], None]) -> None:
        """Register a callback to be called after each tick."""
        self._tick_callbacks.append(callback)

    def step(self) -> TickResult:
        """Run a single simulation tick (synchronous)."""
        result = self.gm.run_tick_sync()

        # Auto-save check
        if self.config.auto_save_interval:
            if result.tick % self.config.auto_save_interval == 0:
                self.save(Path("data/autosave.json"))

        # Notify callbacks
        for callback in self._tick_callbacks:
            try:
                callback(result)
            except Exception as e:
                print(f"Callback error: {e}")

        return result

    async def step_async(self) -> TickResult:
        """Run a single simulation tick (async with LLM)."""
        result = await self.gm.run_tick()

        # Auto-save check
        if self.config.auto_save_interval:
            if result.tick % self.config.auto_save_interval == 0:
                self.save(Path("data/autosave.json"))

        # Notify callbacks
        for callback in self._tick_callbacks:
            try:
                callback(result)
            except Exception as e:
                print(f"Callback error: {e}")

        return result

    def run(self, num_ticks: int | None = None, print_status: bool = True) -> list[TickResult]:
        """
        Run the simulation for a number of ticks (synchronous).

        Args:
            num_ticks: Number of ticks to run. If None, uses config.max_ticks.
            print_status: Whether to print status each tick.

        Returns:
            List of TickResults for each tick.
        """
        ticks = num_ticks or self.config.max_ticks or 48  # Default to 1 day
        results = []

        self._running = True
        try:
            for _ in range(ticks):
                if not self._running:
                    break

                result = self.step()
                results.append(result)

                if print_status:
                    self.gm.print_status()

        except KeyboardInterrupt:
            print("\nSimulation interrupted")
            self._running = False

        return results

    async def run_async(
        self,
        num_ticks: int | None = None,
        print_status: bool = True,
    ) -> list[TickResult]:
        """
        Run the simulation for a number of ticks (async with LLM).

        Args:
            num_ticks: Number of ticks to run.
            print_status: Whether to print status each tick.

        Returns:
            List of TickResults for each tick.
        """
        ticks = num_ticks or self.config.max_ticks or 48
        results = []

        self._running = True
        try:
            for _ in range(ticks):
                if not self._running:
                    break

                result = await self.step_async()
                results.append(result)

                if print_status:
                    self.gm.print_status()

        except KeyboardInterrupt:
            print("\nSimulation interrupted")
            self._running = False

        return results

    def stop(self) -> None:
        """Stop the running simulation."""
        self._running = False

    def save(self, path: Path | str) -> None:
        """Save simulation state to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Collect agent states
        agent_states = []
        for agent in self.agents:
            state = agent.get_state()
            agent_states.append(state)

        sim_state = SimulationState(
            current_tick=self.gm.current_tick,
            agents=agent_states,
            encounter_history=self.gm.encounter_history,
            created_at=self._created_at.isoformat(),
            last_saved=datetime.now().isoformat(),
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(sim_state.model_dump_json(indent=2))

        print(f"Saved simulation to {path}")

    def load(self, path: Path | str) -> None:
        """Load simulation state from a JSON file."""
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sim_state = SimulationState(**data)

        # Restore tick
        self.gm.current_tick = sim_state.current_tick

        # Restore agent states
        for saved_state in sim_state.agents:
            agent = self.gm.get_agent(saved_state.name)
            if agent:
                agent.current_location = saved_state.current_location
                agent.in_transit_to = saved_state.in_transit_to
                agent.in_transit_ticks = saved_state.in_transit_ticks
                agent.needs = saved_state.needs
                agent.relationships = dict(saved_state.relationships)

        # Restore encounter history
        self.gm.encounter_history = list(sim_state.encounter_history)

        self._created_at = datetime.fromisoformat(sim_state.created_at)

        print(f"Loaded simulation from {path} (tick {sim_state.current_tick})")

    def get_agent_info(self, name: str) -> dict | None:
        """Get detailed info about a specific agent."""
        agent = self.gm.get_agent(name)
        if not agent:
            return None

        state = agent.get_state()
        place = self.island.get_place(agent.current_location)

        return {
            "name": agent.name,
            "location": place.name if place else agent.current_location,
            "location_id": agent.current_location,
            "in_transit": agent.in_transit_to,
            "needs": {
                "hunger": round(agent.needs.hunger, 1),
                "fatigue": round(agent.needs.fatigue, 1),
                "social": round(agent.needs.social, 1),
            },
            "relationships": agent.relationships,
            "personality": agent.personality,
            "home": agent.home_place,
            "work": agent.work_place,
        }

    def get_all_agents_summary(self) -> list[dict]:
        """Get a summary of all agents."""
        return [
            {
                "name": a.name,
                "location": a.location_display,
                "needs": a.needs.most_urgent(),
            }
            for a in self.agents
        ]

    def get_place_info(self, place_id: str) -> dict | None:
        """Get info about a place including who's there."""
        place = self.island.get_place(place_id)
        if not place:
            return None

        agents_here = self.gm.agents_at_place(place_id)

        return {
            "id": place_id,
            "name": place.name,
            "type": place.type,
            "open_hours": f"{place.open[0]}:00 - {place.open[1]}:00",
            "is_open": place.is_open(self.gm.current_tick),
            "agents": [a.name for a in agents_here],
            "neighbors": self.island.neighbors(place_id),
        }

    def get_time_info(self) -> dict:
        """Get current simulation time info."""
        day, time_str = tick_to_day_and_time(self.gm.current_tick)
        return {
            "tick": self.gm.current_tick,
            "day": day,
            "time": time_str,
            "ticks_per_day": TICKS_PER_DAY,
        }

    def get_recent_events(self, limit: int = 10) -> list[dict]:
        """Get recent encounters and events."""
        events = []
        for enc in self.gm.encounter_history[-limit:]:
            place = self.island.get_place(enc.place_id)
            events.append({
                "tick": enc.tick,
                "type": "encounter",
                "place": place.name if place else enc.place_id,
                "participants": enc.participants,
                "summary": enc.summary,
                "dialogue": enc.dialogue,  # Include dialogue
            })
        return events

    def get_decision_log(self, limit: int = 20) -> list[dict]:
        """Get recent agent decisions from tick history."""
        decisions = []
        for result in self.gm.history[-limit:]:
            for agent_name, observations in result.observations.items():
                # Extract decision info from observations
                decision_text = observations[-1] if observations else "No action"
                decisions.append({
                    "tick": result.tick,
                    "agent": agent_name,
                    "decision": decision_text,
                    "time": result.time_str,
                })
            # Add movement decisions
            for movement in result.movements:
                decisions.append({
                    "tick": result.tick,
                    "agent": movement.agent_name,
                    "decision": f"GO_TO {movement.to_place}: {movement.reason}",
                    "time": result.time_str,
                    "is_movement": True,
                })
        return decisions[-limit:]

    def get_llm_status(self) -> dict:
        """Get LLM configuration status."""
        return {
            "enabled": self.config.use_llm,
            "provider": self.config.llm_provider,
            "model": self.config.llm_model,
            "connected": self.model is not None,
        }

    def get_llm_logs(self, limit: int = 10) -> list[dict]:
        """Get recent LLM call logs."""
        logs = []
        for entry in self.gm.llm_logs[-limit:]:
            logs.append({
                "tick": entry.tick,
                "type": entry.type,
                "prompt": entry.prompt,
                "response": entry.response,
                "participants": entry.participants,
            })
        return logs

    def get_concordia_logs(self, limit: int = 20) -> list[dict]:
        """
        Get recent Concordia component logs showing agent reasoning.

        Returns logs from the EntityAgentWithLogging.get_last_log() calls,
        which contain the output of each Concordia component (memory, observation,
        act component, etc.) during agent decision-making.
        """
        logs = []
        for entry in self.gm.concordia_logs[-limit:]:
            logs.append({
                "tick": entry.tick,
                "agent": entry.agent_name,
                "intent_type": entry.intent_type,
                "intent_reason": entry.intent_reason,
                "components": entry.component_logs,
            })
        return logs


def create_simulation(
    island_path: str = "data/island.yaml",
    use_llm: bool = False,
    start_tick: int = 16,  # 8:00 AM
    llm_provider: str = "openai",
    llm_model: str = "gpt-4.1-nano",
) -> IslandSimulation:
    """
    Factory function to create and initialize a simulation.

    Args:
        island_path: Path to island YAML file.
        use_llm: Whether to use LLM for agent decisions.
        start_tick: Starting tick (16 = 8:00 AM).
        llm_provider: "openai" or "anthropic".
        llm_model: Model name to use.

    Returns:
        Initialized IslandSimulation ready to run.
    """
    config = SimulationConfig(
        island_yaml_path=island_path,
        use_llm=use_llm,
        start_tick=start_tick,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    sim = IslandSimulation(config=config)
    sim.initialize_agents()

    return sim


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Island Simulation")
    parser.add_argument("--ticks", type=int, default=24, help="Number of ticks to run")
    parser.add_argument("--llm", default=True, action="store_true", help="Use LLM for agent decisions")
    parser.add_argument("--start", type=int, default=16, help="Starting tick (0-47)")
    parser.add_argument("--save", type=str, help="Save state to file after running")
    parser.add_argument("--load", type=str, help="Load state from file before running")

    args = parser.parse_args()

    print("=" * 50)
    print("  Island Time-and-Place Concordia Simulation")
    print("=" * 50)

    sim = create_simulation(use_llm=args.llm, start_tick=args.start)

    if args.load:
        sim.load(args.load)

    print(f"\nRunning for {args.ticks} ticks...")
    print(f"LLM mode: {'enabled' if args.llm else 'disabled (heuristic)'}")
    print()

    sim.run(num_ticks=args.ticks, print_status=True)

    if args.save:
        sim.save(args.save)

    print("\n" + "=" * 50)
    print("Simulation complete!")
    print(f"Final time: Day {sim.get_time_info()['day']}, {sim.get_time_info()['time']}")
    print(f"Total encounters: {len(sim.gm.encounter_history)}")
