"""
Island agents built on Concordia's EntityAgent framework.
Each agent has location, needs, relationships, and produces structured intents.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel

# Concordia imports
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import memory as memory_component
from concordia.components.agent import observation as observation_component
from concordia.components.agent import concat_act_component
from concordia.typing import entity as entity_lib


class IntentType(str, Enum):
    """Actions an agent can intend to take."""
    STAY = "stay"           # Remain at current location
    GO_TO = "go_to"         # Travel to a specific place
    MEET = "meet"           # Seek out a specific person
    GO_HOME = "go_home"     # Return to home location
    GO_WORK = "go_work"     # Go to work location
    WANDER = "wander"       # Go somewhere, GM picks


class Intent(BaseModel):
    """Structured intent produced by an agent each tick."""
    type: IntentType
    target_place: str | None = None   # For GO_TO
    target_person: str | None = None  # For MEET
    reason: str = ""                  # Natural language explanation


class Needs(BaseModel):
    """Agent needs that drive behavior (0-100 scale)."""
    hunger: float = 50.0
    fatigue: float = 30.0
    social: float = 50.0

    def tick_decay(self) -> None:
        """Needs naturally increase each tick."""
        self.hunger = min(100.0, self.hunger + 1.0)
        self.fatigue = min(100.0, self.fatigue + 0.5)
        self.social = min(100.0, self.social + 0.3)

    def eat(self) -> None:
        self.hunger = max(0.0, self.hunger - 40.0)

    def rest(self) -> None:
        self.fatigue = max(0.0, self.fatigue - 30.0)

    def socialize(self, amount: float = 20.0) -> None:
        self.social = max(0.0, self.social - amount)

    def most_urgent(self) -> str:
        """Return the most pressing need."""
        needs = {"hunger": self.hunger, "fatigue": self.fatigue, "social": self.social}
        return max(needs, key=needs.get)


class ScheduleBias(str, Enum):
    """Agent schedule preferences affecting behavior."""
    EARLY_BIRD = "early_bird"     # Active 5am-9pm
    NIGHT_OWL = "night_owl"       # Active 10am-2am
    NINE_TO_FIVE = "nine_to_five" # Active 8am-10pm
    FLEXIBLE = "flexible"         # No strong preference


@dataclass
class AgentConfig:
    """Configuration for creating an island agent."""
    name: str
    home_place: str
    work_place: str | None = None
    personality: str = ""
    backstory: str = ""
    schedule_bias: ScheduleBias = ScheduleBias.FLEXIBLE
    initial_relationships: dict[str, float] = field(default_factory=dict)


class IslandAgentState(BaseModel):
    """Serializable state of an island agent."""
    name: str
    home_place: str
    work_place: str | None
    current_location: str
    in_transit_to: str | None = None
    in_transit_ticks: int = 0
    needs: Needs = Needs()
    relationships: dict[str, float] = {}  # name -> affinity (-100 to 100)
    schedule_bias: ScheduleBias = ScheduleBias.FLEXIBLE
    personality: str = ""
    backstory: str = ""


class IslandAgent:
    """
    An island resident powered by Concordia's EntityAgent.

    Wraps Concordia components with island-specific state (location, needs)
    and produces structured intents for the Game Master.
    """

    def __init__(
        self,
        config: AgentConfig,
        model: Any,  # LLM model for Concordia
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ):
        self.name = config.name
        self.home_place = config.home_place
        self.work_place = config.work_place
        self.current_location = config.home_place  # Start at home
        self.in_transit_to: str | None = None
        self.in_transit_ticks: int = 0
        self.needs = Needs()
        self.relationships = dict(config.initial_relationships)
        self.schedule_bias = config.schedule_bias
        self.personality = config.personality
        self.backstory = config.backstory

        # Build Concordia agent with components
        self._memory_bank = memory_bank
        self._model = model
        self._entity = self._build_concordia_agent(config, model, memory_bank)

        # Seed initial memories
        self._seed_memories(config)

    def _build_concordia_agent(
        self,
        config: AgentConfig,
        model: Any,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the underlying Concordia EntityAgent."""

        # Memory component for associative recall
        agent_memory = memory_component.AssociativeMemory(
            memory_bank=memory_bank,
        )

        # Observation component to track recent events
        agent_observation = observation_component.LastNObservations(
            history_length=20,
        )

        # Collect components
        components = {
            memory_component.DEFAULT_MEMORY_COMPONENT_KEY: agent_memory,
            observation_component.DEFAULT_OBSERVATION_COMPONENT_KEY: agent_observation,
        }

        # Action component that concatenates context for LLM
        act_component = concat_act_component.ConcatActComponent(
            model=model,
            component_order=list(components.keys()),
        )

        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=config.name,
            act_component=act_component,
            context_components=components,
        )

    def _seed_memories(self, config: AgentConfig) -> None:
        """Add initial memories from backstory."""
        if config.backstory:
            self._memory_bank.add(f"{self.name}'s background: {config.backstory}")
        if config.personality:
            self._memory_bank.add(f"{self.name} is {config.personality}")
        self._memory_bank.add(f"{self.name} lives at {self.home_place}.")
        if config.work_place:
            self._memory_bank.add(f"{self.name} works at {config.work_place}.")

    def observe(self, observation: str) -> None:
        """Process an observation from the world."""
        self._entity.observe(observation)
        # Also add significant observations to long-term memory
        if any(word in observation.lower() for word in ["met", "talked", "saw", "heard", "learned"]):
            self._memory_bank.add(f"{self.name} observed: {observation}")

    def generate_intent(self, tick: int, world_context: str) -> Intent:
        """
        Generate a structured intent for this tick.

        Args:
            tick: Current simulation tick
            world_context: GM-provided context (time, location, who's nearby)

        Returns:
            Structured Intent for the GM to adjudicate
        """
        # Build the action prompt
        call_to_action = self._build_intent_prompt(tick, world_context)

        action_spec = entity_lib.ActionSpec(
            call_to_action=call_to_action,
            output_type=entity_lib.OutputType.FREE,
        )

        # Get raw action from Concordia agent
        raw_action = self._entity.act(action_spec=action_spec)

        # Parse into structured intent
        return self._parse_intent(raw_action)

    def _build_intent_prompt(self, tick: int, world_context: str) -> str:
        """Build the prompt for intent generation."""
        from .world import tick_to_time_str

        time_str = tick_to_time_str(tick)
        needs_str = self._format_needs()

        return f"""You are {self.name}. It is {time_str}.

Current situation:
{world_context}

Your current state:
- Location: {self.current_location}
- {needs_str}
- Personality: {self.personality}

What do you want to do next? Choose ONE action:
- STAY: remain here
- GO_TO [place]: travel to a specific place (beach, tavern, market, clinic, temple, work_north)
- MEET [person]: seek out a specific person
- GO_HOME: return to {self.home_place}
- GO_WORK: go to work{f' at {self.work_place}' if self.work_place else ' (you have no workplace)'}

Respond with your action and a brief reason. Example:
GO_TO tavern - I'm feeling social and want to meet people."""

    def _format_needs(self) -> str:
        """Format needs for prompt."""
        parts = []
        if self.needs.hunger > 60:
            parts.append("quite hungry")
        elif self.needs.hunger > 40:
            parts.append("a bit hungry")
        if self.needs.fatigue > 70:
            parts.append("very tired")
        elif self.needs.fatigue > 50:
            parts.append("somewhat tired")
        if self.needs.social > 60:
            parts.append("lonely, wanting company")
        elif self.needs.social > 40:
            parts.append("could use some social time")

        if not parts:
            return "Feeling fine overall"
        return "Feeling: " + ", ".join(parts)

    def _parse_intent(self, raw_action: str) -> Intent:
        """Parse raw LLM output into structured Intent."""
        raw_lower = raw_action.lower().strip()

        # Extract reason after dash or hyphen
        reason = ""
        for sep in [" - ", " – ", ": ", " because "]:
            if sep in raw_action:
                reason = raw_action.split(sep, 1)[1].strip()
                raw_lower = raw_action.split(sep, 1)[0].lower().strip()
                break

        # Parse action type
        if raw_lower.startswith("stay"):
            return Intent(type=IntentType.STAY, reason=reason)

        if raw_lower.startswith("go_home") or raw_lower.startswith("go home"):
            return Intent(type=IntentType.GO_HOME, reason=reason)

        if raw_lower.startswith("go_work") or raw_lower.startswith("go work"):
            return Intent(type=IntentType.GO_WORK, reason=reason)

        if raw_lower.startswith("go_to") or raw_lower.startswith("go to"):
            # Extract target place
            parts = raw_lower.replace("go_to", "").replace("go to", "").strip().split()
            target = parts[0] if parts else None
            return Intent(type=IntentType.GO_TO, target_place=target, reason=reason)

        if raw_lower.startswith("meet"):
            # Extract target person
            parts = raw_lower.replace("meet", "").strip().split()
            target = parts[0] if parts else None
            return Intent(type=IntentType.MEET, target_person=target, reason=reason)

        if raw_lower.startswith("wander"):
            return Intent(type=IntentType.WANDER, reason=reason)

        # Default to stay if unparseable
        return Intent(type=IntentType.STAY, reason=f"(unparsed: {raw_action[:50]})")

    def tick_update(self) -> None:
        """Update agent state each tick (needs decay, transit progress)."""
        self.needs.tick_decay()

        if self.in_transit_ticks > 0:
            self.in_transit_ticks -= 1
            if self.in_transit_ticks == 0 and self.in_transit_to:
                # Arrived at destination
                self.current_location = self.in_transit_to
                self.in_transit_to = None

    def start_travel(self, destination: str, travel_ticks: int) -> None:
        """Begin traveling to a destination."""
        self.in_transit_to = destination
        self.in_transit_ticks = travel_ticks

    def update_relationship(self, other_name: str, delta: float) -> None:
        """Adjust relationship with another agent."""
        current = self.relationships.get(other_name, 0.0)
        self.relationships[other_name] = max(-100.0, min(100.0, current + delta))

    def get_state(self) -> IslandAgentState:
        """Get serializable agent state."""
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

    def is_available(self) -> bool:
        """Check if agent can act (not in transit)."""
        return self.in_transit_ticks == 0

    @property
    def location_display(self) -> str:
        """Human-readable location."""
        if self.in_transit_to:
            return f"traveling to {self.in_transit_to} ({self.in_transit_ticks} ticks left)"
        return self.current_location

    def get_last_log(self) -> dict[str, Any]:
        """
        Get the last Concordia component logs.

        Returns a dictionary mapping component/channel names to their latest output.
        This includes memory retrieval, observations, and the act component's reasoning.
        """
        try:
            return self._entity.get_last_log()
        except Exception:
            return {}

    def get_detailed_state(self) -> dict[str, Any]:
        """Get detailed agent state including Concordia component logs for debugging."""
        return {
            "name": self.name,
            "location": self.current_location,
            "in_transit_to": self.in_transit_to,
            "in_transit_ticks": self.in_transit_ticks,
            "needs": {
                "hunger": self.needs.hunger,
                "fatigue": self.needs.fatigue,
                "social": self.needs.social,
            },
            "personality": self.personality,
            "concordia_logs": self.get_last_log(),
        }


# Pre-defined agent configurations for the island
DEFAULT_ISLAND_AGENTS = [
    AgentConfig(
        name="Marina",
        home_place="beach",
        work_place=None,
        personality="Friendly beach vendor who loves the ocean and telling stories",
        backstory="Marina grew up on the island and knows everyone. She sells shells and snacks at the beach.",
        schedule_bias=ScheduleBias.EARLY_BIRD,
    ),
    AgentConfig(
        name="Finn",
        home_place="tavern",
        work_place="tavern",
        personality="Jovial tavern keeper with a booming laugh and endless gossip",
        backstory="Finn inherited the Driftwood Tavern from his uncle. He knows all the island secrets.",
        schedule_bias=ScheduleBias.NIGHT_OWL,
    ),
    AgentConfig(
        name="Rosa",
        home_place="market",
        work_place="market",
        personality="Sharp-witted market vendor who drives a hard bargain but has a soft heart",
        backstory="Rosa came to the island ten years ago and built up the most successful stall.",
        schedule_bias=ScheduleBias.NINE_TO_FIVE,
    ),
    AgentConfig(
        name="Dr. Kai",
        home_place="clinic",
        work_place="clinic",
        personality="Calm, methodical healer who speaks softly and listens carefully",
        backstory="Dr. Kai studied medicine on the mainland and returned to serve the island.",
        schedule_bias=ScheduleBias.NINE_TO_FIVE,
    ),
    AgentConfig(
        name="Elder Mako",
        home_place="temple",
        work_place="temple",
        personality="Wise temple keeper who speaks in riddles and knows ancient traditions",
        backstory="Elder Mako has tended the Old Stone Temple for fifty years.",
        schedule_bias=ScheduleBias.EARLY_BIRD,
    ),
    AgentConfig(
        name="Jade",
        home_place="work_north",
        work_place="work_north",
        personality="Skilled craftsperson who is quiet but observant",
        backstory="Jade apprenticed under a master woodworker and now runs the North Workshop.",
        schedule_bias=ScheduleBias.NINE_TO_FIVE,
    ),
    AgentConfig(
        name="Coral",
        home_place="beach",
        work_place=None,
        personality="Free-spirited artist who paints seascapes and dances in the moonlight",
        backstory="Coral arrived on the island seeking inspiration and never left.",
        schedule_bias=ScheduleBias.FLEXIBLE,
    ),
    AgentConfig(
        name="Rex",
        home_place="tavern",
        work_place=None,
        personality="Gruff retired sailor with tattoos and tall tales",
        backstory="Rex sailed the world for thirty years before settling at the tavern.",
        schedule_bias=ScheduleBias.NIGHT_OWL,
    ),
    AgentConfig(
        name="Lily",
        home_place="market",
        work_place="clinic",
        personality="Energetic young nurse who dreams of adventure beyond the island",
        backstory="Lily is Dr. Kai's assistant and the fastest runner on the island.",
        schedule_bias=ScheduleBias.NINE_TO_FIVE,
    ),
    AgentConfig(
        name="Sol",
        home_place="temple",
        work_place=None,
        personality="Mysterious wanderer who arrived recently and asks strange questions",
        backstory="Nobody knows where Sol came from. They appeared one morning at the temple.",
        schedule_bias=ScheduleBias.FLEXIBLE,
    ),
]


def create_island_agents(
    model: Any,
    embedder: Any,
    configs: list[AgentConfig] | None = None,
) -> list[IslandAgent]:
    """
    Create all island agents with shared memory infrastructure.

    Args:
        model: LLM model for agent cognition
        embedder: Text embedder for memory retrieval
        configs: Agent configurations (defaults to DEFAULT_ISLAND_AGENTS)

    Returns:
        List of initialized IslandAgent instances
    """
    configs = configs or DEFAULT_ISLAND_AGENTS

    # Create shared memory bank (agents can remember shared events)
    shared_memory = basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=embedder,
    )

    # Seed with island context
    shared_memory.add("This is a small tropical island with a beach, tavern, market, clinic, temple, and workshop.")
    shared_memory.add("The islanders know each other well and often meet at the market or tavern.")

    agents = []
    for config in configs:
        # Each agent gets their own memory bank for personal memories
        personal_memory = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=embedder,
        )
        agent = IslandAgent(config=config, model=model, memory_bank=personal_memory)
        agents.append(agent)

    return agents
