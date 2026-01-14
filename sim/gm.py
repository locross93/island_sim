"""
Time & Place Game Master - the canonical authority on movement, encounters, and world state.

The GM:
- Advances time in discrete ticks (30 min each)
- Validates agent intents against open hours and travel times
- Moves agents through the island graph
- Spawns encounters when agents co-locate
- Runs minimal interaction scenes via LLM
- Updates agent memories and relationships
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .agents import Intent, IntentType, IslandAgent
from .world import Island, Place, tick_to_time_str, tick_to_day_and_time, TICKS_PER_DAY


class EncounterRecord(BaseModel):
    """Record of an encounter between agents."""
    tick: int
    place_id: str
    participants: list[str]
    summary: str
    dialogue: list[tuple[str, str]] = []  # (speaker, utterance)


class MovementRecord(BaseModel):
    """Record of an agent movement."""
    tick: int
    agent_name: str
    from_place: str
    to_place: str
    travel_ticks: int
    reason: str = ""


class TickResult(BaseModel):
    """Result of processing a single tick."""
    tick: int
    time_str: str
    day: int
    movements: list[MovementRecord] = []
    encounters: list[EncounterRecord] = []
    observations: dict[str, list[str]] = {}  # agent_name -> observations


@dataclass
class LLMLogEntry:
    """Log entry for an LLM call."""
    tick: int
    type: str  # "encounter", "intent", etc.
    prompt: str
    response: str
    participants: list[str] = field(default_factory=list)


@dataclass
class ConcordiaLogEntry:
    """Log entry for Concordia component outputs."""
    tick: int
    agent_name: str
    intent_type: str
    intent_reason: str
    component_logs: dict[str, str] = field(default_factory=dict)  # channel -> output


@dataclass
class GameMaster:
    """
    Time & Place Game Master for the island simulation.

    Agents propose actions via intents — GM disposes by validating
    and executing them according to world rules.
    """
    island: Island
    agents: list[IslandAgent]
    model: Any  # LLM for encounter narration
    current_tick: int = 0
    history: list[TickResult] = field(default_factory=list)
    encounter_history: list[EncounterRecord] = field(default_factory=list)
    llm_logs: list[LLMLogEntry] = field(default_factory=list)
    concordia_logs: list[ConcordiaLogEntry] = field(default_factory=list)

    def __post_init__(self):
        self._agents_by_name: dict[str, IslandAgent] = {
            agent.name: agent for agent in self.agents
        }

    def get_agent(self, name: str) -> IslandAgent | None:
        """Get agent by name."""
        return self._agents_by_name.get(name)

    def agents_at_place(self, place_id: str) -> list[IslandAgent]:
        """Get all agents currently at a place (not in transit)."""
        return [
            agent for agent in self.agents
            if agent.current_location == place_id and agent.is_available()
        ]

    def get_world_context(self, agent: IslandAgent) -> str:
        """Generate world context for an agent's decision-making."""
        day, time_str = tick_to_day_and_time(self.current_tick)
        place = self.island.get_place(agent.current_location)

        # Who else is here?
        others_here = [
            a.name for a in self.agents_at_place(agent.current_location)
            if a.name != agent.name
        ]

        # What places are nearby and open?
        neighbors = self.island.neighbors(agent.current_location)
        open_neighbors = []
        for n in neighbors:
            p = self.island.get_place(n)
            if p and p.is_open(self.current_tick):
                travel = self.island.travel_ticks(agent.current_location, n)
                open_neighbors.append(f"{p.name} ({travel * 30} min away)")

        context_parts = [
            f"Day {day}, {time_str}",
            f"You are at {place.name}." if place else f"You are at {agent.current_location}.",
        ]

        if others_here:
            context_parts.append(f"Also here: {', '.join(others_here)}.")
        else:
            context_parts.append("You are alone here.")

        if open_neighbors:
            context_parts.append(f"Nearby open places: {', '.join(open_neighbors)}.")

        # Recent encounters
        recent = self._get_recent_encounters(agent.name, limit=2)
        if recent:
            context_parts.append("Recent events:")
            for enc in recent:
                context_parts.append(f"  - {enc.summary}")

        return "\n".join(context_parts)

    def _get_recent_encounters(self, agent_name: str, limit: int = 3) -> list[EncounterRecord]:
        """Get recent encounters involving an agent."""
        relevant = [
            e for e in self.encounter_history
            if agent_name in e.participants
        ]
        return relevant[-limit:]

    def validate_intent(self, agent: IslandAgent, intent: Intent) -> tuple[bool, str]:
        """
        Validate an agent's intent against world rules.

        Returns (is_valid, reason).
        """
        if not agent.is_available():
            return False, f"Still traveling to {agent.in_transit_to}"

        if intent.type == IntentType.STAY:
            return True, "Staying put"

        if intent.type == IntentType.GO_HOME:
            if agent.current_location == agent.home_place:
                return False, "Already at home"
            return True, f"Going home to {agent.home_place}"

        if intent.type == IntentType.GO_WORK:
            if not agent.work_place:
                return False, "No workplace assigned"
            if agent.current_location == agent.work_place:
                return False, "Already at work"
            return True, f"Going to work at {agent.work_place}"

        if intent.type == IntentType.GO_TO:
            target = intent.target_place
            if not target:
                return False, "No target place specified"

            # Normalize target (handle variations)
            target = self._normalize_place_id(target)
            if not target:
                return False, f"Unknown place: {intent.target_place}"

            if agent.current_location == target:
                return False, "Already at that location"

            place = self.island.get_place(target)
            if not place:
                return False, f"Unknown place: {target}"

            if not place.is_open(self.current_tick):
                return False, f"{place.name} is closed"

            travel_ticks = self.island.travel_ticks(agent.current_location, target)
            if travel_ticks < 0:
                return False, f"No path to {place.name}"

            return True, f"Traveling to {place.name} ({travel_ticks * 30} min)"

        if intent.type == IntentType.MEET:
            target_name = intent.target_person
            if not target_name:
                return False, "No target person specified"

            target_agent = self._find_agent_by_name(target_name)
            if not target_agent:
                return False, f"Unknown person: {target_name}"

            if target_agent.current_location == agent.current_location:
                return True, f"{target_name} is already here"

            # Go to where target is (if known and open)
            target_place = self.island.get_place(target_agent.current_location)
            if target_place and target_place.is_open(self.current_tick):
                return True, f"Going to find {target_name} at {target_place.name}"

            return False, f"Can't reach {target_name} right now"

        if intent.type == IntentType.WANDER:
            # GM picks a random open nearby place
            return True, "Wandering to a nearby place"

        return False, f"Unknown intent type: {intent.type}"

    def _normalize_place_id(self, place_str: str) -> str | None:
        """Normalize a place string to a valid place ID."""
        place_str = place_str.lower().strip()

        # Direct match
        if place_str in self.island.places:
            return place_str

        # Try common variations
        variations = {
            "the beach": "beach",
            "sunny beach": "beach",
            "the tavern": "tavern",
            "driftwood tavern": "tavern",
            "the driftwood tavern": "tavern",
            "the market": "market",
            "island market": "market",
            "the clinic": "clinic",
            "the temple": "temple",
            "old stone temple": "temple",
            "the workshop": "work_north",
            "north workshop": "work_north",
            "workshop": "work_north",
        }

        if place_str in variations:
            return variations[place_str]

        # Partial match
        for pid in self.island.places:
            if place_str in pid or pid in place_str:
                return pid

        return None

    def _find_agent_by_name(self, name: str) -> IslandAgent | None:
        """Find an agent by name (case-insensitive partial match)."""
        name_lower = name.lower().strip()

        # Exact match
        for agent in self.agents:
            if agent.name.lower() == name_lower:
                return agent

        # Partial match
        for agent in self.agents:
            if name_lower in agent.name.lower():
                return agent

        return None

    def execute_intent(self, agent: IslandAgent, intent: Intent) -> MovementRecord | None:
        """Execute a validated intent, returning movement record if applicable."""
        if intent.type == IntentType.STAY:
            return None

        target_place: str | None = None

        if intent.type == IntentType.GO_HOME:
            target_place = agent.home_place

        elif intent.type == IntentType.GO_WORK:
            target_place = agent.work_place

        elif intent.type == IntentType.GO_TO:
            target_place = self._normalize_place_id(intent.target_place or "")

        elif intent.type == IntentType.MEET:
            target_agent = self._find_agent_by_name(intent.target_person or "")
            if target_agent and target_agent.current_location != agent.current_location:
                target_place = target_agent.current_location

        elif intent.type == IntentType.WANDER:
            # Pick a random open neighbor
            import random
            neighbors = self.island.neighbors(agent.current_location)
            open_neighbors = [
                n for n in neighbors
                if self.island.get_place(n) and self.island.get_place(n).is_open(self.current_tick)
            ]
            if open_neighbors:
                target_place = random.choice(open_neighbors)

        if target_place and target_place != agent.current_location:
            travel_ticks = self.island.travel_ticks(agent.current_location, target_place)
            if travel_ticks > 0:
                from_place = agent.current_location
                agent.start_travel(target_place, travel_ticks)

                return MovementRecord(
                    tick=self.current_tick,
                    agent_name=agent.name,
                    from_place=from_place,
                    to_place=target_place,
                    travel_ticks=travel_ticks,
                    reason=intent.reason,
                )

        return None

    def detect_encounters(self) -> list[tuple[str, list[IslandAgent]]]:
        """Detect places where multiple agents are co-located."""
        encounters = []

        for place_id in self.island.all_place_ids():
            agents_here = self.agents_at_place(place_id)
            if len(agents_here) >= 2:
                encounters.append((place_id, agents_here))

        return encounters

    async def run_encounter(self, place_id: str, participants: list[IslandAgent]) -> EncounterRecord:
        """Run a minimal interaction scene between co-located agents."""
        place = self.island.get_place(place_id)
        place_name = place.name if place else place_id

        participant_names = [p.name for p in participants]

        # Build encounter prompt
        prompt = self._build_encounter_prompt(place_name, participants)

        # Generate interaction via LLM (use the model passed to GM)
        narration = self._generate_narration(prompt, participant_names, place_name)

        # Parse dialogue from narration (simple extraction)
        dialogue = self._extract_dialogue(narration, participant_names)

        # Create summary
        summary = self._summarize_encounter(narration, participant_names, place_name)

        record = EncounterRecord(
            tick=self.current_tick,
            place_id=place_id,
            participants=participant_names,
            summary=summary,
            dialogue=dialogue,
        )

        # Update agent memories and relationships
        for agent in participants:
            agent.observe(f"At {place_name}: {summary}")
            # Small positive relationship boost for meeting
            for other in participants:
                if other.name != agent.name:
                    agent.update_relationship(other.name, 2.0)
                    agent.needs.socialize(5.0)

        self.encounter_history.append(record)
        return record

    def _generate_narration(self, prompt: str, participant_names: list[str], place_name: str) -> str:
        """Generate encounter narration using the configured LLM."""
        fallback = f"{', '.join(participant_names)} meet at {place_name} and exchange greetings."

        if not self.model:
            self._log_llm_call("encounter", prompt, f"[No LLM] {fallback}", participant_names)
            return fallback

        try:
            response_text = None
            # Check if it's an OpenAI client
            if hasattr(self.model, 'chat'):
                response = self.model.chat.completions.create(
                    model="gpt-4.1-nano",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content
            # Check if it's an Anthropic client
            elif hasattr(self.model, 'messages'):
                response = self.model.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text

            if response_text:
                self._log_llm_call("encounter", prompt, response_text, participant_names)
                return response_text
            else:
                self._log_llm_call("encounter", prompt, f"[No response] {fallback}", participant_names)
                return fallback

        except Exception as e:
            # Fallback if LLM unavailable
            self._log_llm_call("encounter", prompt, f"[Error: {e}] {fallback}", participant_names)
            return fallback

    def _log_llm_call(self, call_type: str, prompt: str, response: str, participants: list[str] = None):
        """Log an LLM call for visibility."""
        self.llm_logs.append(LLMLogEntry(
            tick=self.current_tick,
            type=call_type,
            prompt=prompt,
            response=response,
            participants=participants or [],
        ))

    def _build_encounter_prompt(self, place_name: str, participants: list[IslandAgent]) -> str:
        """Build the prompt for generating an encounter scene."""
        day, time_str = tick_to_day_and_time(self.current_tick)

        char_descriptions = []
        for p in participants:
            rel_info = []
            for other in participants:
                if other.name != p.name:
                    rel = p.relationships.get(other.name, 0)
                    if rel > 20:
                        rel_info.append(f"friendly with {other.name}")
                    elif rel < -20:
                        rel_info.append(f"wary of {other.name}")

            desc = f"- {p.name}: {p.personality}"
            if rel_info:
                desc += f" ({', '.join(rel_info)})"
            char_descriptions.append(desc)

        return f"""Write a brief encounter scene (3-5 exchanges) at {place_name} on the island.

Time: Day {day}, {time_str}

Characters:
{chr(10).join(char_descriptions)}

Write natural dialogue showing their personalities. Keep it short and authentic to island life.
Format each line as: CHARACTER: "Dialogue"

End with a one-sentence summary of what happened."""

    def _extract_dialogue(self, narration: str, names: list[str]) -> list[tuple[str, str]]:
        """Extract dialogue turns from narration."""
        dialogue = []

        for line in narration.split("\n"):
            line = line.strip()
            for name in names:
                if line.startswith(f"{name}:"):
                    utterance = line[len(name) + 1:].strip().strip('"').strip()
                    if utterance:
                        dialogue.append((name, utterance))
                    break

        return dialogue

    def _summarize_encounter(self, narration: str, names: list[str], place: str) -> str:
        """Create a brief summary of the encounter."""
        # Look for explicit summary at the end
        lines = narration.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and not any(line.startswith(f"{n}:") for n in names):
                if len(line) > 20:
                    return line

        # Fallback summary
        return f"{' and '.join(names)} had a conversation at {place}."

    def run_tick_sync(self) -> TickResult:
        """
        Run a single simulation tick (synchronous version).

        1. Update agents (arrivals, needs decay)
        2. Detect and run encounters FIRST (before new decisions)
        3. Collect intents from agents
        4. Validate and execute intents
        5. Update world state
        """
        day, time_str = tick_to_day_and_time(self.current_tick)

        result = TickResult(
            tick=self.current_tick,
            time_str=time_str,
            day=day,
        )

        # Update all agents (decay needs, progress travel)
        for agent in self.agents:
            agent.tick_update()

        # Detect encounters FIRST - before agents decide to move again
        encounters = self.detect_encounters()
        agents_in_encounter = set()
        for place_id, participants in encounters:
            names = [p.name for p in participants]
            place = self.island.get_place(place_id)
            place_name = place.name if place else place_id

            # Generate LLM encounter if model available
            narration = self._generate_narration(
                self._build_encounter_prompt(place_name, participants),
                names,
                place_name
            )

            dialogue = self._extract_dialogue(narration, names)
            summary = self._summarize_encounter(narration, names, place_name)

            record = EncounterRecord(
                tick=self.current_tick,
                place_id=place_id,
                participants=names,
                summary=summary,
                dialogue=dialogue,
            )
            result.encounters.append(record)
            self.encounter_history.append(record)

            # Update relationships and mark agents as in encounter
            for agent in participants:
                agents_in_encounter.add(agent.name)
                agent.observe(f"At {place_name}: {summary}")
                for other in participants:
                    if other.name != agent.name:
                        agent.update_relationship(other.name, 2.0)
                        agent.needs.socialize(10.0)

        # Collect and execute intents for available agents NOT in an encounter
        for agent in self.agents:
            if not agent.is_available():
                result.observations[agent.name] = [f"Traveling to {agent.in_transit_to}..."]
                continue

            # Agents in encounters stay put this tick
            if agent.name in agents_in_encounter:
                result.observations[agent.name] = ["Had an encounter, staying put."]
                continue

            # Generate world context
            context = self.get_world_context(agent)
            result.observations[agent.name] = [context]

            # For sync version, use a simple heuristic instead of LLM
            intent = self._generate_simple_intent(agent)

            # Validate and execute
            valid, reason = self.validate_intent(agent, intent)
            if valid:
                movement = self.execute_intent(agent, intent)
                if movement:
                    result.movements.append(movement)
                    result.observations[agent.name].append(f"Started traveling: {reason}")
            else:
                result.observations[agent.name].append(f"Stayed put: {reason}")

        self.history.append(result)
        self.current_tick += 1

        return result

    async def run_tick(self) -> TickResult:
        """
        Run a single simulation tick (async version with LLM).

        1. Generate observations for each agent
        2. Collect intents from agents (via LLM)
        3. Validate and execute intents
        4. Detect and run encounters (via LLM)
        5. Update world state
        """
        day, time_str = tick_to_day_and_time(self.current_tick)

        result = TickResult(
            tick=self.current_tick,
            time_str=time_str,
            day=day,
        )

        # Update all agents
        for agent in self.agents:
            agent.tick_update()

        # Collect intents from available agents
        for agent in self.agents:
            if not agent.is_available():
                result.observations[agent.name] = [f"Traveling to {agent.in_transit_to}..."]
                continue

            # Generate world context and get intent
            context = self.get_world_context(agent)
            result.observations[agent.name] = [context]

            intent = agent.generate_intent(self.current_tick, context)

            # Capture Concordia component logs after intent generation
            component_logs = agent.get_last_log()
            if component_logs:
                # Convert any non-string values to strings for display
                formatted_logs = {}
                for channel, value in component_logs.items():
                    if value is not None:
                        formatted_logs[channel] = str(value) if not isinstance(value, str) else value

                self.concordia_logs.append(ConcordiaLogEntry(
                    tick=self.current_tick,
                    agent_name=agent.name,
                    intent_type=str(intent.type.value),
                    intent_reason=intent.reason,
                    component_logs=formatted_logs,
                ))

            # Validate and execute
            valid, reason = self.validate_intent(agent, intent)
            if valid:
                movement = self.execute_intent(agent, intent)
                if movement:
                    result.movements.append(movement)
                    result.observations[agent.name].append(f"Started traveling: {reason}")
            else:
                result.observations[agent.name].append(f"Stayed put: {reason}")

        # Detect and run encounters
        encounters = self.detect_encounters()
        for place_id, participants in encounters:
            record = await self.run_encounter(place_id, participants)
            result.encounters.append(record)

        self.history.append(result)
        self.current_tick += 1

        return result

    def _generate_simple_intent(self, agent: IslandAgent) -> Intent:
        """Generate a simple intent based on needs and schedule (no LLM)."""
        import random

        hour = (self.current_tick * 30 // 60) % 24

        # Check if should be sleeping (based on schedule bias)
        from .agents import ScheduleBias
        should_sleep = False
        if agent.schedule_bias == ScheduleBias.EARLY_BIRD:
            should_sleep = hour >= 21 or hour < 5
        elif agent.schedule_bias == ScheduleBias.NIGHT_OWL:
            should_sleep = hour >= 2 and hour < 10
        elif agent.schedule_bias == ScheduleBias.NINE_TO_FIVE:
            should_sleep = hour >= 22 or hour < 7

        # High fatigue or sleep time -> go home
        if agent.needs.fatigue > 70 or (should_sleep and agent.needs.fatigue > 40):
            if agent.current_location != agent.home_place:
                return Intent(type=IntentType.GO_HOME, reason="Need to rest")
            return Intent(type=IntentType.STAY, reason="Resting at home")

        # Work hours -> maybe go to work
        if agent.work_place and 9 <= hour < 17:
            if agent.current_location != agent.work_place and random.random() < 0.7:
                return Intent(type=IntentType.GO_WORK, reason="Time for work")

        # High hunger -> go to market
        if agent.needs.hunger > 60:
            if agent.current_location != "market":
                place = self.island.get_place("market")
                if place and place.is_open(self.current_tick):
                    return Intent(type=IntentType.GO_TO, target_place="market", reason="Need to eat")

        # High social need -> go somewhere social
        if agent.needs.social > 50:
            social_places = ["tavern", "beach", "market"]
            for place_id in social_places:
                if agent.current_location != place_id:
                    place = self.island.get_place(place_id)
                    if place and place.is_open(self.current_tick):
                        return Intent(type=IntentType.GO_TO, target_place=place_id, reason="Want company")

        # Random wandering
        if random.random() < 0.2:
            return Intent(type=IntentType.WANDER, reason="Feeling restless")

        return Intent(type=IntentType.STAY, reason="Content here")

    def get_state_summary(self) -> dict:
        """Get a summary of current world state."""
        locations = {}
        for place_id in self.island.all_place_ids():
            agents_here = self.agents_at_place(place_id)
            if agents_here:
                locations[place_id] = [a.name for a in agents_here]

        in_transit = [
            {"name": a.name, "to": a.in_transit_to, "ticks": a.in_transit_ticks}
            for a in self.agents if not a.is_available()
        ]

        day, time_str = tick_to_day_and_time(self.current_tick)

        return {
            "tick": self.current_tick,
            "day": day,
            "time": time_str,
            "locations": locations,
            "in_transit": in_transit,
            "recent_encounters": len([e for e in self.encounter_history if e.tick > self.current_tick - 4]),
        }

    def print_status(self) -> None:
        """Print current world status to console."""
        summary = self.get_state_summary()

        print(f"\n=== Day {summary['day']}, {summary['time']} (tick {summary['tick']}) ===")

        for place_id, agents in summary["locations"].items():
            place = self.island.get_place(place_id)
            place_name = place.name if place else place_id
            print(f"  {place_name}: {', '.join(agents)}")

        if summary["in_transit"]:
            print("  In transit:")
            for t in summary["in_transit"]:
                print(f"    {t['name']} -> {t['to']} ({t['ticks']} ticks)")

        if summary["recent_encounters"] > 0:
            print(f"  Recent encounters: {summary['recent_encounters']}")
