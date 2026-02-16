# Concordia Architecture Guide

This document explains the idiomatic Concordia implementation of the Island Simulation, following proper prefab patterns and component architecture.

## CRITICAL: Development Policy

**When working with Concordia, ALWAYS use Concordia's actual functionality.**

### Do NOT:
- Create workaround scripts that bypass Concordia's engine/GM architecture
- Implement custom simulation loops that duplicate Concordia functionality
- Build "lightweight" alternatives that ignore Concordia's component system

### Instead, DO:
- Use existing Concordia engines (`Sequential`, `Simultaneous`, `ParallelQuestionnaireEngine`)
- If an engine doesn't fit your needs, **create a new Engine subclass** that extends `engine_lib.Engine`
- Use Concordia's `concurrency.run_tasks()` for parallel execution
- Build proper GM components using `entity_component.ContextComponent`
- Use `AssociativeMemoryBank` for agent memory, not simple lists
- Follow the prefab pattern for reusable entity/GM definitions

### When Concordia Functionality is Insufficient:
1. First check if there's an existing engine or component that fits
2. If not, create a **new Concordia engine class** that extends the base
3. Create **custom GM components** following the component interface
4. Submit improvements upstream to Concordia if generally useful

This project is maintained by Concordia contributors. We extend Concordia properly, not work around it.

## Overview

The refactored codebase uses **Concordia prefabs** - the recommended pattern for building multi-agent simulations with Google DeepMind's Concordia framework.

### Key Files

- **`sim/prefabs_entity.py`** - Island entity prefab (agent definition)
- **`sim/prefabs_gm.py`** - Island Game Master prefab (world management)
- **`sim/sim_concordia.py`** - Simulation configuration and multi-sim coordinator
- **`run_concordia.py`** - CLI runner using the new architecture

## Concordia Prefab Pattern

### What is a Prefab?

A **prefab** (prefabricated component) is a reusable template for creating Concordia entities or game masters. It's implemented as a dataclass with:

```python
@dataclasses.dataclass
class MyPrefab(prefab_lib.Prefab):
    description: str = "What this prefab creates"
    params: Mapping[str, Any] = dataclasses.field(default_factory=...)

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        # Build and return the entity
        ...
```

### Benefits of Prefabs

1. **Reusability** - Define once, instantiate many times with different params
2. **Composability** - Prefabs can use other prefabs as components
3. **Configuration** - Params allow customization without code changes
4. **Type Safety** - Dataclass structure provides IDE support and validation

## Entity Prefab: `IslandEntity`

### The Three Key Questions Pattern

Our entity prefab follows Concordia's recommended "three key questions" pattern:

1. **"What situation am I in right now?"** (`SituationPerception`)
2. **"What kind of person am I?"** (`SelfPerception`)
3. **"What would a person like me do?"** (`PersonBySituation`)

### Component Architecture

```python
IslandEntity components:
├── Instructions          # Agent identity and backstory
├── Goal                  # Overarching motivation
├── Personality           # Core personality trait
├── TimeDisplay           # Awareness of time/place
├── Memory                # AssociativeMemory for recall
├── RelevantMemories      # Retrieve relevant past experiences
├── Observation           # Track recent events
├── SituationPerception   # Understand current context
├── SelfPerception        # Self-awareness
└── PersonBySituation     # Decision-making
```

### How It Works

1. **Observation Phase** - Agent receives observations from the GM
2. **Memory Retrieval** - Semantic search finds relevant past experiences
3. **Perception** - LLM answers the three key questions
4. **Action** - Components concatenated into prompt, LLM generates action
5. **Memory Storage** - Observations added to memory bank

### Parameters

```python
params = {
    'name': 'Alice',
    'home_place': 'sunset_apartments',
    'work_place': 'cafe',
    'personality': 'Cheerful and hardworking',
    'backstory': 'Moved to the island 5 years ago...',
    'schedule_bias': 'EARLY_BIRD',
    'formative_memories': ['Alice loves coffee...'],
    'goal': 'Build friendships and thrive in cafe work',
}
```

## Game Master Prefab: `IslandGameMaster`

### Situated-in-Time-and-Place Pattern

The GM prefab adapts Concordia's `situated_in_time_and_place` pattern:

- **Generative Clock** - Tracks time in 30-min ticks (48/day)
- **Locations** - Tracks where each agent is located
- **World State** - Aggregates overall simulation state
- **Event Resolution** - Resolves agent actions with thought chains
- **Observations** - Generates personalized observations for agents

### Component Architecture

```python
IslandGameMaster components:
├── Instructions          # GM rules and guidelines
├── PlayerCharacters      # List of all agents
├── ClockConstant         # Time system description
├── LocationsConstant     # Island geography
├── Memory                # Shared GM memory
├── RelevantMemories      # Context from past events
├── Observation           # Event history
├── DisplayEvents         # Show recent events
├── GenerativeClock       # Current time tracking
├── Locations             # Current agent positions
├── WorldState            # Aggregate state
├── MakeObservation       # Create agent observations
├── NextActor             # Determine who acts next
├── NextActionSpec        # Define action format
└── EventResolution       # Resolve actions with thought chains
```

### Thought Chains

Event resolution uses **thought chains** for structured reasoning:

1. **AccountForAgencyOfOthers** - Consider other agents' perspectives
2. **result_to_who_what_where** - Extract structured event data

These can be extended with additional chains like:
- `attempt_to_most_likely_outcome` - Reason about success probability
- `extract_consequences` - Identify downstream effects

## Simulation Configuration

### Config Structure

Concordia uses a `Config` object with **prefabs** and **instances**:

```python
config = Config(
    prefabs={
        'island_entity': IslandEntity(),
        'island_gm': IslandGameMaster(),
    },
    instances=[
        InstanceConfig(
            role=Role.ENTITY,
            prefab='island_entity',
            params={'name': 'Alice', ...}
        ),
        InstanceConfig(
            role=Role.GAME_MASTER,
            prefab='island_gm',
            params={'name': 'Island GM', ...}
        ),
    ]
)
```

### Roles

- **`Role.ENTITY`** - Player agents (passive, react to observations)
- **`Role.GAME_MASTER`** - World managers (have references to all entities)
- **`Role.INITIALIZER`** - Setup-only GMs (run once at start)

## Multi-Simulation System

### Why Multiple Simulations?

For 100 agents, running a single simulation can be computationally expensive. The **MultiIslandSimulation** coordinator allows:

1. **Scalability** - Split agents across multiple smaller simulations
2. **Parallelization** - Run neighborhoods concurrently
3. **Resource Management** - Control memory and LLM call volume

### Neighborhood-Based Splitting

Agents are grouped by home location:

```python
neighborhoods = create_neighborhood_simulations(
    all_agents=ISLAND_100_AGENTS,
    island=island,
    model=model,
    embedder=embedder,
    max_agents_per_sim=25,  # 4 simulations for 100 agents
)
```

### Architecture

```
MultiIslandSimulation
├── Neighborhood 1 (25 agents)
│   └── IslandSimulationConcordia
│       ├── Entities (25)
│       └── GameMaster
├── Neighborhood 2 (25 agents)
├── Neighborhood 3 (25 agents)
└── Neighborhood 4 (25 agents)
```

### Agent Swapping (Planned)

For agents traveling between neighborhoods, we plan to support:

```python
multi_sim.swap_agent(
    agent_name='Alice',
    from_neighborhood='sunset_apartments',
    to_neighborhood='palm_heights',
)
```

This would:
1. Save Alice's state from sim 1
2. Remove Alice from sim 1
3. Add Alice to sim 2
4. Restore Alice's state

## Running the Simulation

### Single Simulation (10 agents)

```bash
python run_concordia.py --agents 10 --steps 10
```

### Multi-Simulation (100 agents)

```bash
python run_concordia.py --agents 100 --steps 10 --multi-sim --max-per-sim 25
```

### Environment Setup

```bash
# Set API key
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"

# Install sentence-transformers for better embeddings
pip install sentence-transformers
```

## Comparison: Old vs New Architecture

### Old (Custom Implementation)

```python
# agents.py - Custom agent class
class IslandAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.entity = self._create_concordia_entity()

# gm.py - Custom game master
class GameMaster:
    def __init__(self, island, agents, model):
        self.island = island
        self.agents = agents
        self.validate_intents()
        self.execute_movements()
```

**Limitations:**
- Custom abstractions hiding Concordia patterns
- Limited component reusability
- Hard to extend with new Concordia features
- Non-standard architecture

### New (Concordia Prefabs)

```python
# prefabs_entity.py - Standard Concordia prefab
@dataclasses.dataclass
class IslandEntity(prefab_lib.Prefab):
    def build(self, model, memory_bank):
        # Uses standard Concordia components
        return entity_agent_with_logging.EntityAgentWithLogging(...)

# prefabs_gm.py - Standard Concordia GM prefab
@dataclasses.dataclass
class IslandGameMaster(prefab_lib.Prefab):
    def build(self, model, memory_bank):
        # Uses gm_components from Concordia
        return entity_agent_with_logging.EntityAgentWithLogging(...)
```

**Benefits:**
- Follows official Concordia patterns
- Easily extendable with new components
- Compatible with Concordia ecosystem
- Better documentation and community support

## Component Prompting

### How Components Build Prompts

When an entity acts, Concordia concatenates component outputs:

```
Instructions: Alice is a resident of the island. She is cheerful and hardworking...

Goal: Build friendships and thrive in cafe work

Core personality: Alice is cheerful and hardworking

Time awareness: Pay attention to the current time and location...

Recent events:
- // Day 1, 08:00 // Alice woke up in her apartment
- // Day 1, 08:30 // Alice walked to the cafe

Recalled memories:
- Alice loves making latte art
- Alice became friends with Maria at the cafe
...

Question: What situation is Alice in right now?
Answer: Alice just arrived at the cafe for her shift...

Question: What kind of person is Alice?
Answer: Alice is a cheerful, hardworking barista who values relationships...

Question: What would a person like Alice do?
Answer: Alice would greet her coworkers warmly and start preparing for customers...

Alice's action: [LLM generates based on above context]
```

### Customizing Prompts

Modify component labels:

```python
observation = agent_components.observation.LastNObservations(
    history_length=50,
    pre_act_label='\n=== RECENT EVENTS ===',  # Custom label
)
```

## Advanced Patterns

### Adding New Components

```python
# Add a relationship tracker
relationships_key = 'Relationships'
relationships = agent_components.constant.Constant(
    state=f'{entity_name} has formed close bonds with...',
    pre_act_label='\nKey relationships'
)

components_of_agent[relationships_key] = relationships
component_order.insert(4, relationships_key)  # Insert at position 4
```

### Extending Thought Chains

```python
# Add custom thought chain for event resolution
def extract_emotions(event: str) -> str:
    """Extract emotional content from event."""
    # Custom logic here
    return emotion_analysis

event_resolution_steps = [
    account_for_agency_of_others,
    extract_emotions,  # Custom chain
    thought_chains_lib.result_to_who_what_where,
]
```

### Custom Components

Create your own component:

```python
from concordia.typing import entity_component

class NeedsTracker(entity_component.ContextComponent):
    def __init__(self, agent_name: str):
        self._agent_name = agent_name
        self._hunger = 0
        self._fatigue = 0

    def pre_act(self, unused_arg) -> str:
        return f'\n{self._agent_name} needs: hunger={self._hunger}, fatigue={self._fatigue}'

    def update_needs(self, hunger: int, fatigue: int):
        self._hunger = hunger
        self._fatigue = fatigue
```

## Debugging and Logging

### Component Logs

Access internal component reasoning:

```python
entity = simulation.get_entities()[0]
logs = entity.get_last_log()

# logs contains output from each component:
# {
#   'SituationPerception': '...',
#   'SelfPerception': '...',
#   'PersonBySituation': '...',
# }
```

### Memory Inspection

```python
memory_component = entity.get_component('__memory__')
recent_memories = memory_component.retrieve(
    query='Alice at the cafe',
    limit=10
)
```

## Performance Considerations

### For 10 Agents

- **Single simulation** - Works well
- **LLM calls per step** - ~10 agent calls + ~5 GM calls
- **Memory usage** - Moderate

### For 100 Agents

- **Multi-simulation recommended** - Split into 4-5 sims
- **LLM calls per step** - ~100 agent + ~25 GM calls
- **Memory usage** - High (each agent has personal memory bank)

### Optimization Strategies

1. **Batch LLM calls** - Group prompts together
2. **Reduce history length** - Limit `LastNObservations` to 20-50
3. **Limit memory retrieval** - Set `num_memories_to_retrieve=10-15`
4. **Use faster models** - GPT-4.1-nano or Claude Haiku
5. **Async execution** - Run neighborhood sims in parallel

## Resources

- **Concordia GitHub**: https://github.com/google-deepmind/concordia
- **Concordia Docs**: https://google-deepmind.github.io/concordia/
- **Prefabs Directory**: `concordia/prefabs/`
- **Component Directory**: `concordia/components/`

## Next Steps

1. **Experiment with components** - Add goal tracking, trait systems, belief models
2. **Implement cross-sim swapping** - Allow agents to move between neighborhoods
3. **Add visualizations** - Build UI for Concordia-based sim
4. **Create custom thought chains** - Domain-specific reasoning for island life
5. **Optimize for scale** - Profile and optimize for 100+ agents

---

For questions or contributions, see the main README.md.
