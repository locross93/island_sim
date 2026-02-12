# Island Time-and-Place Concordia Simulation

A spatial, time-based multi-agent simulation where LLM-powered agents live on an island with realistic economic inequality and social dynamics. Built with Google DeepMind's Concordia framework.

**Two versions available:**
- **Original**: 10 agents on a small island (for testing and development)
- **100-Agent**: 100 agents with economic inequality and power law wealth distribution (realistic simulation)

## Features

- **10 Unique Agents**: Each with personality, home, workplace, needs (hunger, fatigue, social), and relationships
- **Island World**: 6 locations (beach, tavern, market, clinic, temple, workshop) connected by paths with travel times
- **Time System**: 30-minute ticks, 48 ticks per day, with day/night cycles and open hours
- **Idiomatic Concordia Architecture**: Uses proper prefab patterns with Entity and Game Master prefabs
- **Three Key Questions**: Agents reason using SituationPerception, SelfPerception, and PersonBySituation components
- **Situated-in-Time-and-Place GM**: Game Master uses Concordia's world state, generative clock, and event resolution
- **Multi-Simulation Support**: Scale to 100+ agents by splitting across multiple coordinated simulations
- **Encounters**: When agents meet, the LLM generates natural dialogue based on their personalities
- **Streamlit UI**: Real-time visualization with island map, agent panels, and comprehensive logging

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set your API key as an environment variable:
```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

Alternatively, create `api_key.json` in the project root:
```json
{"API_KEY": "your-openai-api-key"}
```

## Running

### ⭐ **NEW: Concordia Prefab Architecture** (Recommended)

Run the simulation using proper Concordia prefabs and patterns:

```bash
# 10-agent simulation with Concordia prefabs
python run_concordia.py --agents 10 --steps 10

# 100-agent multi-simulation (4 neighborhood simulations, 25 agents each)
python run_concordia.py --agents 100 --steps 5 --multi-sim --max-per-sim 25

# Verbose output to see detailed logs
python run_concordia.py --agents 10 --steps 5 --verbose
```

**Benefits:**
- ✅ Follows official Concordia patterns
- ✅ Uses Entity and Game Master prefabs
- ✅ Implements three key questions reasoning
- ✅ Supports multi-simulation coordination for 100+ agents
- ✅ Better extensibility and component reusability

See [`CONCORDIA_GUIDE.md`](CONCORDIA_GUIDE.md) for detailed documentation.

---

### Original Implementations

### Option 1: Streamlit UI (Original 10-Agent Simulation)

```bash
# Start the Streamlit UI
streamlit run ui/app.py
```

Then open http://localhost:8501 in your browser.

### Option 2: 100-Agent Simulation (Command Line)

Run the large-scale simulation with economic inequality:

```bash
# Run with heuristic AI (fast, no API key needed)
python run_100_agents.py --ticks 48

# Run with LLM-powered agents (requires API key)
python run_100_agents.py --ticks 48 --use-llm --llm-provider openai

# Save simulation state
python run_100_agents.py --ticks 96 --save data/save_100.json

# Load and continue simulation
python run_100_agents.py --ticks 48 --load data/save_100.json

# See all options
python run_100_agents.py --help
```

### LLM Mode

1. Ensure your API key is configured
2. Check "Use LLM Mode (Concordia + AI reasoning)" in the UI
3. Click "Step" to run with full Concordia agent reasoning
4. View LLM prompts/responses and Concordia component logs at the bottom of the page

## Project Structure

```
island_sim/
├── data/
│   ├── island.yaml          # Original 6-location island
│   └── island_100.yaml      # Expanded 31-location island
├── sim/
│   ├── prefabs_entity.py    # ⭐ Concordia entity prefab (agents)
│   ├── prefabs_gm.py        # ⭐ Concordia GM prefab (world management)
│   ├── sim_concordia.py     # ⭐ Concordia simulation & multi-sim coordinator
│   ├── world.py             # Island world model and graph
│   ├── agents.py            # Agent configurations (10 agents)
│   ├── agents_100.py        # Agent configurations (100 agents)
│   ├── gm.py                # Original custom game master
│   └── sim.py               # Original simulation controller
├── ui/
│   └── app.py               # Streamlit dashboard
├── run_concordia.py         # ⭐ Run with Concordia prefabs (RECOMMENDED)
├── run_100_agents.py        # Run 100-agent simulation (original)
├── CONCORDIA_GUIDE.md       # ⭐ Detailed Concordia architecture guide
├── requirements.txt
└── README.md
```

**⭐ = New Concordia prefab architecture**

## 100-Agent Simulation Features

The large-scale simulation introduces **economic inequality** with a realistic power law distribution:

### Economic Classes

- **50 agents** - Lower Middle Class (Sunset Apartments)
  - Service workers, retail clerks, laborers
  - Live in a modest apartment complex
  - Work at cafes, stores, harbor, market, etc.

- **25 agents** - Middle Class (Coral Village)
  - Teachers, nurses, shop managers, craftspeople
  - Live in small single-family homes
  - Professional occupations and skilled trades

- **15 agents** - Upper Middle Class (Palm Heights)
  - Doctors, principals, business owners, artists
  - Live in spacious homes with gardens
  - Leadership and specialized professional roles

- **7 agents** - Upper Class (Ocean View Estates)
  - Specialists, wealthy investors, collectors
  - Live in large estates with ocean views
  - High-level positions and private investments

- **3 agents** - Elite (Paradise Point Mansions)
  - Heirs, philanthropists, tech entrepreneurs
  - Live in palatial mansions with extensive grounds
  - Generational wealth, minimal need to work

### Expanded Island (31 Locations)

**Residential Areas**: 12 locations across 5 economic tiers
**Shared Spaces**: Beach, church, restaurant, cafe, market, town square, park, tavern, library, art gallery, gym, cemetery
**Work/Services**: Clinic, town hall, school, harbor, workshop, general store, marina

The island graph has realistic travel times creating spatial inequality - elite estates are more isolated with longer commutes to amenities.

## Original 10-Agent Simulation

| Name | Home | Work | Personality |
|------|------|------|-------------|
| Marina | Beach | - | Friendly beach vendor who loves the ocean |
| Finn | Tavern | Tavern | Jovial tavern keeper with endless gossip |
| Rosa | Market | Market | Sharp-witted market vendor |
| Dr. Kai | Clinic | Clinic | Calm, methodical healer |
| Elder Mako | Temple | Temple | Wise temple keeper who speaks in riddles |
| Jade | Workshop | Workshop | Skilled, quiet craftsperson |
| Coral | Beach | - | Free-spirited artist |
| Rex | Tavern | - | Gruff retired sailor |
| Lily | Market | Clinic | Energetic young nurse |
| Sol | Temple | - | Mysterious wanderer |

## Architecture Comparison

### Concordia Prefab Architecture (NEW) ⭐

**Files:** `prefabs_entity.py`, `prefabs_gm.py`, `sim_concordia.py`, `run_concordia.py`

- ✅ **Proper Concordia prefabs** - Follows official patterns from `concordia/prefabs/`
- ✅ **Entity prefab** - "Three key questions" reasoning (SituationPerception, SelfPerception, PersonBySituation)
- ✅ **GM prefab** - Situated-in-time-and-place pattern with GenerativeClock, Locations, WorldState
- ✅ **Config/Role system** - Proper Concordia `Config` with `Role.ENTITY` and `Role.GAME_MASTER`
- ✅ **Component architecture** - Modular components that can be extended or swapped
- ✅ **Multi-simulation** - Coordinate multiple sims for 100+ agents
- ✅ **Extensible** - Easy to add new components, thought chains, or custom behaviors

**When to use:** New projects, learning Concordia, scaling to 100+ agents

### Original Custom Implementation

**Files:** `agents.py`, `gm.py`, `sim.py`, `run_100_agents.py`

- Custom `IslandAgent` and `GameMaster` classes
- Direct Concordia integration without prefab abstraction
- Intent-based movement system
- Heuristic fallback for non-LLM mode
- Streamlit UI integration

**When to use:** Existing projects, Streamlit UI, heuristic mode

---

## Built With

- [Concordia](https://github.com/google-deepmind/concordia) - Multi-agent simulation framework
- [Streamlit](https://streamlit.io/) - Web UI
- [Plotly](https://plotly.com/) - Visualization
- [NetworkX](https://networkx.org/) - Graph algorithms
- [OpenAI API](https://openai.com/) / [Anthropic API](https://anthropic.com/) - LLM backends

## Documentation

- **[CONCORDIA_GUIDE.md](CONCORDIA_GUIDE.md)** - Comprehensive guide to the new Concordia architecture
- **[README.md](README.md)** - This file, project overview
- **[requirements.txt](requirements.txt)** - Python dependencies
