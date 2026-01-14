# Island Time-and-Place Concordia Simulation

A spatial, time-based multi-agent simulation where 10 LLM-powered agents live on an island. Built with Google DeepMind's Concordia framework.

## Features

- **10 Unique Agents**: Each with personality, home, workplace, needs (hunger, fatigue, social), and relationships
- **Island World**: 6 locations (beach, tavern, market, clinic, temple, workshop) connected by paths with travel times
- **Time System**: 30-minute ticks, 48 ticks per day, with day/night cycles and open hours
- **Concordia Integration**: Agents use EntityAgentWithLogging for memory, observations, and LLM-powered decisions
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

Create `api_key.json` in the project root:
```json
{"API_KEY": "your-openai-api-key"}
```

## Running

```bash
# Start the Streamlit UI
streamlit run ui/app.py
```

Then open http://localhost:8501 in your browser.

### LLM Mode

1. Ensure your API key is configured
2. Check "Use LLM Mode (Concordia + AI reasoning)" in the UI
3. Click "Step" to run with full Concordia agent reasoning
4. View LLM prompts/responses and Concordia component logs at the bottom of the page

## Project Structure

```
island_sim/
├── data/
│   └── island.yaml       # Island configuration (places, paths)
├── sim/
│   ├── world.py          # Island world model and graph
│   ├── agents.py         # Concordia-powered agents
│   ├── gm.py             # Game Master (movement, encounters)
│   └── sim.py            # Main simulation controller
├── ui/
│   └── app.py            # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Agents

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

## Built With

- [Concordia](https://github.com/google-deepmind/concordia) - Multi-agent simulation framework
- [Streamlit](https://streamlit.io/) - Web UI
- [Plotly](https://plotly.com/) - Visualization
- [NetworkX](https://networkx.org/) - Graph algorithms
- [OpenAI API](https://openai.com/) - LLM backend
