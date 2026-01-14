"""
Streamlit UI for the Island Time-and-Place Concordia Simulation.

Features:
- Island map with agent positions
- Time display and controls
- Agent panels (location, needs, relationships)
- Event log with encounter history
- Replay slider for reviewing past states
"""

import sys
import asyncio
import json
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load API key from file if it exists
API_KEY_PATH = Path(__file__).parent.parent / "api_key.json"
if API_KEY_PATH.exists():
    with open(API_KEY_PATH) as f:
        api_data = json.load(f)
        if "API_KEY" in api_data:
            os.environ["OPENAI_API_KEY"] = api_data["API_KEY"]

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from sim.world import Island, tick_to_day_and_time, TICKS_PER_DAY
from sim.sim import IslandSimulation, SimulationConfig, create_simulation
from sim.gm import EncounterRecord


# Island place coordinates for visualization (hand-crafted layout)
PLACE_COORDS = {
    "beach": (0.1, 0.7),
    "market": (0.5, 0.5),
    "tavern": (0.7, 0.6),
    "clinic": (0.6, 0.3),
    "temple": (0.3, 0.2),
    "work_north": (0.8, 0.15),
}

PLACE_COLORS = {
    "social": "#4ECDC4",
    "work": "#FF6B6B",
    "quiet": "#9B59B6",
    "service": "#3498DB",
}

AGENT_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E91E63", "#00BCD4", "#FF5722", "#607D8B",
]


def init_session_state():
    """Initialize Streamlit session state."""
    if "sim" not in st.session_state:
        st.session_state.sim = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "selected_agent" not in st.session_state:
        st.session_state.selected_agent = None
    if "auto_run" not in st.session_state:
        st.session_state.auto_run = False
    if "replay_tick" not in st.session_state:
        st.session_state.replay_tick = None
    if "llm_logs" not in st.session_state:
        st.session_state.llm_logs = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = None


def create_island_map(sim: IslandSimulation, replay_tick: int | None = None) -> go.Figure:
    """Create a Plotly figure showing the island map with agents."""
    fig = go.Figure()

    island = sim.island

    # Draw edges (paths between places)
    for edge in island.graph.edges(data=True):
        p1, p2, data = edge
        if p1 in PLACE_COORDS and p2 in PLACE_COORDS:
            x1, y1 = PLACE_COORDS[p1]
            x2, y2 = PLACE_COORDS[p2]
            weight = data.get("weight", 1)

            fig.add_trace(go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(color="#BDC3C7", width=2 + weight),
                hoverinfo="text",
                hovertext=f"{weight * 30} min",
                showlegend=False,
            ))

    # Draw places
    for place_id, place in island.places.items():
        if place_id not in PLACE_COORDS:
            continue

        x, y = PLACE_COORDS[place_id]
        color = PLACE_COLORS.get(place.type, "#95A5A6")

        # Check if open
        current_tick = replay_tick if replay_tick is not None else sim.gm.current_tick
        is_open = place.is_open(current_tick)
        opacity = 1.0 if is_open else 0.4

        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode="markers+text",
            marker=dict(size=40, color=color, opacity=opacity, line=dict(width=2, color="white")),
            text=[place.name],
            textposition="bottom center",
            textfont=dict(size=11, color="white"),
            hoverinfo="text",
            hovertext=f"{place.name}<br>Type: {place.type}<br>Hours: {place.open[0]}:00-{place.open[1]}:00<br>{'OPEN' if is_open else 'CLOSED'}",
            showlegend=False,
        ))

    # Draw agents
    agent_positions = _get_agent_positions(sim, replay_tick)

    for i, (agent_name, pos_data) in enumerate(agent_positions.items()):
        x, y = pos_data["coords"]
        color = AGENT_COLORS[i % len(AGENT_COLORS)]

        # Offset agents slightly so they don't overlap
        offset_x = (i % 3 - 1) * 0.03
        offset_y = (i // 3 - 1) * 0.03

        marker_symbol = "circle" if pos_data["available"] else "diamond"
        opacity = 1.0 if pos_data["available"] else 0.6

        fig.add_trace(go.Scatter(
            x=[x + offset_x],
            y=[y + offset_y],
            mode="markers+text",
            marker=dict(
                size=20,
                color=color,
                symbol=marker_symbol,
                opacity=opacity,
                line=dict(width=2, color="white"),
            ),
            text=[agent_name.split()[0]],  # First name only
            textposition="top center",
            textfont=dict(size=9, color=color),
            hoverinfo="text",
            hovertext=f"{agent_name}<br>{pos_data['status']}",
            showlegend=False,
        ))

    # Layout
    fig.update_layout(
        title=None,
        xaxis=dict(visible=False, range=[-0.1, 1.1]),
        yaxis=dict(visible=False, range=[-0.1, 1.0]),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#1a1a2e",
        margin=dict(l=10, r=10, t=10, b=10),
        height=400,
    )

    return fig


def _get_agent_positions(sim: IslandSimulation, replay_tick: int | None = None) -> dict:
    """Get agent positions for map display."""
    positions = {}

    for agent in sim.agents:
        if agent.in_transit_to and agent.in_transit_ticks > 0:
            # Interpolate position during transit
            from_place = agent.current_location
            to_place = agent.in_transit_to

            if from_place in PLACE_COORDS and to_place in PLACE_COORDS:
                x1, y1 = PLACE_COORDS[from_place]
                x2, y2 = PLACE_COORDS[to_place]

                # Calculate progress
                total_ticks = sim.island.travel_ticks(from_place, to_place)
                progress = 1 - (agent.in_transit_ticks / max(total_ticks, 1))

                x = x1 + (x2 - x1) * progress
                y = y1 + (y2 - y1) * progress

                positions[agent.name] = {
                    "coords": (x, y),
                    "available": False,
                    "status": f"Traveling to {to_place} ({agent.in_transit_ticks * 30} min left)",
                }
            else:
                positions[agent.name] = {
                    "coords": PLACE_COORDS.get(from_place, (0.5, 0.5)),
                    "available": False,
                    "status": f"In transit",
                }
        else:
            loc = agent.current_location
            coords = PLACE_COORDS.get(loc, (0.5, 0.5))
            place = sim.island.get_place(loc)
            place_name = place.name if place else loc

            positions[agent.name] = {
                "coords": coords,
                "available": True,
                "status": f"At {place_name}",
            }

    return positions


def render_agent_panel(sim: IslandSimulation, agent_name: str):
    """Render detailed agent information panel."""
    info = sim.get_agent_info(agent_name)
    if not info:
        st.warning(f"Agent {agent_name} not found")
        return

    agent_idx = next(
        (i for i, a in enumerate(sim.agents) if a.name == agent_name),
        0
    )
    color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]

    st.markdown(f"### <span style='color:{color}'>{info['name']}</span>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Location**")
        if info["in_transit"]:
            st.write(f"Traveling to {info['in_transit']}")
        else:
            st.write(f"{info['location']}")

        st.markdown("**Home / Work**")
        st.write(f"{info['home']} / {info['work'] or 'None'}")

    with col2:
        st.markdown("**Personality**")
        st.write(info["personality"][:100] + "..." if len(info["personality"]) > 100 else info["personality"])

    # Needs bars
    st.markdown("**Needs**")
    needs = info["needs"]

    for need_name, value in needs.items():
        color_need = "#E74C3C" if value > 60 else "#F39C12" if value > 40 else "#2ECC71"
        st.markdown(
            f"""
            <div style="margin-bottom: 5px;">
                <span style="width: 70px; display: inline-block;">{need_name.capitalize()}</span>
                <div style="display: inline-block; width: 150px; background: #333; border-radius: 5px; height: 15px;">
                    <div style="width: {value}%; background: {color_need}; height: 100%; border-radius: 5px;"></div>
                </div>
                <span style="margin-left: 10px;">{value:.0f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Relationships
    if info["relationships"]:
        st.markdown("**Relationships**")
        for other, affinity in sorted(info["relationships"].items(), key=lambda x: -x[1])[:5]:
            rel_color = "#2ECC71" if affinity > 0 else "#E74C3C"
            st.markdown(f"<span style='color:{rel_color}'>{other}: {affinity:+.0f}</span>", unsafe_allow_html=True)


def render_event_log(sim: IslandSimulation, limit: int = 10):
    """Render the event log with dialogue."""
    events = sim.get_recent_events(limit=limit)

    if not events:
        st.info("No events yet. Run the simulation to see encounters.")
        return

    for event in reversed(events):
        day, time_str = tick_to_day_and_time(event["tick"])
        participants = ", ".join(event["participants"])

        # Build dialogue HTML if available
        dialogue_html = ""
        if event.get("dialogue"):
            dialogue_lines = []
            for speaker, utterance in event["dialogue"]:
                dialogue_lines.append(
                    f'<div style="margin-left: 10px; color: #aaa;"><b style="color: #4ECDC4;">{speaker}:</b> "{utterance}"</div>'
                )
            dialogue_html = "".join(dialogue_lines)

        st.markdown(
            f"""
            <div style="background: #2a2a4a; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <div style="color: #888; font-size: 0.8em;">Day {day}, {time_str} - {event['place']}</div>
                <div style="color: #4ECDC4; font-size: 0.9em;">{participants}</div>
                <div style="color: #fff; margin: 5px 0;">{event['summary']}</div>
                {dialogue_html}
            </div>
            """,
            unsafe_allow_html=True
        )


def render_decision_log(sim: IslandSimulation):
    """Render agent decision log."""
    decisions = sim.get_decision_log(limit=15)

    if not decisions:
        st.info("No decisions yet. Step the simulation to see agent reasoning.")
        return

    # Group by tick
    by_tick = {}
    for d in decisions:
        tick = d["tick"]
        if tick not in by_tick:
            by_tick[tick] = []
        by_tick[tick].append(d)

    for tick in sorted(by_tick.keys(), reverse=True)[:5]:
        items = by_tick[tick]
        time_str = items[0]["time"] if items else ""

        st.markdown(f"**Tick {tick} ({time_str})**")

        for item in items:
            is_movement = item.get("is_movement", False)
            icon = "🚶" if is_movement else "💭"
            color = "#4ECDC4" if is_movement else "#aaa"

            st.markdown(
                f"<div style='font-size: 0.85em; color: {color};'>{icon} <b>{item['agent']}</b>: {item['decision']}</div>",
                unsafe_allow_html=True
            )
        st.markdown("---")


def render_llm_status(sim: IslandSimulation):
    """Render LLM connection status."""
    status = sim.get_llm_status()

    if status["enabled"]:
        if status["connected"]:
            st.markdown(
                f"""<div style="background: #1a4a1a; padding: 8px; border-radius: 5px; font-size: 0.85em;">
                    🟢 <b>LLM Active</b>: {status['provider']} / {status['model']}
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style="background: #4a1a1a; padding: 8px; border-radius: 5px; font-size: 0.85em;">
                    🔴 <b>LLM Error</b>: {status['provider']} not connected (check API key)
                </div>""",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """<div style="background: #3a3a3a; padding: 8px; border-radius: 5px; font-size: 0.85em;">
                ⚪ <b>Heuristic Mode</b>: LLM disabled (simple AI)
            </div>""",
            unsafe_allow_html=True
        )


def render_llm_logs(sim: IslandSimulation):
    """Render LLM call logs showing prompts and responses."""
    logs = sim.get_llm_logs(limit=5)

    if not logs:
        st.info("No LLM calls yet. Step the simulation to see LLM reasoning.")
        return

    for log in reversed(logs):
        participants = ", ".join(log["participants"]) if log["participants"] else "N/A"
        is_error = log["response"].startswith("[Error") or log["response"].startswith("[No LLM")

        # Response preview (first 200 chars)
        response_preview = log["response"][:300] + "..." if len(log["response"]) > 300 else log["response"]

        with st.expander(f"Tick {log['tick']} - {log['type'].upper()} ({participants})", expanded=False):
            st.markdown("**Prompt sent to LLM:**")
            st.code(log["prompt"], language=None)

            st.markdown("**LLM Response:**")
            if is_error:
                st.error(log["response"])
            else:
                st.success(log["response"])


def render_concordia_logs(sim: IslandSimulation):
    """Render Concordia component logs showing agent reasoning process."""
    logs = sim.get_concordia_logs(limit=10)

    if not logs:
        st.info("No Concordia component logs yet. Use async mode (LLM) to see agent reasoning.")
        return

    # Group by tick for cleaner display
    by_tick = {}
    for log in logs:
        tick = log["tick"]
        if tick not in by_tick:
            by_tick[tick] = []
        by_tick[tick].append(log)

    for tick in sorted(by_tick.keys(), reverse=True)[:3]:
        tick_logs = by_tick[tick]

        with st.expander(f"Tick {tick} - Agent Reasoning ({len(tick_logs)} agents)", expanded=False):
            for log in tick_logs:
                # Agent header
                st.markdown(
                    f"**{log['agent']}** decided: `{log['intent_type']}` - *{log['intent_reason'] or 'no reason given'}*"
                )

                # Component outputs
                if log["components"]:
                    for channel, output in log["components"].items():
                        # Format channel name
                        channel_display = channel.replace("__", "").replace("_", " ").title()

                        # Truncate long outputs
                        if output:
                            output_display = str(output)[:500]
                            if len(str(output)) > 500:
                                output_display += "..."

                            st.markdown(f"*{channel_display}:*")
                            st.code(output_display, language=None)
                else:
                    st.markdown("*No component outputs captured*")

                st.markdown("---")


def run_async_step(sim: IslandSimulation):
    """Run a single async step with LLM/Concordia."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(sim.step_async())
    finally:
        loop.close()


def render_time_controls(sim: IslandSimulation):
    """Render time display and simulation controls."""
    time_info = sim.get_time_info()

    # Time display
    st.markdown(
        f"""
        <div style="text-align: center; padding: 20px; background: #2a2a4a; border-radius: 10px; margin-bottom: 20px;">
            <div style="font-size: 2.5em; color: #4ECDC4; font-weight: bold;">
                Day {time_info['day']} - {time_info['time']}
            </div>
            <div style="color: #888;">Tick {time_info['tick']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Mode toggle
    use_llm_mode = st.checkbox(
        "Use LLM Mode (Concordia + AI reasoning)",
        value=st.session_state.get("use_llm_mode", False),
        key="llm_mode_toggle",
        help="Enable to use full Concordia agent pipeline with LLM. Requires API key."
    )
    st.session_state.use_llm_mode = use_llm_mode

    # Control buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Step", use_container_width=True, type="primary"):
            if use_llm_mode and sim.model:
                with st.spinner("Running LLM step..."):
                    result = run_async_step(sim)
            else:
                result = sim.step()
            st.session_state.history.append(result)
            st.rerun()

    with col2:
        if st.button("Step x5", use_container_width=True):
            with st.spinner("Running 5 steps..."):
                for _ in range(5):
                    if use_llm_mode and sim.model:
                        result = run_async_step(sim)
                    else:
                        result = sim.step()
                    st.session_state.history.append(result)
            st.rerun()

    with col3:
        if st.button("Step x24", use_container_width=True):
            with st.spinner("Running 24 steps..."):
                for _ in range(24):
                    if use_llm_mode and sim.model:
                        result = run_async_step(sim)
                    else:
                        result = sim.step()
                    st.session_state.history.append(result)
            st.rerun()

    with col4:
        if st.button("Reset", use_container_width=True):
            st.session_state.sim = None
            st.session_state.history = []
            st.rerun()


def render_place_summary(sim: IslandSimulation):
    """Render a summary of all places and their occupants."""
    st.markdown("### Places")

    for place_id in sim.island.all_place_ids():
        info = sim.get_place_info(place_id)
        if not info:
            continue

        color = PLACE_COLORS.get(info["type"], "#95A5A6")
        status = "OPEN" if info["is_open"] else "CLOSED"
        status_color = "#2ECC71" if info["is_open"] else "#E74C3C"

        agents_str = ", ".join(info["agents"]) if info["agents"] else "Empty"

        st.markdown(
            f"""
            <div style="background: #2a2a4a; padding: 10px; border-radius: 5px; margin-bottom: 8px; border-left: 4px solid {color};">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: {color}; font-weight: bold;">{info['name']}</span>
                    <span style="color: {status_color}; font-size: 0.8em;">{status}</span>
                </div>
                <div style="color: #aaa; font-size: 0.85em;">{agents_str}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_agents_list(sim: IslandSimulation):
    """Render list of all agents for selection."""
    st.markdown("### Agents")

    for i, agent in enumerate(sim.agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        info = sim.get_agent_info(agent.name)

        location = info["location"] if info else agent.current_location
        if agent.in_transit_to:
            location = f"-> {agent.in_transit_to}"

        urgent_need = agent.needs.most_urgent()
        need_value = getattr(agent.needs, urgent_need)
        need_color = "#E74C3C" if need_value > 60 else "#F39C12" if need_value > 40 else "#2ECC71"

        is_selected = st.session_state.selected_agent == agent.name

        if st.button(
            f"{agent.name}",
            key=f"agent_{agent.name}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            st.session_state.selected_agent = agent.name
            st.rerun()

        # Show brief status below button
        st.markdown(
            f"<div style='font-size: 0.75em; color: #888; margin-top: -10px; margin-bottom: 10px;'>"
            f"{location} | <span style='color:{need_color}'>{urgent_need}</span>"
            f"</div>",
            unsafe_allow_html=True
        )


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Island Simulation",
        page_icon="🌴",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #1a1a2e;
            color: #eee;
        }
        .stButton > button {
            background-color: #4ECDC4;
            color: #1a1a2e;
            border: none;
        }
        .stButton > button:hover {
            background-color: #3dbdb5;
        }
        .stButton > button[kind="secondary"] {
            background-color: #2a2a4a;
            color: #eee;
        }
        div[data-testid="stSidebar"] {
            background-color: #16213e;
        }
        .stMarkdown h3 {
            color: #4ECDC4;
        }
        </style>
    """, unsafe_allow_html=True)

    init_session_state()

    # Header
    st.markdown(
        """
        <h1 style="text-align: center; color: #4ECDC4;">
            🌴 Island Time-and-Place Simulation
        </h1>
        <p style="text-align: center; color: #888;">
            A Concordia-powered multi-agent social simulation
        </p>
        """,
        unsafe_allow_html=True
    )

    # Initialize simulation if needed
    if st.session_state.sim is None:
        with st.spinner("Initializing simulation..."):
            try:
                sim = create_simulation(
                    island_path="data/island.yaml",
                    use_llm=True,
                    start_tick=16,
                )
                st.session_state.sim = sim
                st.rerun()
            except Exception as e:
                st.error(f"Failed to initialize simulation: {e}")
                st.stop()

    sim = st.session_state.sim

    # Sidebar
    with st.sidebar:
        # API Key Input
        st.markdown("### OpenAI API Key")
        api_key = st.text_input(
            "API Key",
            value=st.session_state.api_key or "",
            type="password",
            placeholder="sk-...",
            key="api_key_input"
        )
        if api_key and api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            import os
            os.environ["OPENAI_API_KEY"] = api_key
            # Reset simulation to use new key
            st.session_state.sim = None
            st.rerun()

        st.markdown("---")

        render_agents_list(sim)

        st.markdown("---")

        render_place_summary(sim)

        st.markdown("---")

        # Save/Load
        st.markdown("### Save / Load")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save", use_container_width=True):
                try:
                    sim.save(Path("data/quicksave.json"))
                    st.success("Saved!")
                except Exception as e:
                    st.error(f"Save failed: {e}")

        with col2:
            if st.button("Load", use_container_width=True):
                try:
                    sim.load(Path("data/quicksave.json"))
                    st.success("Loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Load failed: {e}")

        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
            "Built with Concordia + Claude"
            "</div>",
            unsafe_allow_html=True
        )

    # Main content
    render_time_controls(sim)

    # LLM Status
    render_llm_status(sim)
    st.markdown("")

    # Layout: Map and Agent Panel
    col_map, col_panel = st.columns([2, 1])

    with col_map:
        st.markdown("### Island Map")
        fig = create_island_map(sim)
        st.plotly_chart(fig, use_container_width=True)

        # Legend
        st.markdown(
            """
            <div style="display: flex; gap: 20px; justify-content: center; font-size: 0.8em; color: #888;">
                <span>🟢 Social</span>
                <span>🔴 Work</span>
                <span>🟣 Quiet</span>
                <span>🔵 Service</span>
                <span>◆ In Transit</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_panel:
        if st.session_state.selected_agent:
            render_agent_panel(sim, st.session_state.selected_agent)
        else:
            st.info("Click an agent in the sidebar to see details")

    # Event Log and Decision Log side by side
    st.markdown("---")
    col_events, col_decisions = st.columns([1, 1])

    with col_events:
        st.markdown("### Encounters (LLM Narration)")
        render_event_log(sim, limit=5)

    with col_decisions:
        st.markdown("### Agent Decisions")
        render_decision_log(sim)

    # LLM Logs section
    st.markdown("---")
    col_llm, col_concordia = st.columns([1, 1])

    with col_llm:
        st.markdown("### LLM Prompts & Responses")
        render_llm_logs(sim)

    with col_concordia:
        st.markdown("### Concordia Component Logs")
        render_concordia_logs(sim)

    # Replay slider (if we have history)
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Replay")

        max_tick = sim.gm.current_tick
        min_tick = max(0, max_tick - len(st.session_state.history))

        replay_tick = st.slider(
            "Tick",
            min_value=min_tick,
            max_value=max_tick,
            value=max_tick,
            key="replay_slider",
        )

        if replay_tick != max_tick:
            st.info(f"Viewing tick {replay_tick} (current is {max_tick})")


if __name__ == "__main__":
    main()
