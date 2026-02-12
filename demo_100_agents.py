#!/usr/bin/env python3
"""
Demo script showing the 100-agent Concordia simulation architecture.
Uses mock LLM responses to demonstrate the structure without API costs.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from sim.sim_concordia import (
    create_island_config,
    create_neighborhood_simulations,
    MultiIslandSimulation,
)
from sim.agents_100 import ISLAND_100_AGENTS
from sim.world import Island


class MockLanguageModel:
    """Mock LLM that returns simple responses for demo purposes."""

    def __init__(self):
        self.call_count = 0

    def sample_text(
        self,
        prompt: str,
        **kwargs  # Accept any additional keyword arguments
    ) -> str:
        """Concordia-compatible sample_text method."""
        return self.__call__(prompt)

    def __call__(self, prompt: str) -> str:
        self.call_count += 1

        # Simple mock responses based on prompt content
        if "situation" in prompt.lower():
            return "The agent is going about their daily routine on the island."
        elif "kind of person" in prompt.lower():
            return "A typical island resident with their own personality and goals."
        elif "what would" in prompt.lower():
            return "They would continue with their planned activities."
        elif "time" in prompt.lower():
            return "Day 1, 08:00"
        elif "location" in prompt.lower():
            return "At their home or workplace"
        else:
            return "The agent observes their surroundings and decides on their next action."


def create_mock_embedder():
    """Create a simple embedder."""
    import hashlib

    def embedder(text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).hexdigest()
        embedding = [int(h[i:i+2], 16) / 255.0 for i in range(0, 64, 2)]
        return np.array(embedding, dtype=np.float32)

    return embedder


def main():
    print("=" * 70)
    print("  100-Agent Island Simulation - Architecture Demo")
    print("=" * 70)
    print()
    print("This demo shows the Concordia prefab architecture without API calls.")
    print("Using mock LLM responses for demonstration purposes.")
    print()

    # Load island
    print("Loading island configuration...")
    island = Island.load(Path("data/island_100.yaml"))
    print(f"  ✓ Island loaded: {len(island.places)} locations, {len(island.graph.edges)} connections")
    print()

    # Show agent distribution
    print("Agent Distribution:")
    from collections import Counter
    home_dist = Counter(agent.home_place for agent in ISLAND_100_AGENTS)

    print(f"  Total agents: {len(ISLAND_100_AGENTS)}")
    print(f"  Unique home locations: {len(home_dist)}")
    print()

    # Create mock model
    print("Initializing mock language model and embedder...")
    model = MockLanguageModel()
    embedder = create_mock_embedder()
    print("  ✓ Mock model initialized")
    print()

    # Create neighborhood simulations
    max_per_sim = 25
    print(f"Creating multi-simulation system (max {max_per_sim} agents per sim)...")
    neighborhoods = create_neighborhood_simulations(
        all_agents=ISLAND_100_AGENTS,
        island=island,
        model=model,
        embedder=embedder,
        max_agents_per_sim=max_per_sim,
    )

    print(f"  ✓ Created {len(neighborhoods)} neighborhood simulations:")
    for i, neighborhood in enumerate(neighborhoods, 1):
        print(f"    {i}. {neighborhood.name[:40]:40s} - {len(neighborhood.agent_names):2d} agents")
    print()

    # Show details of first simulation
    print("Example: First Simulation Details")
    print("-" * 70)
    first_sim = neighborhoods[0].simulation
    entities = first_sim.get_entities()
    gms = first_sim.get_game_masters()

    print(f"  Entities: {len(entities)}")
    print(f"  Game Masters: {len(gms)}")
    print()
    print(f"  Sample agents in this simulation:")
    for entity in entities[:5]:
        print(f"    - {entity.name}")
    if len(entities) > 5:
        print(f"    ... and {len(entities) - 5} more")
    print()

    # Show GM components
    if gms:
        gm = gms[0]
        print(f"  Game Master: {gm.name}")
        print(f"  GM Components:")

        # Try to access component structure
        try:
            components = gm._context_components
            for key in list(components.keys())[:10]:
                print(f"    - {key}")
            if len(components) > 10:
                print(f"    ... and {len(components) - 10} more components")
        except:
            print("    (Component details not accessible in this view)")
    print()

    # Show island locations by type
    print("Island Locations by Type:")
    print("-" * 70)
    from collections import defaultdict
    locations_by_type = defaultdict(list)
    for place_id, place in island.places.items():
        locations_by_type[place.type].append(place.name)

    for place_type, places in sorted(locations_by_type.items()):
        print(f"  {place_type:20s} ({len(places):2d}): {', '.join(places[:3])}", end='')
        if len(places) > 3:
            print(f" ... +{len(places)-3} more")
        else:
            print()
    print()

    # Create multi-sim coordinator
    print("Multi-Simulation Architecture:")
    print("-" * 70)
    multi_sim = MultiIslandSimulation(neighborhoods=neighborhoods)

    print(f"  Total agents across all sims: {sum(len(n.agent_names) for n in neighborhoods)}")
    print(f"  Number of parallel simulations: {len(neighborhoods)}")
    print(f"  Average agents per simulation: {sum(len(n.agent_names) for n in neighborhoods) / len(neighborhoods):.1f}")
    print()

    # Show economic distribution
    print("Economic Distribution Across Simulations:")
    print("-" * 70)

    residential_types = {
        'sunset_apartments': 'Lower Middle Class',
        'coral_village_1': 'Middle Class',
        'coral_village_2': 'Middle Class',
        'palm_heights_1': 'Upper Middle Class',
        'palm_heights_2': 'Upper Middle Class',
        'ocean_view_1': 'Upper Class',
        'ocean_view_2': 'Upper Class',
        'ocean_view_3': 'Upper Class',
        'ocean_view_4': 'Upper Class',
        'paradise_point_1': 'Elite',
        'paradise_point_2': 'Elite',
        'paradise_point_3': 'Elite',
    }

    class_counts = Counter()
    for agent in ISLAND_100_AGENTS:
        econ_class = residential_types.get(agent.home_place, 'Other')
        class_counts[econ_class] += 1

    for econ_class, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        bar = '█' * (count // 2)
        print(f"  {econ_class:20s} {count:3d} agents {bar}")
    print()

    # Summary
    print("=" * 70)
    print("Architecture Summary")
    print("=" * 70)
    print()
    print("✓ Concordia Prefab Pattern:")
    print("  - IslandEntity prefab (3 key questions reasoning)")
    print("  - IslandGameMaster prefab (situated-in-time-and-place)")
    print("  - Config/Role-based setup (proper Concordia architecture)")
    print()
    print("✓ Entity Components (per agent):")
    print("  - Memory, Observation, Instructions, Goal, Personality")
    print("  - SituationPerception, SelfPerception, PersonBySituation")
    print("  - RelevantMemories (semantic retrieval)")
    print()
    print("✓ GM Components (per simulation):")
    print("  - GenerativeClock, Locations, WorldState")
    print("  - EventResolution (with thought chains)")
    print("  - MakeObservation, NextActing, NextActionSpec")
    print("  - Memory, RelevantMemories, DisplayEvents")
    print()
    print("✓ Multi-Simulation Scaling:")
    print(f"  - {len(neighborhoods)} parallel simulations")
    print(f"  - {max_per_sim} agents max per simulation")
    print(f"  - Neighborhood-based agent distribution")
    print()
    print("✓ Economic Inequality Model:")
    print("  - Power law wealth distribution")
    print("  - Spatial inequality (elite estates more isolated)")
    print("  - 5 economic classes from lower-middle to elite")
    print()
    print(f"Mock LLM calls made: {model.call_count}")
    print()
    print("To run with real LLM:")
    print("  export OPENAI_API_KEY='your-key'")
    print("  python run_concordia.py --agents 100 --steps 5 --multi-sim")
    print()


if __name__ == "__main__":
    main()
