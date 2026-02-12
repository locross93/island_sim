#!/usr/bin/env python3
"""
Run the Island Simulation using proper Concordia prefabs and patterns.

This script demonstrates the idiomatic Concordia approach with:
- Entity prefabs (three key questions pattern)
- Game Master prefabs (situated_in_time_and_place pattern)
- Proper Config/Role-based simulation setup
- Multi-simulation support for scaling to 100+ agents
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sim.sim_concordia import (
    create_island_config,
    IslandSimulationConcordia,
    create_neighborhood_simulations,
    MultiIslandSimulation,
)
from sim.agents import DEFAULT_ISLAND_AGENTS, ScheduleBias
from sim.agents_100 import ISLAND_100_AGENTS
from sim.world import Island


def create_simple_model():
    """Create a simple language model wrapper for OpenAI or Anthropic."""
    import os

    # Try to get API key from environment
    openai_key = os.environ.get('OPENAI_API_KEY')
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')

    if openai_key:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)

        class OpenAIModel:
            def __init__(self, client, model="gpt-4.1-nano"):
                self.client = client
                self.model = model

            def __call__(self, prompt: str) -> str:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                return response.choices[0].message.content

        return OpenAIModel(client)

    elif anthropic_key:
        from anthropic import Anthropic
        client = Anthropic(api_key=anthropic_key)

        class AnthropicModel:
            def __init__(self, client, model="claude-3-5-haiku-20241022"):
                self.client = client
                self.model = model

            def __call__(self, prompt: str) -> str:
                response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                return response.content[0].text

        return AnthropicModel(client)

    else:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        )


def create_simple_embedder():
    """Create a sentence embedder."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return lambda text: model.encode(text)
    except ImportError:
        import hashlib
        import numpy as np

        def dummy_embedder(text: str) -> np.ndarray:
            """Create a deterministic but non-semantic embedding."""
            h = hashlib.sha256(text.encode()).hexdigest()
            embedding = [int(h[i:i+2], 16) / 255.0 for i in range(0, 64, 2)]
            return np.array(embedding, dtype=np.float32)

        print("Warning: Using dummy embedder. Install sentence-transformers for better memory.")
        return dummy_embedder


def main():
    parser = argparse.ArgumentParser(
        description="Run Island Simulation with Concordia Prefabs"
    )
    parser.add_argument(
        "--agents",
        type=int,
        choices=[10, 100],
        default=10,
        help="Number of agents (10 or 100)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of simulation steps to run"
    )
    parser.add_argument(
        "--multi-sim",
        action="store_true",
        help="Use multiple simulations for 100 agents (recommended)"
    )
    parser.add_argument(
        "--max-per-sim",
        type=int,
        default=25,
        help="Maximum agents per simulation when using --multi-sim"
    )
    parser.add_argument(
        "--island",
        type=str,
        help="Path to island YAML file (auto-selected based on agent count if not specified)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  Island Simulation - Concordia Prefabs Architecture")
    print("=" * 70)
    print()

    # Select agents and island
    if args.agents == 10:
        agents = DEFAULT_ISLAND_AGENTS
        island_path = args.island or "data/island.yaml"
        print(f"Using 10-agent configuration")
    else:
        agents = ISLAND_100_AGENTS
        island_path = args.island or "data/island_100.yaml"
        print(f"Using 100-agent configuration with economic inequality")

    # Load island
    print(f"Loading island from {island_path}...")
    island = Island.load(Path(island_path))
    print(f"  - {len(island.places)} locations")
    print(f"  - {len(island.graph.edges)} connections")
    print()

    # Create model and embedder
    print("Initializing language model and embedder...")
    try:
        model = create_simple_model()
        print("  ✓ Language model initialized")
    except ValueError as e:
        print(f"  ✗ Error: {e}")
        return

    embedder = create_simple_embedder()
    print("  ✓ Embedder initialized")
    print()

    # Decide on simulation mode
    use_multi_sim = args.multi_sim or (args.agents == 100 and len(agents) > 50)

    if use_multi_sim:
        print(f"Creating multi-simulation system (max {args.max_per_sim} agents per sim)...")
        neighborhoods = create_neighborhood_simulations(
            all_agents=agents,
            island=island,
            model=model,
            embedder=embedder,
            max_agents_per_sim=args.max_per_sim,
        )
        print(f"  ✓ Created {len(neighborhoods)} neighborhood simulations:")
        for neighborhood in neighborhoods:
            print(f"    - {neighborhood.name}: {len(neighborhood.agent_names)} agents")
        print()

        multi_sim = MultiIslandSimulation(neighborhoods=neighborhoods)

        print(f"Running {args.steps} steps across all simulations...")
        print("-" * 70)

        results = multi_sim.run_all(num_steps=args.steps)

        print()
        print("=" * 70)
        print("Simulation Complete!")
        print("=" * 70)
        print()
        print("Results by neighborhood:")
        for name, neighborhood_results in results.items():
            print(f"\n{name}:")
            if args.verbose:
                for i, result in enumerate(neighborhood_results, 1):
                    print(f"  Step {i}: {result[:100]}...")
            else:
                print(f"  {len(neighborhood_results)} steps completed")

    else:
        print(f"Creating single simulation with {len(agents)} agents...")
        config = create_island_config(
            agents=agents,
            island=island,
            gm_name="Island Game Master",
            start_tick=16,  # 8:00 AM
        )

        sim = IslandSimulationConcordia(
            config=config,
            model=model,
            embedder=embedder,
            island=island,
        )
        print(f"  ✓ Simulation initialized")
        print(f"  - Entities: {len(sim.get_entities())}")
        print(f"  - Game Masters: {len(sim.get_game_masters())}")
        print()

        print(f"Running {args.steps} steps...")
        print("-" * 70)

        results = sim.run(num_steps=args.steps)

        print()
        print("=" * 70)
        print("Simulation Complete!")
        print("=" * 70)
        print()
        if args.verbose:
            print("Results:")
            for i, result in enumerate(results, 1):
                print(f"\nStep {i}:")
                print(result)
        else:
            print(f"Completed {len(results)} steps successfully.")

    print()
    print("Concordia Architecture Features Used:")
    print("  ✓ Entity Prefab (three key questions pattern)")
    print("  ✓ Game Master Prefab (situated_in_time_and_place pattern)")
    print("  ✓ Proper Config/Role-based setup")
    print("  ✓ AssociativeMemory with semantic embeddings")
    print("  ✓ Component-based agent architecture")
    if use_multi_sim:
        print("  ✓ Multi-simulation coordination")
    print()


if __name__ == "__main__":
    main()
