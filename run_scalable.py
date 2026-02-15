#!/usr/bin/env python3
"""
Run the Scalable Island Simulation with location-based efficiency.

This script demonstrates the scalable simulation architecture with:
- Parallel location observations (1 per occupied location)
- Parallel movement decisions (all agents simultaneously)
- Sequential dialogue only for co-located agents

Usage:
    # Quick prototype with 5 agents
    python run_scalable.py --agents 5 --steps 2

    # Medium scale
    python run_scalable.py --agents 20 --steps 10

    # Full scale
    python run_scalable.py --agents 100 --steps 20

    # With logging
    python run_scalable.py --agents 10 --log-llm --verbose
"""

import argparse
import json
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sim.scalable_sim import create_scalable_simulation, ScalableSimulation
from sim.agents import DEFAULT_ISLAND_AGENTS, AgentConfig
from sim.agents_100 import ISLAND_100_AGENTS
from sim.world import Island


def create_simple_model():
    """Create a simple language model wrapper for OpenAI or Anthropic."""
    import os

    # Try to get API key from environment
    openai_key = os.environ.get('OPENAI_API_KEY')
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')

    # If not in environment, try api_key.json
    if not openai_key and not anthropic_key:
        try:
            with open('api_key.json', 'r') as f:
                data = json.load(f)
                openai_key = data.get('API_KEY') or data.get('OPENAI_API_KEY')
                if not anthropic_key:
                    anthropic_key = data.get('ANTHROPIC_API_KEY')
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            print("Warning: api_key.json found but could not be parsed")

    if openai_key:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)

        class OpenAIModel:
            def __init__(self, client, model="gpt-4.1-nano"):
                self.client = client
                self.model = model
                self.call_log = []
                self.verbose = False

            def __call__(self, prompt: str) -> str:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                result = response.choices[0].message.content

                self.call_log.append({
                    "prompt": prompt,
                    "response": result,
                    "model": self.model
                })

                if self.verbose:
                    print("\n" + "="*60)
                    print("LLM PROMPT:")
                    print("-"*60)
                    print(prompt[:1500] + ("..." if len(prompt) > 1500 else ""))
                    print("-"*60)
                    print("LLM RESPONSE:", result[:200])
                    print("="*60)

                return result

            def save_log(self, path: str):
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self.call_log, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(self.call_log)} LLM calls to {path}")

        return OpenAIModel(client)

    elif anthropic_key:
        from anthropic import Anthropic
        client = Anthropic(api_key=anthropic_key)

        class AnthropicModel:
            def __init__(self, client, model="claude-3-5-haiku-20241022"):
                self.client = client
                self.model = model
                self.call_log = []
                self.verbose = False

            def __call__(self, prompt: str) -> str:
                response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                result = response.content[0].text

                self.call_log.append({
                    "prompt": prompt,
                    "response": result,
                    "model": self.model
                })

                if self.verbose:
                    print("\n" + "="*60)
                    print("LLM PROMPT:")
                    print("-"*60)
                    print(prompt[:1500] + ("..." if len(prompt) > 1500 else ""))
                    print("-"*60)
                    print("LLM RESPONSE:", result[:200])
                    print("="*60)

                return result

            def save_log(self, path: str):
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self.call_log, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(self.call_log)} LLM calls to {path}")

        return AnthropicModel(client)

    else:
        raise ValueError(
            "No API key found. Either:\n"
            "  1. Set environment variable: OPENAI_API_KEY or ANTHROPIC_API_KEY\n"
            "  2. Create api_key.json with: {\"API_KEY\": \"your-key-here\"}"
        )


def select_agents(count: int) -> list[AgentConfig]:
    """Select the appropriate number of agents."""
    if count <= 10:
        return DEFAULT_ISLAND_AGENTS[:count]
    elif count <= 100:
        return ISLAND_100_AGENTS[:count]
    else:
        # For > 100, repeat agents with modified names
        agents = list(ISLAND_100_AGENTS)
        while len(agents) < count:
            for base in ISLAND_100_AGENTS:
                if len(agents) >= count:
                    break
                # Create a copy with modified name
                new_agent = AgentConfig(
                    name=f"{base.name} Jr.",
                    home_place=base.home_place,
                    work_place=base.work_place,
                    personality=base.personality,
                    backstory=base.backstory,
                    schedule_bias=base.schedule_bias,
                )
                agents.append(new_agent)
        return agents[:count]


def select_island(agent_count: int, custom_path: str | None = None) -> Island:
    """Select the appropriate island configuration."""
    if custom_path:
        return Island.load(Path(custom_path))

    # Use larger island for more agents
    if agent_count <= 10:
        return Island.load(Path("data/island.yaml"))
    else:
        return Island.load(Path("data/island_100.yaml"))


def print_location_summary(sim: ScalableSimulation) -> None:
    """Print a summary of agent locations."""
    print("\nAgent Distribution by Location:")
    print("-" * 40)

    populations = sim.get_location_populations()
    for loc, count in sorted(populations.items(), key=lambda x: -x[1]):
        place = sim.island.places.get(loc)
        name = place.name if place else loc.replace("_", " ").title()
        bar = "*" * min(count, 30)
        print(f"  {name:25s} [{count:3d}] {bar}")


def print_encounter_summary(results: list) -> None:
    """Print a summary of encounters."""
    print("\nEncounters Summary:")
    print("-" * 40)

    all_encounters = []
    for result in results:
        all_encounters.extend(result.encounters)

    if not all_encounters:
        print("  No encounters occurred.")
        return

    # Group by location
    by_location = {}
    for enc in all_encounters:
        if enc.location not in by_location:
            by_location[enc.location] = []
        by_location[enc.location].append(enc)

    for loc, encounters in by_location.items():
        print(f"\n  {loc}: {len(encounters)} encounter(s)")
        for i, enc in enumerate(encounters[:3], 1):  # Show first 3
            print(f"    {i}. {', '.join(enc.participants)}")
            if enc.dialogue:
                speaker, utterance = enc.dialogue[0]
                print(f"       {speaker}: \"{utterance[:50]}...\"")


def main():
    parser = argparse.ArgumentParser(
        description="Run Scalable Island Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_scalable.py --agents 5 --steps 2      # Quick test
  python run_scalable.py --agents 20 --steps 10    # Medium scale
  python run_scalable.py --agents 100 --steps 20   # Full scale
  python run_scalable.py --agents 10 --log-llm     # With LLM logging
        """
    )
    parser.add_argument(
        "--agents", "-a",
        type=int,
        default=10,
        help="Number of agents (default: 10)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=3,
        help="Number of simulation steps (default: 3)"
    )
    parser.add_argument(
        "--encounters", "-e",
        type=int,
        default=3,
        help="Max dialogue turns per encounter (default: 3)"
    )
    parser.add_argument(
        "--island",
        type=str,
        help="Path to island YAML file (auto-selected if not specified)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--log-llm",
        action="store_true",
        help="Log all LLM calls to llm_log.json"
    )
    parser.add_argument(
        "--verbose-llm",
        action="store_true",
        help="Print all LLM prompts and responses"
    )
    parser.add_argument(
        "--start-tick",
        type=int,
        default=16,
        help="Starting tick (default: 16 = 8:00 AM)"
    )
    parser.add_argument(
        "--export",
        type=str,
        default="simulation_results.json",
        help="Export results to JSON file (default: simulation_results.json)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Scalable Island Simulation")
    print("  Location-Based Efficiency Architecture")
    print("=" * 60)
    print()

    # Select agents
    agents = select_agents(args.agents)
    print(f"Agents: {len(agents)}")

    # Load island
    island = select_island(args.agents, args.island)
    print(f"Island: {len(island.places)} locations, {len(island.graph.edges)} connections")

    # Create model
    print("\nInitializing language model...")
    try:
        model = create_simple_model()
        if args.verbose_llm:
            model.verbose = True
        print("  [OK] Model initialized")
    except ValueError as e:
        print(f"  [ERROR] {e}")
        return 1

    # Create simulation
    print("\nCreating simulation...")
    sim = create_scalable_simulation(
        agents=agents,
        island=island,
        model=model,
        start_tick=args.start_tick,
        max_dialogue_turns=args.encounters,
        verbose=args.verbose,
    )
    print(f"  [OK] Simulation created with {len(sim.agents)} agents")

    # Show initial distribution
    print_location_summary(sim)

    # Run simulation
    print(f"\n{'='*60}")
    print(f"Running {args.steps} steps...")
    print(f"{'='*60}")

    results = sim.run(num_steps=args.steps)

    # Print results
    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")

    # Statistics
    stats = sim.get_statistics()
    print("\nStatistics:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Total LLM calls: {stats['total_llm_calls']}")
    print(f"  Average calls/step: {stats['average_llm_calls_per_step']:.1f}")
    print(f"  Total encounters: {stats['total_encounters']}")
    print(f"  Total movements: {stats['total_movements']}")

    # Efficiency comparison
    traditional_estimate = len(agents) * args.steps * 5  # Rough estimate
    print(f"\nEfficiency Comparison:")
    print(f"  Traditional approach (estimated): ~{traditional_estimate} LLM calls")
    print(f"  Scalable approach (actual): {stats['total_llm_calls']} LLM calls")
    if traditional_estimate > 0:
        savings = (1 - stats['total_llm_calls'] / traditional_estimate) * 100
        print(f"  Estimated savings: {savings:.0f}%")

    # Final location distribution
    print_location_summary(sim)

    # Encounter summary
    print_encounter_summary(results)

    # Timing summary
    print("\nTiming Breakdown:")
    total_time = sum(r.timing.total_ms for r in results)
    obs_time = sum(r.timing.observation_ms for r in results)
    move_time = sum(r.timing.movement_ms for r in results)
    enc_time = sum(r.timing.encounter_ms for r in results)
    print(f"  Total time: {total_time/1000:.2f}s")
    print(f"  Observations (parallel): {obs_time/1000:.2f}s ({obs_time/total_time*100:.1f}%)")
    print(f"  Movements (parallel): {move_time/1000:.2f}s ({move_time/total_time*100:.1f}%)")
    print(f"  Encounters (sequential): {enc_time/1000:.2f}s ({enc_time/total_time*100:.1f}%)")

    # Step-by-step summary
    if args.verbose:
        print("\nStep-by-Step Summary:")
        print("-" * 40)
        for i, result in enumerate(results, 1):
            print(f"\nStep {i}:")
            print(f"  Tick: {result.tick}")
            print(f"  LLM calls: {result.llm_calls}")
            print(f"  Observations: {len(result.observations)}")
            moves = sum(1 for v in result.movements.values() if v)
            print(f"  Movements: {moves}")
            print(f"  Encounters: {len(result.encounters)}")
            print(f"  Time: {result.timing.total_ms:.0f}ms "
                  f"(obs: {result.timing.observation_ms:.0f}ms, "
                  f"move: {result.timing.movement_ms:.0f}ms, "
                  f"enc: {result.timing.encounter_ms:.0f}ms)")

    # Export simulation results
    sim.export_results(args.export)

    # Save LLM log
    if args.log_llm:
        model.save_log("llm_log.json")
        print(f"LLM log saved to: llm_log.json")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
