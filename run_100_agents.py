#!/usr/bin/env python3
"""
Run the Island Simulation with 100 agents and economic inequality.

This script initializes and runs the simulation with:
- 100 agents distributed across economic classes
- Larger island with more diverse locations
- Power law distribution of wealth and land ownership
"""

import argparse
from sim.sim import IslandSimulation, SimulationConfig
from sim.agents_100 import ISLAND_100_AGENTS


def main():
    parser = argparse.ArgumentParser(
        description="Run 100-Agent Island Simulation with Economic Inequality"
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=48,
        help="Number of ticks to run (default: 48 = 1 day)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="Use LLM for agent decisions (requires API key)"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider to use (default: openai)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4.1-nano",
        help="LLM model to use (default: gpt-4.1-nano)"
    )
    parser.add_argument(
        "--start-tick",
        type=int,
        default=16,
        help="Starting tick (default: 16 = 8:00 AM)"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save simulation state to file after running"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Load simulation state from file before running"
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Don't print status updates during simulation"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  🏝️  Island Simulation - 100 Agents with Economic Inequality  🏝️")
    print("=" * 70)
    print()
    print("Agent Distribution:")
    print("  - Lower Middle Class (Sunset Apartments):    50 agents")
    print("  - Middle Class (Coral Village):              25 agents")
    print("  - Upper Middle Class (Palm Heights):         15 agents")
    print("  - Upper Class (Ocean View Estates):           7 agents")
    print("  - Elite (Paradise Point Mansions):            3 agents")
    print("=" * 70)
    print()

    # Create simulation config
    config = SimulationConfig(
        island_yaml_path="data/island_100.yaml",
        use_llm=args.use_llm,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        start_tick=args.start_tick,
        auto_save_interval=48,  # Auto-save every day
    )

    # Initialize simulation
    print("Initializing simulation...")
    sim = IslandSimulation(config=config)

    # Initialize with 100 agents
    print("Loading 100 agents...")
    sim.initialize_agents(configs=ISLAND_100_AGENTS)

    if args.load:
        print(f"Loading state from {args.load}...")
        sim.load(args.load)

    print()
    print(f"Configuration:")
    print(f"  - Ticks to run: {args.ticks}")
    print(f"  - LLM mode: {'enabled' if args.use_llm else 'disabled (heuristic)'}")
    if args.use_llm:
        print(f"  - Provider: {args.llm_provider}")
        print(f"  - Model: {args.llm_model}")
    print(f"  - Start time: {sim.get_time_info()['time']}")
    print()
    print("Starting simulation...")
    print("-" * 70)
    print()

    # Run simulation
    try:
        sim.run(num_ticks=args.ticks, print_status=not args.no_print)
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")

    print()
    print("=" * 70)
    print("Simulation Complete!")
    print("=" * 70)

    # Print summary
    time_info = sim.get_time_info()
    print(f"Final time: Day {time_info['day']}, {time_info['time']}")
    print(f"Total encounters: {len(sim.gm.encounter_history)}")
    print(f"Total ticks elapsed: {sim.gm.current_tick}")
    print()

    # Save if requested
    if args.save:
        print(f"Saving simulation state to {args.save}...")
        sim.save(args.save)
        print()

    # Print some interesting stats
    print("Agent Location Summary:")
    location_counts = {}
    for agent in sim.agents:
        loc = agent.current_location
        location_counts[loc] = location_counts.get(loc, 0) + 1

    for loc, count in sorted(location_counts.items(), key=lambda x: -x[1])[:10]:
        place = sim.island.get_place(loc)
        place_name = place.name if place else loc
        print(f"  - {place_name}: {count} agents")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
