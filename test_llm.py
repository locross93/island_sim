"""
Test script for running the island simulation with GPT-4.1 nano.
Prompts for OpenAI API key and runs a few ticks to test encounters.
"""

import os
import sys
from getpass import getpass

def main():
    print("=" * 60)
    print("  Island Simulation - GPT-4.1 Nano Test")
    print("=" * 60)
    print()

    # Get API key from command line arg or environment
    api_key = None
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("Usage: python test_llm.py <OPENAI_API_KEY>")
        print("   or: set OPENAI_API_KEY=your-key-here")
        print()
        print("Enter your API key as command line argument:")
        sys.exit(1)

    os.environ["OPENAI_API_KEY"] = api_key
    print(f"API key set (ends with ...{api_key[-4:]})")

    print()
    print("Initializing simulation with GPT-4.1 nano...")
    print()

    # Import after setting API key
    from sim.sim import create_simulation

    # Create simulation with LLM enabled
    sim = create_simulation(
        island_path="data/island.yaml",
        use_llm=True,
        start_tick=16,  # 8:00 AM
        llm_provider="openai",
        llm_model="gpt-4.1-nano",
    )

    print()
    print("Running 12 ticks (6 hours of island time)...")
    print("Agents will make LLM-powered decisions and have encounters.")
    print()
    print("-" * 60)

    # Run simulation
    results = sim.run(num_ticks=12, print_status=True)

    print("-" * 60)
    print()

    # Show encounter summary
    encounters = sim.get_recent_events(limit=10)
    if encounters:
        print(f"Encounters that occurred ({len(encounters)}):")
        print()
        for event in encounters:
            print(f"  [{event['place']}] {', '.join(event['participants'])}")
            print(f"    {event['summary']}")
            print()
    else:
        print("No encounters occurred yet. Run more ticks for agents to meet.")

    print("=" * 60)
    print("Test complete!")
    print(f"Final time: Day {sim.get_time_info()['day']}, {sim.get_time_info()['time']}")


if __name__ == "__main__":
    main()
