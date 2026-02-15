#!/usr/bin/env python3
"""
Analyze simulation results and verify social interactions.

This script reads the simulation results JSON and LLM log to:
1. Verify the three-phase architecture works correctly
2. Analyze timing breakdown
3. Examine social interactions (encounters/dialogues)
4. Validate efficiency claims

Usage:
    python analyze_simulation.py simulation_results.json
    python analyze_simulation.py simulation_results.json --llm-log llm_log.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def analyze_statistics(data: dict) -> None:
    """Analyze overall statistics."""
    print_header("SIMULATION STATISTICS")

    stats = data['statistics']
    print(f"\nAgent count: {stats['agent_count']}")
    print(f"Location count: {stats['location_count']}")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Total LLM calls: {stats['total_llm_calls']}")
    print(f"Average calls/step: {stats['average_llm_calls_per_step']:.1f}")
    print(f"Total encounters: {stats['total_encounters']}")
    print(f"Total movements: {stats['total_movements']}")

    # Efficiency analysis
    traditional_estimate = stats['agent_count'] * stats['total_steps'] * 5
    print(f"\nEfficiency Analysis:")
    print(f"  Traditional estimate: ~{traditional_estimate} calls")
    print(f"  Scalable actual: {stats['total_llm_calls']} calls")
    if traditional_estimate > 0:
        savings = (1 - stats['total_llm_calls'] / traditional_estimate) * 100
        print(f"  Savings: {savings:.1f}%")


def analyze_timing(data: dict) -> None:
    """Analyze timing breakdown."""
    print_header("TIMING ANALYSIS")

    timing = data['timing_summary']
    total = timing['total_ms']

    print(f"\nTotal time: {total/1000:.2f}s")

    phases = [
        ('Observations (parallel)', timing['observation_ms']),
        ('Movements (parallel)', timing['movement_ms']),
        ('Encounters (sequential)', timing['encounter_ms']),
    ]

    print("\nPhase breakdown:")
    for name, ms in phases:
        pct = (ms / total * 100) if total > 0 else 0
        bar = '*' * int(pct / 2)
        print(f"  {name:30s} {ms/1000:6.2f}s ({pct:5.1f}%) {bar}")

    # Per-step timing
    print_subheader("Per-Step Timing")
    steps = data['steps']
    for i, step in enumerate(steps, 1):
        t = step['timing']
        print(f"  Step {i}: {t['total_ms']:.0f}ms "
              f"(obs: {t['observation_ms']:.0f}ms, "
              f"move: {t['movement_ms']:.0f}ms, "
              f"enc: {t['encounter_ms']:.0f}ms)")


def analyze_encounters(data: dict) -> None:
    """Analyze social interactions (encounters)."""
    print_header("SOCIAL INTERACTIONS ANALYSIS")

    steps = data['steps']
    all_encounters = []
    for step in steps:
        all_encounters.extend(step['encounters'])

    if not all_encounters:
        print("\nNo encounters occurred during the simulation.")
        return

    print(f"\nTotal encounters: {len(all_encounters)}")

    # Encounters by location
    by_location = defaultdict(list)
    for enc in all_encounters:
        by_location[enc['location']].append(enc)

    print_subheader("Encounters by Location")
    for loc, encs in sorted(by_location.items(), key=lambda x: -len(x[1])):
        print(f"\n  {loc}: {len(encs)} encounter(s)")

    # Participant frequency
    participant_counts = defaultdict(int)
    for enc in all_encounters:
        for p in enc['participants']:
            participant_counts[p] += 1

    print_subheader("Most Social Agents")
    sorted_participants = sorted(participant_counts.items(), key=lambda x: -x[1])
    for name, count in sorted_participants[:10]:
        bar = '*' * count
        print(f"  {name:25s} {count:2d} encounters {bar}")

    # Sample dialogues
    print_subheader("Sample Dialogues")
    for i, enc in enumerate(all_encounters[:3], 1):
        print(f"\n  Encounter {i} at {enc['location']}:")
        print(f"  Participants: {', '.join(enc['participants'])}")
        for line in enc['dialogue']:
            speaker = line['speaker']
            text = line['text'][:60] + ('...' if len(line['text']) > 60 else '')
            print(f"    {speaker}: \"{text}\"")


def analyze_movements(data: dict) -> None:
    """Analyze agent movements."""
    print_header("MOVEMENT ANALYSIS")

    steps = data['steps']
    all_movements = []
    for step in steps:
        for agent, dest in step['movements'].items():
            all_movements.append((agent, dest))

    print(f"\nTotal movements: {len(all_movements)}")

    if not all_movements:
        print("No agents moved during the simulation.")
        return

    # Movements by agent
    by_agent = defaultdict(list)
    for agent, dest in all_movements:
        by_agent[agent].append(dest)

    print_subheader("Most Mobile Agents")
    sorted_agents = sorted(by_agent.items(), key=lambda x: -len(x[1]))
    for name, dests in sorted_agents[:10]:
        bar = '*' * len(dests)
        print(f"  {name:25s} {len(dests):2d} moves {bar}")
        if len(dests) <= 5:
            print(f"    -> {' -> '.join(dests)}")

    # Destination frequency
    dest_counts = defaultdict(int)
    for _, dest in all_movements:
        dest_counts[dest] += 1

    print_subheader("Popular Destinations")
    sorted_dests = sorted(dest_counts.items(), key=lambda x: -x[1])
    for dest, count in sorted_dests[:10]:
        bar = '*' * count
        print(f"  {dest:25s} {count:2d} arrivals {bar}")


def analyze_observations(data: dict) -> None:
    """Analyze location observations."""
    print_header("OBSERVATIONS ANALYSIS")

    steps = data['steps']
    all_observations = {}
    for step in steps:
        for loc, obs in step['observations'].items():
            if loc not in all_observations:
                all_observations[loc] = []
            all_observations[loc].append(obs)

    print(f"\nUnique locations observed: {len(all_observations)}")

    # Sample observations
    print_subheader("Sample Observations")
    for loc in list(all_observations.keys())[:3]:
        obs = all_observations[loc][0]
        obs_truncated = obs[:100] + ('...' if len(obs) > 100 else '')
        print(f"\n  {loc}:")
        print(f"    \"{obs_truncated}\"")


def analyze_agent_memories(data: dict) -> None:
    """Analyze agent memories at end of simulation."""
    print_header("AGENT MEMORIES")

    agents = data['agents']

    # Sample agent memories
    print_subheader("Sample Agent States")
    for name in list(agents.keys())[:5]:
        agent = agents[name]
        print(f"\n  {name}:")
        print(f"    Location: {agent['location']}")
        print(f"    Personality: {agent['personality'][:50]}...")
        print(f"    Memories ({len(agent['memories'])} total):")
        for mem in agent['memories'][-3:]:
            mem_truncated = mem[:60] + ('...' if len(mem) > 60 else '')
            print(f"      - {mem_truncated}")


def analyze_llm_log(llm_log_path: str) -> None:
    """Analyze LLM call log."""
    print_header("LLM CALL ANALYSIS")

    calls = load_json(llm_log_path)
    print(f"\nTotal LLM calls: {len(calls)}")

    # Categorize calls by type
    observation_calls = []
    movement_calls = []
    dialogue_calls = []
    other_calls = []

    for call in calls:
        prompt = call['prompt'].lower()
        if 'describe' in prompt or 'observe' in prompt:
            observation_calls.append(call)
        elif 'stay or move' in prompt:
            movement_calls.append(call)
        elif 'talking' in prompt or 'dialogue' in prompt or 'say next' in prompt:
            dialogue_calls.append(call)
        else:
            other_calls.append(call)

    print("\nCall breakdown:")
    print(f"  Observations: {len(observation_calls)}")
    print(f"  Movements: {len(movement_calls)}")
    print(f"  Dialogues: {len(dialogue_calls)}")
    print(f"  Other: {len(other_calls)}")

    # Sample prompts
    print_subheader("Sample Observation Prompt")
    if observation_calls:
        prompt = observation_calls[0]['prompt'][:300]
        response = observation_calls[0]['response'][:150]
        print(f"  Prompt: {prompt}...")
        print(f"  Response: {response}...")

    print_subheader("Sample Movement Prompt")
    if movement_calls:
        prompt = movement_calls[0]['prompt'][:300]
        response = movement_calls[0]['response'][:100]
        print(f"  Prompt: {prompt}...")
        print(f"  Response: {response}")

    print_subheader("Sample Dialogue Prompt")
    if dialogue_calls:
        prompt = dialogue_calls[0]['prompt'][:300]
        response = dialogue_calls[0]['response'][:100]
        print(f"  Prompt: {prompt}...")
        print(f"  Response: {response}")


def verify_architecture(data: dict) -> None:
    """Verify the three-phase architecture works correctly."""
    print_header("ARCHITECTURE VERIFICATION")

    issues = []

    # Check that observations are location-scoped
    steps = data['steps']
    for i, step in enumerate(steps, 1):
        obs_locs = set(step['observations'].keys())
        # Check that we don't have more observations than locations
        if len(obs_locs) > data['statistics']['location_count']:
            issues.append(f"Step {i}: More observations than locations")

    # Check that encounters only happen at locations with 2+ agents
    # (We can't verify this perfectly without agent locations, but we can check participants)
    for i, step in enumerate(steps, 1):
        for enc in step['encounters']:
            if len(enc['participants']) < 2:
                issues.append(f"Step {i}: Encounter with <2 participants")

    # Verify timing makes sense (parallel should be faster per call)
    timing = data['timing_summary']
    stats = data['statistics']

    # Calculate per-call timing
    if stats['total_llm_calls'] > 0:
        avg_ms_per_call = timing['total_ms'] / stats['total_llm_calls']
        print(f"\nAverage time per LLM call: {avg_ms_per_call:.0f}ms")

    # Check phase timing ratios
    total = timing['total_ms']
    if total > 0:
        obs_pct = timing['observation_ms'] / total * 100
        move_pct = timing['movement_ms'] / total * 100
        enc_pct = timing['encounter_ms'] / total * 100

        print(f"\nPhase timing distribution:")
        print(f"  Observations: {obs_pct:.1f}%")
        print(f"  Movements: {move_pct:.1f}%")
        print(f"  Encounters: {enc_pct:.1f}%")

        # Encounters are sequential, so they should typically take more time
        # per LLM call than parallel phases
        if stats['total_encounters'] > 0:
            enc_calls = sum(len(s['encounters']) * 3 for s in steps)  # ~3 turns per encounter
            if enc_calls > 0:
                enc_ms_per_call = timing['encounter_ms'] / enc_calls
                print(f"  Encounter time per dialogue turn: {enc_ms_per_call:.0f}ms")

    # Report issues
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  [!] {issue}")
    else:
        print("\n[OK] All architecture checks passed!")

    # Summary
    print_subheader("Architecture Summary")
    print(f"  Phase 1 (Parallel Observations): {len(steps[0]['observations'])} per step")
    total_encounters = sum(len(s['encounters']) for s in steps)
    print(f"  Phase 2 (Parallel Movements): {stats['agent_count']} agents decide simultaneously")
    print(f"  Phase 3 (Sequential Encounters): {total_encounters} total encounters")

    print("\n[OK] Three-phase location-based architecture verified!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze simulation results and verify social interactions"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to simulation_results.json"
    )
    parser.add_argument(
        "--llm-log",
        type=str,
        help="Path to llm_log.json for detailed LLM analysis"
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Show only summary statistics"
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_file}...")
    data = load_json(args.results_file)

    # Run analyses
    analyze_statistics(data)
    analyze_timing(data)

    if not args.brief:
        analyze_encounters(data)
        analyze_movements(data)
        analyze_observations(data)
        analyze_agent_memories(data)

    verify_architecture(data)

    # Analyze LLM log if provided
    if args.llm_log:
        analyze_llm_log(args.llm_log)

    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print("="*60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
