"""
Island world loader and graph logic.
Tick system: 0 = midnight, 48 ticks/day (30-minute intervals)
"""

from pathlib import Path
from typing import Optional

import networkx as nx
import yaml
from pydantic import BaseModel, ConfigDict


TICKS_PER_DAY = 48
MINUTES_PER_TICK = 30


class Place(BaseModel):
    """A location on the island."""
    id: str
    name: str
    type: str  # social, work, quiet, service
    open: tuple[int, int]  # (open_hour, close_hour) in 24h format

    def tick_to_hour(self, tick: int) -> int:
        """Convert tick to hour of day (0-23)."""
        return (tick * MINUTES_PER_TICK // 60) % 24

    def is_open(self, tick: int) -> bool:
        """Check if this place is open at the given tick."""
        hour = self.tick_to_hour(tick)
        open_hour, close_hour = self.open

        if close_hour > 24:
            # Handle hours like 24 (midnight) or beyond
            close_hour = close_hour % 24
            if close_hour == 0:
                close_hour = 24

        if open_hour < close_hour:
            # Normal hours (e.g., 9-17)
            return open_hour <= hour < close_hour
        else:
            # Overnight hours (e.g., 22-6)
            return hour >= open_hour or hour < close_hour


class Island(BaseModel):
    """The island world with places and travel graph."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    places: dict[str, Place]
    graph: nx.Graph

    @classmethod
    def load(cls, yaml_path: Path | str) -> "Island":
        """Load island from YAML file."""
        yaml_path = Path(yaml_path)
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Build places dict
        places = {}
        for p in data["places"]:
            place = Place(
                id=p["id"],
                name=p["name"],
                type=p["type"],
                open=tuple(p["open"]),
            )
            places[place.id] = place

        # Build graph with travel times as edge weights
        graph = nx.Graph()
        for place_id in places:
            graph.add_node(place_id)

        for edge in data.get("edges", []):
            from_place, to_place, travel_ticks = edge
            graph.add_edge(from_place, to_place, weight=travel_ticks)

        return cls(places=places, graph=graph)

    def get_place(self, place_id: str) -> Optional[Place]:
        """Get a place by ID."""
        return self.places.get(place_id)

    def shortest_path(self, from_id: str, to_id: str) -> list[str]:
        """Get shortest path between two places."""
        try:
            return nx.shortest_path(self.graph, from_id, to_id, weight="weight")
        except nx.NetworkXNoPath:
            return []

    def travel_ticks(self, from_id: str, to_id: str) -> int:
        """Get travel time in ticks between two places."""
        try:
            return nx.shortest_path_length(self.graph, from_id, to_id, weight="weight")
        except nx.NetworkXNoPath:
            return -1

    def neighbors(self, place_id: str) -> list[str]:
        """Get directly connected places."""
        return list(self.graph.neighbors(place_id))

    def all_place_ids(self) -> list[str]:
        """Get all place IDs."""
        return list(self.places.keys())


def tick_to_time_str(tick: int) -> str:
    """Convert tick to human-readable time string."""
    total_minutes = (tick % TICKS_PER_DAY) * MINUTES_PER_TICK
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def tick_to_day_and_time(tick: int) -> tuple[int, str]:
    """Convert tick to (day_number, time_string)."""
    day = tick // TICKS_PER_DAY + 1
    time_str = tick_to_time_str(tick)
    return day, time_str


if __name__ == "__main__":
    # Quick test
    island = Island.load(Path(__file__).parent.parent / "data" / "island.yaml")

    print("Places:")
    for pid, place in island.places.items():
        print(f"  {pid}: {place.name} ({place.type}) open {place.open[0]}-{place.open[1]}")

    print("\nTravel times from market:")
    for pid in island.all_place_ids():
        if pid != "market":
            ticks = island.travel_ticks("market", pid)
            print(f"  market -> {pid}: {ticks} ticks ({ticks * 30} min)")

    print("\nOpen status at tick 20 (10:00):")
    for pid, place in island.places.items():
        status = "OPEN" if place.is_open(20) else "closed"
        print(f"  {place.name}: {status}")
