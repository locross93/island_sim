"""
Island Game Master prefab using Concordia's situated_in_time_and_place pattern.

This GM manages:
- Time progression (30-minute ticks, 48 per day)
- Agent locations and movement
- Event resolution and encounters
- World state maintenance
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains as thought_chains_lib
from concordia.typing import prefab as prefab_lib


_ISLAND_CLOCK_DESCRIPTION = (
    'Time on the island is tracked in 30-minute increments called "ticks". '
    'There are 48 ticks per day (tick 0 = midnight, tick 16 = 8:00 AM, tick 24 = noon, etc.). '
    'The day/night cycle affects when locations are open and when agents are active. '
    'Track the current day number and time of day. Locations have open hours expressed in ticks. '
    'Infer the time progression from the simulation step count and agent activities.'
)


@dataclasses.dataclass
class IslandGameMaster(prefab_lib.Prefab):
    """
    Prefab implementing Game Master for island simulation.

    Uses Concordia's situated_in_time_and_place pattern adapted for the island:
    - Generative clock tracking time in ticks
    - Locations component tracking agents at each place
    - World state aggregation
    - Event resolution with thought chains
    - Movement validation and execution
    """

    description: str = (
        'A Game Master for an island simulation with time-and-place '
        'mechanics, agent movement, and social encounters.'
    )

    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            'name': 'Island Game Master',
            # Clock configuration
            'start_time': 'Day 1, 08:00 (tick 16)',
            # Island locations
            'locations': (
                'The island has multiple locations including residential areas, '
                'social spaces (beach, cafe, restaurant, tavern, park, town square), '
                'and work/service locations (clinic, school, harbor, market, etc.). '
                'Each location has open hours and a type (residential, social, work, service, quiet). '
                'Agents can travel between connected locations, with travel taking time.'
            ),
            # Extra thought chain steps for event resolution
            'extra_event_resolution_steps': '',
            # Extra components to add
            'extra_components': {},
            'extra_components_index': {},
        }
    )

    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the Island Game Master entity.

        Args:
            model: The language model to use
            memory_bank: The shared memory bank for game master(s)

        Returns:
            An EntityAgentWithLogging instance acting as the Game Master
        """
        # Extract parameters
        name = self.params.get('name', 'Island Game Master')
        start_time = self.params.get('start_time', 'Day 1, 08:00 (tick 16)')
        location_descriptions = self.params.get('locations')
        extra_event_resolution_steps = self.params.get(
            'extra_event_resolution_steps', ''
        )
        extra_components = self.params.get('extra_components', {})
        extra_components_index = self.params.get('extra_components_index', {})

        # Validate extra components
        if extra_components_index and extra_components:
            if extra_components_index.keys() != extra_components.keys():
                raise ValueError(
                    'extra_components_index must have the same keys as extra_components.'
                )

        # Parse extra event resolution steps
        if ',' in extra_event_resolution_steps:
            extra_event_resolution_steps = [
                step.strip()
                for step in extra_event_resolution_steps.split(',')
                if step
            ]
        else:
            extra_event_resolution_steps = [extra_event_resolution_steps] if extra_event_resolution_steps else []

        # === CORE GM COMPONENTS ===

        # Memory component
        memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        memory_component = actor_components.memory.AssociativeMemory(
            memory_bank=memory_bank
        )

        # Instructions for the GM
        instructions_key = 'instructions'
        instructions = gm_components.instructions.Instructions()

        # Examples for synchronous interactions
        examples_synchronous_key = 'examples'
        examples_synchronous = gm_components.instructions.ExamplesSynchronous()

        # Player character names
        player_names = [entity.name for entity in self.entities]
        player_characters_key = 'player_characters'
        player_characters = gm_components.instructions.PlayerCharacters(
            player_characters=player_names,
        )

        # Observation handling
        observation_to_memory_key = 'observation_to_memory'
        observation_to_memory = actor_components.observation.ObservationToMemory()

        observation_component_key = (
            actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        )
        observation = actor_components.observation.LastNObservations(
            history_length=1000,  # Keep extensive history for GM
        )

        # === EVENT AND STATE TRACKING ===

        # Display recent events
        display_events_key = 'display_events'
        display_events = gm_components.event_resolution.DisplayEvents(
            model=model,
            pre_act_label=(
                'Recent events (ordered from oldest to most recent)'
            ),
        )

        # Relevant memories based on current context
        relevant_memories_key = 'relevant_memories'
        relevant_memories = (
            actor_components.all_similar_memories.AllSimilarMemories(
                model=model,
                components=[
                    display_events_key,
                ],
                num_memories_to_retrieve=25,
                pre_act_label='Background information',
            )
        )

        # === TIME AND PLACE COMPONENTS ===

        # Location descriptions constant
        locations_constant_key = 'locations_constant'
        locations_constant = actor_components.constant.Constant(
            location_descriptions, pre_act_label='Island locations'
        )

        # Clock description constant
        clock_constant_key = 'clock_constant'
        clock_constant = actor_components.constant.Constant(
            _ISLAND_CLOCK_DESCRIPTION, pre_act_label='Time system'
        )

        # Generative clock that tracks current time
        generative_clock_key = 'generative_clock'
        generative_clock = gm_components.world_state.GenerativeClock(
            model=model,
            prompt=_ISLAND_CLOCK_DESCRIPTION,
            start_time=start_time,
            components=[
                instructions_key,
                clock_constant_key,
                display_events_key,
            ],
            pre_act_label='\nCurrent time',
        )

        # Agent locations tracker
        locations_key = 'locations'
        entity_locations = gm_components.world_state.Locations(
            model=model,
            entity_names=player_names,
            prompt=location_descriptions,
            components=[
                instructions_key,
                locations_constant_key,
                clock_constant_key,
                player_characters_key,
                relevant_memories_key,
                display_events_key,
                generative_clock_key,
            ],
            pre_act_label='\nCurrent agent locations',
        )

        # Aggregate world state
        world_state_key = 'world_state'
        world_state = gm_components.world_state.WorldState(
            model=model,
            components=[
                instructions_key,
                player_characters_key,
                locations_constant_key,
                clock_constant_key,
                locations_key,
                generative_clock_key,
                relevant_memories_key,
                display_events_key,
            ],
            pre_act_label='\nCurrent world state',
        )

        # === INTERACTION COMPONENTS ===

        # Make observations for players
        make_observation_key = (
            gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
        )
        make_observation = gm_components.make_observation.MakeObservation(
            model=model,
            player_names=player_names,
            components=[
                instructions_key,
                player_characters_key,
                locations_constant_key,
                clock_constant_key,
                relevant_memories_key,
                display_events_key,
                generative_clock_key,
                locations_key,
                world_state_key,
            ],
            reformat_observations_in_specified_style=(
                'Format observations as: "// Day X, HH:MM // situation description".'
            ),
        )

        # Determine next acting player
        next_acting_kwargs = dict(
            model=model,
            components=[
                instructions_key,
                player_characters_key,
                locations_constant_key,
                clock_constant_key,
                relevant_memories_key,
                display_events_key,
                generative_clock_key,
                locations_key,
                world_state_key,
            ],
        )
        next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
        next_actor = gm_components.next_acting.NextActing(
            **next_acting_kwargs,
            player_names=player_names,
        )

        # Determine action spec for next player
        next_action_spec_key = (
            gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
        )
        next_action_spec = gm_components.next_acting.NextActionSpec(
            **next_acting_kwargs,
            player_names=player_names,
        )

        # === EVENT RESOLUTION ===

        # Define thought chains for event resolution
        account_for_agency_of_others = thought_chains_lib.AccountForAgencyOfOthers(
            model=model, players=self.entities, verbose=False
        )

        event_resolution_steps = [
            account_for_agency_of_others,
            thought_chains_lib.result_to_who_what_where,
        ]

        # Add any extra thought chain steps
        if extra_event_resolution_steps:
            for step in extra_event_resolution_steps:
                if step:
                    event_resolution_steps.append(getattr(thought_chains_lib, step))

        event_resolution_components = [
            instructions_key,
            player_characters_key,
            locations_constant_key,
            clock_constant_key,
            relevant_memories_key,
            display_events_key,
            generative_clock_key,
            locations_key,
            world_state_key,
        ]

        event_resolution_key = (
            gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
        )
        event_resolution = gm_components.event_resolution.EventResolution(
            model=model,
            event_resolution_steps=event_resolution_steps,
            components=event_resolution_components,
            notify_observers=True,
        )

        # === ASSEMBLE GAME MASTER ===

        components_of_game_master = {
            instructions_key: instructions,
            examples_synchronous_key: examples_synchronous,
            player_characters_key: player_characters,
            locations_constant_key: locations_constant,
            clock_constant_key: clock_constant,
            relevant_memories_key: relevant_memories,
            observation_component_key: observation,
            observation_to_memory_key: observation_to_memory,
            display_events_key: display_events,
            generative_clock_key: generative_clock,
            locations_key: entity_locations,
            world_state_key: world_state,
            memory_component_key: memory_component,
            make_observation_key: make_observation,
            next_actor_key: next_actor,
            next_action_spec_key: next_action_spec,
            event_resolution_key: event_resolution,
        }

        component_order = list(components_of_game_master.keys())

        # Add extra components if provided
        if extra_components:
            components_of_game_master.update(extra_components)
            if extra_components_index:
                for component_name in extra_components.keys():
                    component_order.insert(
                        extra_components_index[component_name],
                        component_name,
                    )
            else:
                component_order = list(components_of_game_master.keys())

        # Act component for GM (SwitchAct coordinates between components)
        act_component = gm_components.switch_act.SwitchAct(
            model=model,
            entity_names=player_names,
            component_order=component_order,
        )

        # Create the Game Master entity
        game_master = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components_of_game_master,
        )

        return game_master
