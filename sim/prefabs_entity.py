"""
Island entity prefab using proper Concordia component patterns.

This implements the "three key questions" pattern:
1. What situation am I in right now?
2. What kind of person am I?
3. What would a person like me do in a situation like this?
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class IslandEntity(prefab_lib.Prefab):
    """
    Prefab for island resident entities with personality, needs, and schedules.

    Uses proper Concordia component architecture with:
    - Situation perception
    - Self perception (personality/backstory/traits)
    - Person-by-situation reasoning
    - Goals and needs tracking
    - Relevant memory retrieval
    """

    description: str = (
        'An island resident entity that makes decisions based on their personality, '
        'current needs, schedule preferences, and social relationships.'
    )

    params: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {
        'name': 'Islander',
        'home_place': 'home',
        'work_place': None,
        'personality': 'Friendly and helpful',
        'backstory': 'A resident of the island',
        'schedule_bias': 'FLEXIBLE',  # EARLY_BIRD, NIGHT_OWL, NINE_TO_FIVE, FLEXIBLE
        'formative_memories': [],  # List of initial memories
        'goal': 'Live a fulfilling life on the island',
    })

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build an island entity with proper Concordia components.

        Args:
            model: The language model to use for reasoning
            memory_bank: The associative memory bank for this entity

        Returns:
            An EntityAgentWithLogging instance
        """
        # Extract parameters
        entity_name = self.params.get('name', 'Islander')
        home_place = self.params.get('home_place', 'home')
        work_place = self.params.get('work_place', None)
        personality = self.params.get('personality', 'Friendly and helpful')
        backstory = self.params.get('backstory', 'A resident of the island')
        schedule_bias = self.params.get('schedule_bias', 'FLEXIBLE')
        formative_memories = self.params.get('formative_memories', [])
        goal = self.params.get('goal', 'Live a fulfilling life on the island')

        # === CORE COMPONENTS ===

        # Memory component
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        memory = agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank
        )

        # Instructions
        instructions_key = 'Instructions'
        instructions_text = (
            f'{entity_name} is a resident of the island. '
            f'{backstory}\n\n'
            f'Personality: {personality}\n'
            f'Home: {home_place}\n'
            + (f'Work: {work_place}\n' if work_place else '')
            + f'Schedule preference: {schedule_bias}\n\n'
            f'{entity_name} has basic needs like hunger, fatigue, and social connection. '
            f'{entity_name} forms relationships with other islanders and remembers '
            'past encounters and experiences.'
        )
        instructions = agent_components.instructions.Instructions(
            agent_name=entity_name,
            pre_act_label='\nInstructions',
        )
        # Add formative memories
        if formative_memories:
            for memory_text in formative_memories:
                memory_bank.add(memory_text)
        # Add core identity memories
        memory_bank.add(f'{entity_name} is a resident of the island. {backstory}')
        memory_bank.add(f'{entity_name} has the following personality: {personality}')
        memory_bank.add(f'{entity_name} lives at {home_place}.')
        if work_place:
            memory_bank.add(f'{entity_name} works at {work_place}.')

        # Observation handling
        observation_to_memory_key = 'ObservationToMemory'
        observation_to_memory = agent_components.observation.ObservationToMemory()

        observation_key = (
            agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        )
        observation = agent_components.observation.LastNObservations(
            history_length=50,  # Track last 50 observations
            pre_act_label=(
                '\nRecent events (ordered from least recent to most recent)'
            ),
        )

        # === THREE KEY QUESTIONS COMPONENTS ===

        # 1. What situation am I in right now?
        situation_perception_key = 'SituationPerception'
        situation_perception = (
            agent_components.question_of_recent_memories.SituationPerception(
                model=model,
                pre_act_label=(
                    f'\nQuestion: What situation is {entity_name} in right now?\n'
                    f'Answer'
                ),
            )
        )

        # 2. What kind of person am I?
        self_perception_key = 'SelfPerception'
        self_perception = (
            agent_components.question_of_recent_memories.SelfPerception(
                model=model,
                pre_act_label=(
                    f'\nQuestion: What kind of person is {entity_name}?\n'
                    f'Answer'
                ),
            )
        )

        # 3. What would a person like me do in a situation like this?
        person_by_situation_key = 'PersonBySituation'
        person_by_situation = (
            agent_components.question_of_recent_memories.PersonBySituation(
                model=model,
                components=[
                    self_perception_key,
                    situation_perception_key,
                ],
                pre_act_label=(
                    f'\nQuestion: What would a person like {entity_name} do in '
                    'a situation like this?\n'
                    f'Answer'
                ),
            )
        )

        # === ADDITIONAL COMPONENTS ===

        # Relevant memories retrieval
        relevant_memories_key = 'RelevantMemories'
        relevant_memories = (
            agent_components.all_similar_memories.AllSimilarMemories(
                model=model,
                components=[
                    situation_perception_key,
                ],
                num_memories_to_retrieve=15,
                pre_act_label='\nRecalled memories and observations',
            )
        )

        # Goal/motivation
        goal_key = 'Goal'
        overarching_goal = agent_components.constant.Constant(
            state=goal,
            pre_act_label='\nOverarching goal'
        )

        # Personality trait constant (for easy access)
        personality_key = 'Personality'
        personality_constant = agent_components.constant.Constant(
            state=f'{entity_name} is {personality}',
            pre_act_label='\nCore personality'
        )

        # Time awareness
        time_display_key = 'TimeDisplay'
        time_display = agent_components.constant.Constant(
            state='Pay attention to the current time and location when making decisions.',
            pre_act_label='\nTime awareness'
        )

        # === ASSEMBLE COMPONENTS ===

        components_of_agent = {
            instructions_key: instructions,
            goal_key: overarching_goal,
            personality_key: personality_constant,
            time_display_key: time_display,
            observation_to_memory_key: observation_to_memory,
            relevant_memories_key: relevant_memories,
            self_perception_key: self_perception,
            situation_perception_key: situation_perception,
            person_by_situation_key: person_by_situation,
            observation_key: observation,
            memory_key: memory,
        }

        # Component order determines prompt construction order
        component_order = [
            instructions_key,
            goal_key,
            personality_key,
            time_display_key,
            memory_key,
            relevant_memories_key,
            observation_key,
            situation_perception_key,
            self_perception_key,
            person_by_situation_key,
            observation_to_memory_key,
        ]

        # Act component concatenates all components and generates action
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=component_order,
            randomize_choices=True,  # Add stochasticity to choices
        )

        # Create the entity agent
        agent = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=entity_name,
            act_component=act_component,
            context_components=components_of_agent,
        )

        return agent
