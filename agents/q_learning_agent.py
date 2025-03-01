import random
from typing import Literal

from agents import Agent
from grid_environment_with_treasures import GridEnvironmentWithTreasures, Action


class QLearningAgent(Agent):
    def __init__(
        self,
        env: GridEnvironmentWithTreasures,
        agent_id: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
    ):
        super().__init__(env, agent_id, f"q-learning {agent_id}")

        self.q_table: dict[str, dict[Action, float]] = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def _get_q_value(self, state: list[int], action: Action) -> float:
        return self.q_table.get(str(state), {}).get(action, 0)

    def _get_available_actions(self, position: int) -> list[Action]:
        available_action: list[Literal[Action]] = []
        if position >= self.env.size:  # If the agent is not in the first row
            available_action.append("u")
        if position % self.env.size < self.env.size - 1:  # If the agent is not in the most right column
            available_action.append("r")
        if position < self.env.size * (self.env.size - 1):  # If the agent is not in the bottom row
            available_action.append("d")
        if position % self.env.size > 0:  # If the agent is not in the most left column
            available_action.append("l")

        return available_action

    def choose_action(self, state: list[int]) -> Action:
        position = state.index(self.agent_id)
        available_action = self._get_available_actions(position)
        # In probability epsilon, select randomly:
        if random.random() < self.epsilon:
            return random.choice(available_action)
        # Otherwise, select the action with the highest Q value:
        return max(available_action, key=lambda action: self._get_q_value(state, action))

    def train(self, old_state: list[int], action: Action, reward: float, new_state: list[int]):
        # The Q-learning update rule is:
        # Q(state, move) = Q(state, action) + learning_rate*[reward + discount_factor*max(Q(new_state, future_action)-Q(state, action)]
        old_q_value = self._get_q_value(old_state, action)
        best_future_q_value = max(
            self._get_q_value(new_state, future_action)
            for future_action in self._get_available_actions(new_state.index(self.agent_id))
        )

        if not self.q_table.get(str(old_state)):
            self.q_table[str(old_state)] = {}

        self.q_table[str(old_state)][action] = old_q_value + self.learning_rate * (
            reward + self.discount_factor * best_future_q_value - old_q_value
        )

        # Decay epsilon value
        self.epsilon *= self.epsilon_decay
