import random

from agents import Agent
from grid_environment_with_treasures import GridEnvironmentWithTreasures, Action


class RandomAgent(Agent):
    def __init__(
        self,
        env: GridEnvironmentWithTreasures,
        agent_id: int
    ):
        super().__init__(env, agent_id, f"random {agent_id}")

    def choose_action(self, state: list[int]) -> Action:
        return random.choice(["u", "d", "l", "r"])
