from abc import ABC, abstractmethod

from grid_environment_with_treasures import Action, GridEnvironmentWithTreasures


class Agent(ABC):
    def __init__(self,env: GridEnvironmentWithTreasures, agent_id: int, name: str):
        self.env = env
        self.agent_id = agent_id
        self.name = name

    @abstractmethod
    def choose_action(self, state: list[int]) -> Action:
        raise NotImplementedError()

    def train(self, old_state: int, action: Action, reward: float, new_state: int):
        ...
