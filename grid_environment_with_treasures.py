import random
import logging
from copy import copy
from typing import Literal

Action = Literal["u", "r", "d", "l"]

logger = logging.getLogger(__name__)


class GridEnvironmentWithTreasures:
    def __init__(self):
        self.size = 3
        # Initialize treasures at specific locations (excluding the goal)

        self.goal = 1  # index of goal position
        self.reward_treasure = 1
        self.reward_win_collusion = 0.1
        self.reward_finish_first = 10
        self.probabilities = (0.1, 0.1, 0.1, 0.2, 0.3, 0.2, 0, 0.5, 0)  # the probability of a treasure to exist.
        self.prob_win_collision = (0.5, 0.5)

        # Status of each cell: -1: unvisited, -2: visited and empty, 0: agent 0 there. 1: agent 1 there.
        self.state = [-1, -1, -1, -1, -1, -1, 0, -1, 1]
        self.position = [self.size * (self.size - 1), self.size * self.size - 1]
        self.collected_treasures = [0, 0]

        self.collected_treasures: list[int] | None = None
        self.treasures: list[int] | None = None

        self.reset()

    def put_treasures(self):
        self.treasures = [0] * self.size * self.size
        for i, prob in enumerate(self.probabilities):
            if random.random() <= prob:
                self.treasures[i] = 1

    def reset(self):
        # Status of each cell: -1: unvisited, -2: visited and empty, 0: agent 0 there. 1: agent 1 there.
        self.state = [-1, -1, -1, -1, -1, -1, 0, -1, 1]
        self.position = [self.size * (self.size - 1), self.size * self.size - 1]
        self.collected_treasures = [0, 0]  # Track whether each agent has collected a treasure
        self.put_treasures()
        return self.position, self.state

    def from_row_column_to_index(self, row, column):
        return row * self.size + column

    def get_available_moves(self, agent_id: int) -> list[str]:
        available_moves = []
        agent_position = self.position[agent_id]
        if agent_position >= self.size:  # If the agent is not in the first row
            available_moves.append("u")
        if agent_position % self.size < self.size - 1:  # If the agent is not in the most right column
            available_moves.append("r")
        if agent_position < self.size * (self.size - 1):  # If the agent is not in the bottom row
            available_moves.append("d")
        if agent_position % self.size > 0:  # If the agent is not in the most left column
            available_moves.append("l")

        return available_moves

    def step(self, agent_id, action):
        # Actions: 'u' (up), 'd' (down), 'l' (left), 'r' (right)
        reward = 0
        reward_opponent = 0
        old_position = self.position[agent_id]
        new_position = old_position
        if action == "u" and self.position[agent_id] >= self.size:
            new_position -= self.size
        elif action == "r" and self.position[agent_id] % self.size < self.size - 1:
            new_position += 1
        elif action == "d" and self.position[agent_id] < self.size * (self.size - 1):
            new_position += self.size
        elif action == "l" and self.position[agent_id] % self.size > 0:
            new_position -= 1

        if self.position[1 - agent_id] == new_position:
            if random.random() < self.prob_win_collision[agent_id]:
                self.position[1 - agent_id] = old_position

                self.position[agent_id] = new_position
                reward += self.reward_win_collusion
                reward_opponent -= self.reward_win_collusion
                self.state[old_position] = 1 - agent_id
                self.state[new_position] = agent_id
                logger.debug(f"A collusion: Agent {agent_id} Wins")

            else:
                new_position = old_position
                self.position[agent_id] = new_position
                reward -= self.reward_win_collusion
                reward_opponent += self.reward_win_collusion
                logger.debug(f"A collusion: Agent {1 - agent_id} Wins")
        else:
            self.position[agent_id] = new_position
            self.state[old_position] = -2  # already visited, but now empty
            self.state[new_position] = agent_id  # agent 0 marked as 1, agent 1 marked as 2

        done = False
        # Check if the agent reached its goal
        if self.position[agent_id] == self.goal:
            reward += self.reward_finish_first
            done = True
        if self.treasures[new_position]:
            self.treasures[new_position] = 0
            reward += self.reward_treasure
            self.collected_treasures[agent_id] += 1  # Mark the treasure as collected
            logger.debug(f"Agent {agent_id} found a treasure")
        return self.position, self.state, reward, done, reward_opponent

    def log_grid_state(self):
        logger.debug("Grid state (-1: unvisited, -2: visited and empty, 0: agent 0 there. 1: agent 1 there)")
        state_to_print = copy(self.state)

        # For easy debugging
        # for i, c in enumerate(state_to_print):
        #     if c == 0:
        #         state_to_print[i] = "🤮"
        #     elif c == 1:
        #         state_to_print[i] = "👑"

        for i in range(self.size):
            logger.debug(state_to_print[i * self.size : (i + 1) * self.size])

        # logger.debug("Treasure Map:")
        # for i in range(self.size):
        #     logger.debug(self.treasures[i * self.size : (i + 1) * self.size])
