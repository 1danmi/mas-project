import logging
from copy import copy

from agents import Agent, RandomAgent, QLearningAgent
from grid_environment_with_treasures import GridEnvironmentWithTreasures

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(message)s"))
logger.addHandler(ch)

TRAINING_EPISODES = 100
TESTING_EPISODES = 100


def simulate_game(env: GridEnvironmentWithTreasures, agents: list[Agent], train_agents: bool = False) -> list[float]:
    position, state = env.reset()

    done = [False, False]
    total_rewards = [0, 0]

    env.log_grid_state()

    while not any(done):
        for agent_id, agent in enumerate(agents):

            action = agent.choose_action(state)

            new_position, new_state, reward, done[agent_id], reward_opponent = env.step(agent_id, action)
            env.log_grid_state()
            logging.debug(f"Agent {agent.name} moves {action}")

            if train_agents:
                agent.train(state, action, reward, new_state)

            total_rewards[agent_id] += reward
            total_rewards[1 - agent_id] += reward_opponent

            logging.debug(f"new_position: {new_position} new_state: {new_state}")

            # New state points directly to the games states that changes during the env.step method, therefore we must
            # copy it.
            state = copy(new_state)

            if done[agent_id]:
                break

    logger.info("***GAME OVER***")
    for agent in agents:
        logger.info(f"Agent {agent.name} total reward: {total_rewards[agent.agent_id]}")
    if total_rewards[0] > total_rewards[1]:
        logger.info(f"Agent {agents[0].name} wins this game")
    elif total_rewards[0] < total_rewards[1]:
        logger.info(f"Agent {agents[1].name} wins this game")
    else:
        logger.info("TIE")

    return total_rewards


def train_agents_with_treasures(env: GridEnvironmentWithTreasures, agents: list[Agent], episodes: int):
    for episode in range(episodes):
        logging.info(f"***Training: Game {episode}:***")
        simulate_game(env, agents, True)


def test_agents_with_treasures(env: GridEnvironmentWithTreasures, agents: list[Agent], episodes=5):
    grand_total = {}
    for episode in range(episodes):
        logger.info(f"***Testing: Game {episode}:***")
        totals = simulate_game(env, agents)
        grand_total[agents[0].name] = grand_total.get(agents[0].name, 0) + totals[agents[0].agent_id]
        grand_total[agents[1].name] = grand_total.get(agents[1].name, 0) + totals[agents[1].agent_id]

    return grand_total


if __name__ == "__main__":
    env_with_treasures = GridEnvironmentWithTreasures()

    agents = [
        RandomAgent(env_with_treasures, 0),
        QLearningAgent(env_with_treasures, 1)
    ]

    env_with_treasures.reset()

    env_with_treasures.log_grid_state()

    train_agents_with_treasures(env_with_treasures, agents, TRAINING_EPISODES)

    results = {}


    # logger.setLevel(logging.DEBUG)
    logger.info(f"Testing when {agents[0].name} starts")
    results[agents[0].name] = test_agents_with_treasures(env_with_treasures, agents, TESTING_EPISODES)

    agents.reverse()
    logger.info(f"Testing when {agents[0].name} starts")
    results[agents[0].name] = test_agents_with_treasures(env_with_treasures, agents, TESTING_EPISODES)

    for agent_name, final_results in results.items():
        total_points = sum(final_results.values())
        logger.info(
            f"""
            Final Results when {agent_name} starts: 
            {agents[0].name}: {round(final_results[agents[0].name], 2)} ({round(round(final_results[agents[0].name]/total_points*100, 2))} %)
            {agents[1].name}: {round(final_results[agents[1].name], 2)} ({round(round(final_results[agents[1].name]/total_points*100, 2))} %)"""
        )