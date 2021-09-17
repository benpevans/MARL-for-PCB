import agent
import board
import random
import numpy as np
import wandb



if __name__ == '__main__':

    """
    Experiment to run a single agent in a partially observable setting.

    This code acts as a baseline to which all other experiments were adapted

    Multi-agent experiments were done by creating only a single instance of the agent class and
    sampling actions from its q-network. Experiences were stored in its replay buffer
    
    """


    # Initialise WandB
    wandb.init(project='MARL_for_PCB', entity='bevans')

    # board size and number of obstacles
    n = 7

    # For reproducibility
    random.seed(a=52)

    # board parameters
    n_paths=1
    boardsize=(n,n)
    obstacles = n
    partial_obs = 4

    # initialise new game
    game = board.Game(boardsize, n_paths, obstacles, partial_obs)

    #tf.compat.v1.disable_eager_execution()
    n_games = 500000

    agent1 = agent.Agent(
        gamma=0.99, 
        epsilon=1, 
        learning_rate=0.001,
        batch_size=32,
        input_dims=game.get_observation_dims(),
        n_actions=game.get_actions(), 
        partial_obs=game.get_partial_obs(),
        fname="9x9_partial_obs")

    frames = 0
    scores = []
    eps_history = []

    for _ in range(n_games):

        done = False
        score = 0
        avg_score = np.mean(scores[-100:])

        #if new high score, save model
        if avg_score > 5 and n < 15:
            agent1.save_model()
            agent1.epsilon = 1
            n+=1

        # initialise new game episode
        game = board.Game(boardsize=(n,n), n_paths=1, obstacles=n, partial_obs=4)

        # get observation
        obs = game.get_observation(game.paths[0])

        while not done:
            action = agent1.choose_action(obs)
            nextobs, reward, done = game.step([action])
            score += reward[0]
            agent1.store(obs, nextobs[0][0], action, reward[0], done)
            obs = nextobs[0][0]
            agent1.learn()
            frames += 1
        scores.append(score)

        wandb.log({'episode reward':score, 'moving average':avg_score, 'epsilon':agent1.epsilon, 'frames':frames})


    # agent = agenttest.Agent(
    # gamma=0.99, 
    # epsilon=1, 
    # learning_rate=0.001,
    # batch_size=32,
    # input_dims=game.get_observation_dims(),
    # n_actions=game.get_actions(), 
    # partial_obs=game.get_partial_obs(),
    # fname="test")
