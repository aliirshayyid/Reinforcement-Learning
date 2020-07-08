import gym 
import numpy as np 
from ddpg import DDpgAgent
from utils import plot_learning_curve



if __name__ == '__main__':
    n_games = 1000
    env = gym.make('LunarLanderContinuous-v2')
    # print('action size,', env.action_space.shape[0])
    # print('state size', env.observation_space.shape)
    agent = DDpgAgent(alpha=0.0001, beta=0.001, input_dims=env.observation_space.shape,
                         tau=0.001, n_actions=env.action_space.shape[0],
                         gamma=0.99, fc1Dms=400,fc2Dms=300, max_size=1000000,batch_size=64)
    filename = 'LunarLander_alpha_' + str(agent.alpha)+'_beta_'+\
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'
    best_score = env.reward_range[0] # lowest score
    score_history = []
    for episode in range(n_games):
        state = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(state)
            # print('actions after choose', action)
            new_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, new_state, done)
            agent.learn()
            score += reward
            state = new_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', episode, 'score %.1f' % score, 
                'average score %.1f' % avg_score )

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(score_history, x, figure_file)