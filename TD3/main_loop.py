import gym 
import numpy as np 
from utils import plot_learning_curve
from td3 import TD3Agent


if __name__ == "__main__":
    n_games = 1000
    score_history = []
    # env = gym.make('BipedalWalker-v3')
    env = gym.make('LunarLanderContinuous-v2')
    agent = TD3Agent(alpha =0.001, beta=0.001, input_dims=env.observation_space.shape,
                    tau=0.005, env=env, n_actions=env.action_space.shape[0], gamma=0.99,
                    update_actor_interval=2, fc1Dms=400, fc2Dms=300, max_size=1000000,
                    batch_size=100, warmup=1000, noise=0.1)
    filename = 'LunarLander_alpha_' + str(agent.alpha)+'_beta_'+ \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'
    best_score = env.reward_range[0] # lowest score
    
    agent.load_models()

    for episode in range(n_games):
        score = 0 
        state = env.reset()
        done = False 
        while not done:
            action = agent.choose_action(state)
            # print(action, "the action")
            # if episode % 100==0:
            env.render()
            new_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, new_state, done)
            score +=reward
            # agent.learn()
            state = new_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # if avg_score > best_score:
        #     best_score = avg_score
        #     agent.save_models()

        print('episode', episode, 'score %.1f' % score, 
                'average score %.1f' % avg_score )

    x = [i+1 for i in range(n_games)]
    plot_learning_curve( score_history, x, figure_file)




