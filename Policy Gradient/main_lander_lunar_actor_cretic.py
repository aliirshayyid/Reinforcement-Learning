import gym 
from Actor_critic_lunar_lunder import ACAgent
import matplotlib.pyplot as plt 
import numpy as np 
from utils import plot_learning_curve



if __name__ == '__main__':
	n_games = 2000
	env = gym.make('LunarLander-v2') 
	agent = ACAgent(input_dims=[8], action_size =4,lr= 5e-6, gamma = 0.99 , fc1Dm = 2048, fc2Dm = 1536)	
	fname = 'Actor_cretic_' + 'Lunar_Lander_'+ str(agent.fc1Dm)+ 'fc1_dims_' + \
			 str(agent.fc2Dm) + 'fc2Dm_lr'+ str(agent.lr)+ '_' + str(n_games) + 'games'
	figure_file = 'plots/' + fname + '.png'
	scores = [] 
	for episode in range(n_games):
		state = env.reset()
		score = 0 
		done = False 
		while not done:
			action = agent.choose_action(state)
			new_state, reward, done, info = env.step(action)
			agent.learn(state, reward, new_state, done)
			score += reward
			state = new_state

		scores.append(score)
		avg_score = np.mean(scores[-100:])
		print('episode ', episode, 'score %.2f' % score, 'avg_score %.2f' % avg_score)

	x = [i+1 for i in range(len(scores))]
	plot_learning_curve(scores, x, figure_file)


