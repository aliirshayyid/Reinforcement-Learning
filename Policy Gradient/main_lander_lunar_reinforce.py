import gym 
import matplotlib.pyplot as plt 
from Policy_gradient_lander_lunar import PGAgent
import numpy as np 

def plot_learning_curve(scores, x, figure_file):
	running_avg = np.zeros(len(scores))
	for i in range(len(running_avg)):
		running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
	plt.plot(x, running_avg)
	plt.title('running avg of previuos 100 scores')
	plt.savefig(figure_file)

if __name__ == '__main__':
	env = gym.make('LunarLander-v2')
	n_games = 3000
	agent = PGAgent(gamma = 0.99, lr = 0.0005, input_dims = [8], action_size =4)


	fname = 'REINFORCE_' + 'Lunar_Lander_lr' + str(agent.lr) + '_' \
				+ str(n_games) + 'games'
	figure_file = 'plots/' + fname + '.png'

	scores = []
	for i in range(n_games):
		done = False 
		state = env.reset()
		score = 0 
		while not done:
			action = agent.choose_action(state)
			new_state, reward, done, info = env.step(action)
			score += reward
			agent.store_rewards(reward)
			state = new_state
		agent.learn()
		scores.append(score)

		avg_score = np.mean(scores[-100:])
		print('episode', i, 'score %.2f' % score, 
				'avg_score %.2f' % avg_score )
	x = [i+1 for i in range(len(scores))]
	plot_learning_curve(scores, x, figure_file)


	