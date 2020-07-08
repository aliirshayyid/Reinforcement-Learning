import gym 
import numpy as np 
from reinforce_MC_lander_lunar import PGAgent # practice
# from Policy_gradient_lander_lunar import PGAgent
import matplotlib.pyplot as plt 




def plot_learning_curve(scores, x, figure_file):
	running_avg = np.zeros(len(scores))
	for i in range(len(running_avg)):
		running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
	plt.plot(x, running_avg)
	plt.title('running avg of previuos 100 scores')
	plt.savefig(figure_file)


if __name__ == '__main__':
	n_games = 3000
	env = gym.make('LunarLander-v2')

	agent = PGAgent(lr = 0.0005)
	# agent = PGAgent(gamma = 0.99, lr = 0.0005, input_dims = [8], action_size =4)

	fname = 'REINFORCE_' + 'Lunar_Lander_lr' + str(agent.lr) + '_' \
			+ str(n_games) + 'games'
	figure_file = 'plots/' + fname + '.png'

	scores = []
	for episode in range(n_games):
		state = env.reset()
		done = False 
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
		print('eps ',episode , 'score ', score, 'avg_score % .2f' % avg_score)

	x = [i+1 for i in range(len(scores))]
	plot_learning_curve(scores, x, figure_file)







