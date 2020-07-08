import gym
from Q_control_cartpole import Agent
import matplotlib.pyplot as plt 

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	n_episodes = 20000
	agent = Agent(gamma = 0.99, alpha = 0.1, epsilon = 1, epsilon_min = 0.01, epsilon_decay = 1/(4*n_episodes))
	all_rewards = []
	for episode in range(n_episodes):
		state = env.reset()
		state = agent.get_state(state)
		done = False 
		score = 0
		while not done:
			action = agent.choose_action(state)
			new_state, reward, done, info = env.step(action)
			new_state = agent.get_state(new_state)
			agent.update_Q(state, reward, new_state, action)
			state = new_state
			score = score + reward
		all_rewards.append(score)
		print('the score ', score, 'episode' , episode, 'epsilon ', agent.epsilon)
	plt.plot(all_rewards[-100:])
	plt.show()

	