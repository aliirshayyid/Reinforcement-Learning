import gym 
from prediction_blackJack import Agent 
if __name__ == '__main__':
	env = gym.make('Blackjack-v0')
	agent = Agent()
	n_episodes = 500000
	for episode in range(n_episodes):
		if episode % 50000 == 0:
			print('starting episode ', episode)
		state = env.reset()
		done = False
		while not done:
			action = agent.policy(state)
			new_state, reward, done, info = env.step(action)
			agent.memory.append((state, reward))
			state = new_state
		agent.update_V()
	print(agent.V[(21, 3, True)])
	print(agent.V[(4, 1, False)])