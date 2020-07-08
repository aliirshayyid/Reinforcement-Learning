import gym
from prediction_cartpole_TD0 import Agent

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	n_episodes = 5000
	agent = Agent(gamma = 0.99, alpha = 0.1)
	for episode in range(n_episodes):
		
		state = env.reset()
		state = agent.get_state(state)
		done = False 
		while not done:
			action = agent.policy(state)
			new_state, reward, done, info = env.step(action)
			new_state = agent.get_state(new_state)
			agent.update_V(state, reward, new_state)
			state = new_state
	print(agent.V)
