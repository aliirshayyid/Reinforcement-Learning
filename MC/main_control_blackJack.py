import gym 
import matplotlib.pyplot as plt 
from control_blackJack import Agent

if __name__ == '__main__':
	env = gym.make('Blackjack-v0')
	agent = Agent(epsilon=0.001)
	n_episodes = 200000
	win_lose_draw = {-1:0, 0:0, 1:0}
	win_rates = []
	for episode in range(n_episodes):
		if episode > 0 and episode % 1000 ==0:
			pct = win_lose_draw[1] / episode # percentage of winning
			win_rates.append(pct)
		if episode % 50000 == 0:
			rates = win_rates[-1] if win_rates else 0.0
			print('starting episode' , episode, 'win rates %.3f' % rates )
		state = env.reset()
		done = False
		while not done:
			action = agent.choose_action(state)
			new_state, reward, done, info = env.step(action)
			agent.memory.append((state, action, reward))
			state = new_state
		agent.update_Q()
		win_lose_draw[reward] +=1
	plt.plot(win_rates)
	plt.show()



