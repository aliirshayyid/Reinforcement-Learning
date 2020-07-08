import numpy as np 

class Agent():
	def __init__(self, gamma , alpha):
		self.V = {} # values of each state
		self.theta_space = np.linspace(-0.2094, 0.2094, 10) # radians 
		self.action_space = [0, 1] # left or right

		self.state_space = []
		self.gamma = gamma 
		self.alpha = alpha

		self.init_vals()

	def init_vals(self):
		for state in range(11):
			self.V[state] = 0
			self.state_space.append((state))

	def get_state(self, state):

		return int(np.digitize(state[2], self.theta_space))


	def policy(self, theta):
		action = 0 if theta < 5 else 1
		return action 

	def update_V(self, state, reward, new_state): #update the value of state 
		delta = reward + self.gamma * self.V[new_state] - self.V[state]
		self.V[state] += self.alpha * delta

