import numpy as np 

class Agent():
	def __init__(self, gamma, alpha, epsilon, epsilon_min, epsilon_decay ):
		# state spaces 
		self.Q = {}
		self.cartPos_space = np.linspace(-2.4, 2.4, 10)
		self.cartVel_space = np.linspace(-4, 4, 10)
		self.poleAngle_space = np.linspace(-0.2094, 0.2094, 10)
		self.poleVel_space = np.linspace(-4, 4, 10)
		self.action_space = [0, 1]


		self.gamma = gamma
		self.alpha = alpha
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay

		self.init_vals()

	def init_vals(self):
		for cartPos in range(len(self.cartPos_space)+1):
			for cartVel in range(len(self.cartVel_space) +1):
				for poleAngle in range(len(self.poleAngle_space) +1):
					for poleVel in range(len(self.poleVel_space) +1):
						state = (cartPos, cartVel, poleAngle, poleVel)
						for action in self.action_space:
							self.Q[(state, action)] = 0
		# print(self.Q[(1,2,3,5,(0,1))])

	def get_state(self, state):
		cartP, cartV, poleA, poleV = state
		cartP = int(np.digitize(cartP, self.cartPos_space))
		cartV = int(np.digitize(cartV, self.cartVel_space))
		poleA = int(np.digitize(poleA, self.poleAngle_space))
		poleV = int(np.digitize(poleV, self.poleVel_space))

		return (cartP, cartV, poleA, poleV)




	def choose_action(self, state):
		ghess = np.random.rand()

		if ghess < self.epsilon:
			action = np.random.choice(self.action_space)
		else:
			# print('greeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeedy')
			actions = [self.Q[(state, a)] for a in self.action_space ]
			action = np.argmax(actions)
			# print('value at 0',self.Q[(state, 0)], 'value at 1',self.Q[(state, 1)])
		if self.epsilon <= self.epsilon_min:
			self.epsilon = self.epsilon_min
		else: 
			self.epsilon -= self.epsilon_decay
		# print('then we chose ', action)
		return action

	def update_Q(self, state, reward, new_state, action):
		max_future_q = np.max([self.Q[(new_state, a)] for a in self.action_space ])
		# next_state_best_action = np.argmax(actions)
		delta = reward + (self.gamma * max_future_q) - self.Q[(state, action)]
		self.Q[(state, action)] = self.Q[(state, action)] + self.alpha * delta




	