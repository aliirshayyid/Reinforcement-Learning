import numpy as np 

class Agent():
	def __init__(self, epsilon = 0.1, gamma = 0.99):
		self.Q = {} # values of each state
		self.sum_space = [i for i in range(4, 22)]
		self.dealer_show_card_space = [i+1 for i in range (10)]
		self.ace_space = [False, True]
		self.action_space = [0, 1] # stick or hit

		self.state_space = []
		self.returns = {}
		self.pairs_visited = {} # first visit or not 
		self.memory = []

		self.gamma = gamma 
		self.epsilon = epsilon

		self.init_vals()
		self.init_policy()

	def init_vals(self):
		for total in self.sum_space:
			for card in self.dealer_show_card_space:
				for ace in self.ace_space:
					state = (total, card, ace)
					self.state_space.append(state)
					for action in self.action_space:
						self.returns[(state, action)] = [] # return of each state 
						self.pairs_visited[(state, action)] = 0 # didn't visit any state yet
						self.Q[(state, action)] = 0 # initialize each stae_value to 0


	def init_policy(self):
		policy = {}
		n = len(self.action_space)
		for state in self.state_space:
			policy[state] = [1/n for _ in range(n)] # we start with an equal prob for each action in each state
		self.policy = policy

	def choose_action(self, state):
		# action = np.random.choice([0, 1], p = [0.5, 0.5 ]) for example
		action = np.random.choice(self.action_space, p=self.policy[state])
		return action

	def update_Q(self): #update the value of state_action pair 
		for idt, (state, action, _) in enumerate(self.memory):
			G = 0
			if self.pairs_visited[(state , action)] == 0:  #if it's the first visit
				self.pairs_visited[(state, action)] += 1
				discount = 1
				for t, (_, _, reward) in enumerate(self.memory[idt:]): # form idt forward
					G += reward * discount
					discount *= self.gamma
					self.returns[(state, action)].append(G)
		#update the value of each state_action pair with the mean of all the next state discounted reward
		for state, action, _ in self.memory:
			self.Q[(state, action)] = np.mean(self.returns[(state, action)])
			self.update_policy(state)

		# for the new episode we reset 0 becuase it's new start and we didn't visited any yet
		for state in self.state_space: 
			for action in self.action_space:
				self.pairs_visited[(state, action)] =0
		# reset the memory
		self.memory = []

	def update_policy(self, state):
		actions = [self.Q[(state, a)] for a in self.action_space ] # bring the value of each a in that state
		n_actions = len(self.action_space)
		a_max = np.argmax(actions) # select the best action (exploit])
		probs = []
		for action in self.action_space:
			prob = 1 - self.epsilon + self.epsilon / n_actions if action == a_max else self.epsilon / n_actions
			probs.append(prob)
		self.policy[state] = probs


