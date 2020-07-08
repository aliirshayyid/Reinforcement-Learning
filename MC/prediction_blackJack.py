import numpy as np 

class Agent():
	def __init__(self, gamma = 0.99):
		self.V = {} # values of each state
		self.sum_space = [i for i in range(4, 22)]
		self.dealer_show_card_space = [i+1 for i in range (10)]
		self.ace_space = [False, True]
		self.action_space = [0, 1] # stick or hit

		self.state_space = []
		self.returns = {}
		self.states_visited = {} # first visit or not 
		self.memory = []
		self.gamma = gamma 

		self.init_vals()

	def init_vals(self):
		for total in self.sum_space:
			for card in self.dealer_show_card_space:
				for ace in self.ace_space:
					self.V[(total, card, ace)] = 0 # initialize each stae_value to 0
					self.returns[(total, card, ace)] = [] # return of each state 
					self.states_visited[(total, card, ace)] = 0 # didn't visit any state yet
					self.state_space.append((total, card, ace))

	def policy(self, state):
		total, _, _ = state 
		action = 0 if total >= 20 else 1
		return action 

	def update_V(self): #update the value of state 
		for idt, (state, _) in enumerate(self.memory):
			G = 0
			if self.states_visited[state] == 0:  #if it's the first visit
				self.states_visited[state] += 1
				discount = 1
				for t, (_, reward) in enumerate(self.memory[idt:]): # form idt forward
					G += reward * discount
					discount *= self.gamma
					self.returns[state].append(G)
		#update the value of each state with the mean of all the next state discounted reward
		for state, _ in self.memory:
			self.V[state] = np.mean(self.returns[state])

# for the new episode we reset 0 becuase it's new start and we didn't visited any yet
		for state in self.state_space: 
			self.states_visited[state] =0
			# reset the memory
		self.memory = []















