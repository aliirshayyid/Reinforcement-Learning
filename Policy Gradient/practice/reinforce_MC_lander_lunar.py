import numpy as np 
import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

class PGNetwork(nn.Module):
	def __init__(self, fc1_dm, fc2_dm, input_dims, action_size, lr):
		super(PGNetwork, self).__init__()
		self.hidden_layer1 = nn.Linear(*input_dims, fc1_dm)
		self.hidden_layer2 = nn.Linear(fc1_dm, fc2_dm)
		self.output_layer = nn.Linear(fc2_dm, action_size)

		self.optimizer = optim.Adam(self.parameters(), lr = lr)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)


	def forward(self, x):
		x = self.hidden_layer1(x)
		x = F.relu(x)
		x = self.hidden_layer2(x)
		x = F.relu(x)
		x = self.output_layer(x)

		return x

class PGAgent():
	def __init__( self, lr , gamma = 0.99):

		self.lr = lr
		self.gamma = gamma 
		self.rewards = [] 
		self.action_memory = [] 

		self.policy = PGNetwork(fc1_dm = 128, fc2_dm =128, input_dims = [8], \
							 action_size = 4, lr = self.lr)

	def choose_action(self, state):
		state = T.Tensor([state]).to(self.policy.device)

		actions = F.softmax(self.policy.forward(state))
		actions_probs = T.distributions.Categorical(actions)
		action = actions_probs.sample()
		log_probs = actions_probs.log_prob(action)
		self.action_memory.append(log_probs)
		return action.item()


	def store_rewards(self, reward ):
		self.rewards.append(reward) 

	def learn(self):
		self.policy.optimizer.zero_grad()
		G = np.zeros_like(self.rewards)
		for t in range(len(self.rewards)):
			discount = 1
			G_sum = 0
			for idx in range(t, len(self.rewards)):
				G_sum += discount * self.rewards[idx]
				discount *= self.gamma
			G[t] = G_sum
		G = T.tensor(G, dtype = T.float).to(self.policy.device)
		loss = 0 
		for g, log_prob in zip(G ,self.action_memory):
			loss += -g * log_prob

		loss.backward()
		self.policy.optimizer.step()

		self.action_memory = []
		self.rewards = []







