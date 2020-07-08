import torch as T
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 

class PolicyNetwork(nn.Module):
	def __init__(self,lr , input_dims, action_size, fc1Dims = 128, fc2Dims= 128):
		super(PolicyNetwork, self).__init__()
		self.hidden_layer1 = nn.Linear(*input_dims, fc1Dims)
		self.hidden_layer2 = nn.Linear(fc1Dims, fc2Dims)
		self.out_put = nn.Linear(fc2Dims, action_size)
		# optimizer 
		self.optimizer = optim.Adam(self.parameters(), lr = lr)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)
		# activations
		# self.relu = F.relu()

	def forward(self, x):
		x = self.hidden_layer1(x)
		x = F.relu(x)
		x = self.hidden_layer2(x)
		x = F.relu(x)
		x = self.out_put(x)

		return x


class PGAgent():
	def __init__(self, lr, input_dims, gamma = 0.99, action_size = 4):
		self.lr = lr 
		self.gamma = gamma 
		self.reward_memory = [] 
		self.action_memory = [] 

		self.policy = PolicyNetwork(self.lr, input_dims, action_size)

	def choose_action(self, state):
		# first change the state into a tensor and also send it to device
		state = T.Tensor([state]).to(self.policy.device) 
		probs = F.softmax(self.policy.forward(state)) # o/p is a probability distribution
		action_probs = T.distributions.Categorical(probs)
		action = action_probs.sample()
		# we take the log cuz the nn likes small numbers that don't change much
		log_probs = action_probs.log_prob(action) # taking the ln of the probability of the selected action
		self.action_memory.append(log_probs)
		return action.item() # action is a tensor so with .item() we make it a scalar

	def store_rewards(self, reward):
		self.reward_memory.append(reward)

	def learn(self):
		self.policy.optimizer.zero_grad()
		# G_t = R_t+1 + gamma * R_t+2 + gamma**2 * R_t+3 ...
		# G_t = sum from k=0 to k=T {gamma**k * R_t+k+1 } 
		G = np.zeros_like(self.reward_memory, dtype=np.float64)
		for t in range(len(self.reward_memory)):
			G_sum = 0
			discount = 1 
			for k in range(t, len(self.reward_memory)):
				G_sum += self.reward_memory[k] * discount
				discount *= self.gamma 
			G[t] = G_sum

		G = T.tensor(G, dtype=T.float).to(self.policy.device)

		loss = 0 
		for g, log_prob in zip(G, self.action_memory):
			loss += -g * log_prob 
		loss.backward()
		self.policy.optimizer.step()

		self.action_memory = []
		self.reward_memory = [] 

































