import torch as T
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 


class ActorCriticNetwork(nn.Module):
	def __init__(self,  input_dims, action_size, lr, gamma = 0.99,  fc1Dm = 256, fc2Dm = 256 ):
		super(ActorCriticNetwork, self).__init__()

		self.hidden_layer1 = nn.Linear( *input_dims, fc1Dm)
		self.hidden_layer2 = nn.Linear(fc1Dm, fc2Dm )

		self.actor = nn.Linear(fc2Dm, action_size)
		self.critic = nn.Linear(fc2Dm, 1)
		# optimization 
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
		print(self.device)
		# self.device = torch.device("cpu")
		self.to(self.device)

	def forward(self, x):
		x = self.hidden_layer1(x)
		x = F.relu(x)
		x = self.hidden_layer2(x)
		x = F.relu(x)

		policy = self.actor(x)
		value = self.critic(x)

		return (policy, value)



class ACAgent():
	def __init__(self,fc1Dm, fc2Dm , lr, input_dims, action_size, gamma = 0.99 ):
		self.gamma = gamma 
		self.lr = lr 
		self.fc1Dm = fc1Dm
		self.fc2Dm = fc2Dm

		self.actor_critic = ActorCriticNetwork(lr = self.lr, input_dims= input_dims,action_size=action_size, \
												 fc1Dm=self.fc1Dm, fc2Dm=self.fc2Dm)
		self.log_prob = None

	def choose_action(self, state):
		state = T.tensor([state], dtype= T.float).to(self.actor_critic.device)
		probabilities, _ = self.actor_critic.forward(state)
		probabilities = F.softmax(probabilities)
		action_probs = T.distributions.Categorical(probabilities)
		action = action_probs.sample()
		self.log_prob = action_probs.log_prob(action)
		return action.item()

	def learn(self, state, reward, new_state, done): # since it's TD 
		self.actor_critic.optimizer.zero_grad()
		state = T.tensor([state], dtype = T.float).to(self.actor_critic.device)
		new_state = T.tensor([new_state], dtype = T.float).to(self.actor_critic.device)
		reward = T.tensor(reward, dtype = T.float).to(self.actor_critic.device)

		_, value_state = self.actor_critic.forward(state)
		_, value_new_state = self.actor_critic.forward(new_state)
		# we used the (1-int(done)) to make it zero at the terminal state 
		delta = reward + self.gamma * value_new_state * (1 - int(done)) - value_state
		# delta is positive if the reward was greater than what the critic expected 
		# and negative if it was smaller than what the critic expected 
		actor_loss = -self.log_prob * delta
		critic_loss = delta**2 

		(actor_loss + critic_loss).backward() 
		self.actor_critic.optimizer.step()









