import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class policy(torch.nn.Module):
	def __init__(self, alpha, inputDims, fc1Dims, fc2Dims, numActions):
		super().__init__()
		self.inputDims = inputDims
		self.numActions = numActions
		self.fc1Dims = fc1Dims
		self.fc2Dims = fc2Dims
		#   primary network
		self.fc1 = nn.Linear(inputDims, fc1Dims)
		self.fc2 = nn.Linear(fc1Dims, fc2Dims)
		self.output = nn.Linear(fc2Dims, numActions)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.output(x)
		return x

class Agent():
	def __init__(self):
		self.


