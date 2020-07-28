import os
import numpy as np 
import torch as T 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim


class CriticNet(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1Dms, fc2Dms,
                    name, chkpt_dir='tmp/td3'):
        super(CriticNet, self).__init__()
        self.input_dims = input_dims
        self.fc1Dms = fc1Dms
        self.fc2Dms = fc2Dms 
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        # this breaks if the env has a 2Dstate representation
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1Dms )
        self.fc2 = nn.Linear(self.fc1Dms, self.fc2Dms)

        self.q1 = nn.Linear(self.fc2Dms, 1)

        #optimization 
        self.optimizer = optim.Adam(self.parameters(), lr= beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        # dim =1 becuase at the batch size is at dim=0
        state_action_value = self.fc1(T.cat([state, action], dim=1) )
        state_action_value = F.relu(state_action_value)
        state_action_value = self.fc2(state_action_value)
        state_action_value = F.relu((state_action_value))

        state_action_value = self.q1(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print(' ,,, saving checkpoint ,,,')
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print(' ,,, load checkpoint ,,,')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNet(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, fc1Dms, fc2Dms,
                    name, chkpt_dir='tmp/td3'):
        super(ActorNet, self).__init__()
        self.input_dims = input_dims
        # n_actions is 4 components for the walker env  
        self.n_actions = n_actions
        self.fc1Dms = fc1Dms
        self.fc2Dms = fc2Dms
        self.name = name 
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1Dms)
        self.fc2 = nn.Linear(self.fc1Dms, self.fc2Dms)
        self.mu = nn.Linear(self.fc2Dms, self.n_actions)

        #optimization 
        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.mu(x)
        # we have our actions in environment bounded between(1,-1,0) so tanh works 
        # cuz it bounds(1,-1), but if we have ex[-2 to 2] then we should multiply it with 2
        x = T.tanh(x)
        return x

    def save_checkpoint(self):
        print(' ,,, saving checkpoint ,,,')
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print(' ,,, load checkpoint ,,,')
        self.load_state_dict(T.load(self.checkpoint_file))

