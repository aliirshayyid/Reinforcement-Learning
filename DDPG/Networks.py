import os
import numpy as np 
import torch as T 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim


class CriticNet(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1Dms, fc2Dms,
                    name, chkpt_dir='tmp/ddpg'):
        super(CriticNet, self).__init__()
        self.input_dims = input_dims
        self.fc1Dms = fc1Dms
        self.fc2Dms = fc2Dms
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1Dms )
        self.fc2 = nn.Linear(self.fc1Dms, self.fc2Dms)

        # batch normalization 
        self.bn1 = nn.LayerNorm(self.fc1Dms)
        self.bn2 = nn.LayerNorm(self.fc2Dms)

        self.action_value = nn.Linear(self.n_actions, self.fc2Dms)

        self.q = nn.Linear(self.fc2Dms, 1)

        #layer weights initialization we are using fan-out not fan-in ("it does not matter")
        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0]) # fc2Dms is the size[0]
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003 
        self.q.weight.data.uniform_(-f3, f3)
        self.q.weight.data.uniform_(-f3, f3)

        #optional action value layer initialization 
        f4 = 1/np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.weight.data.uniform_(-f4, f4)

        #optimization 
        self.optimizer = optim.Adam(self.parameters(), lr= beta, weight_decay=0.01)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value) # layer normuaization (batch normalization)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(action_value, state_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print(' ,,, saving checkpoint ,,,')
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print(' ,,, load checkpoint ,,,')
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNet(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, fc1Dms, fc2Dms,
                    name, chkpt_dir='tmp/ddpg'):
        super(ActorNet, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1Dms = fc1Dms
        self.fc2Dms = fc2Dms
        self.name = name 
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1Dms)
        self.fc2 = nn.Linear(self.fc1Dms, self.fc2Dms)
        self.mu = nn.Linear(self.fc2Dms, self.n_actions)
        
        # layer normalization 
        self.bn1 = nn.LayerNorm(self.fc1Dms)
        self.bn2 = nn.LayerNorm(self.fc2Dms)

        # Batch normalization
        # self.bn1 = nn.BatchNorm1d(self.fc1Dms)
        # self.bn2 = nn.BatchNorm1d(self.fc2Dms)



        # initialization 
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        #optimization 
        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        # print(self.device)


    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.mu(x)
        # we have our actions in environment bounded between(1,-1,0) so tanh works 
        # well but if we have ex[-2 to 2] then we shoul multiply it with 2
        x = T.tanh(x)
        return x

    def save_checkpoint(self):
        print(' ,,, saving checkpoint ,,,')
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print(' ,,, load checkpoint ,,,')
        self.load_state_dict(T.load(self.checkpoint_file))
        
        




