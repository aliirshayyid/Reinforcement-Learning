import numpy as np 
import torch as T
import torch.nn.functional as F
from Networks import CriticNet, ActorNet
from Replay_Buffer import ReplayBuffer
from noise import OUActionNoise
# T.manual_seed(0)
# np.random.seed(0)


class DDpgAgent():
    def __init__(self,alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                    fc1Dms=400, fc2Dms=300, max_size = 1000000, batch_size=64):
        self.alpha = alpha
        self.tau = tau
        self.beta = beta
        self.batch_size = batch_size
        self.gamma = gamma 
        self.n_actions = n_actions
        print(batch_size,fc1Dms, fc2Dms )
        self.memory = ReplayBuffer(max_size, input_dims , n_actions)
        self.noise = OUActionNoise(mu= np.zeros(n_actions))

        self.actor = ActorNet(alpha=alpha, input_dims=input_dims, n_actions= n_actions,
                                fc1Dms=fc1Dms,fc2Dms=fc2Dms, name = 'actor')
        self.critic = CriticNet(beta=beta,input_dims=input_dims, n_actions=n_actions,
                                fc1Dms=fc1Dms, fc2Dms=fc2Dms, name = 'critic')
        self.target_actor = ActorNet(alpha=alpha, input_dims=input_dims, n_actions= n_actions,
                               fc1Dms=fc1Dms,fc2Dms=fc2Dms, name = 'target_actor')
        self.target_critic = CriticNet(beta=beta, input_dims=input_dims, n_actions=n_actions,
                               fc1Dms=fc1Dms,fc2Dms=fc2Dms,name = 'target_critic')
                               
        self.update_network_parameters(tau=1)


    def choose_action(self, state):
        self.actor.eval() # set the network into evaluation mode (because we are batch norm)
        state = T.tensor([state], dtype = T.float).to(self.actor.device)
        # the actions we got is totally determenistic so we need to add noise 
        mu = self.actor.forward(state).to(self.actor.device) 
        # adding noise to the actor output (states in the paper p4)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        # back to train mode
        self.actor.train()
        # detach() takes the tensor from the cpu, then we convert it to numpy 
        # in order to feed it to the invironment
        return mu_prime.cpu().detach().numpy()[0]


    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 

        states, actions, rewards, new_states, dones = \
                self.memory.sampling(self.batch_size)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.actor.device)
        # print(states.shape)
        # print('actions size inside learn', actions.shape)
        target_actions = self.target_actor.forward(new_states)
        target_critic_value = self.target_critic.forward(new_states, target_actions)
        critic_value = self.critic.forward(states, actions)

        target_critic_value[dones] = 0.0 # make the value of the terminal state =0
        target_critic_value = target_critic_value.view(-1) # not sure why ?? TODO:test


        target = rewards + self.gamma * target_critic_value
        target = target.view(self.batch_size, 1) # convert to the same size as critic_value

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step() 

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        # print("im inside the learn function >>>>>>")
        self.update_network_parameters()
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)
        
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                    (1-tau) * target_critic_state_dict[name].clone()
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                    (1-tau) * target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)    
        self.target_actor.load_state_dict(actor_state_dict)  
        # if batch normalization 
        # self.target_critic.load_state_dict(critic_state_dict, strict=False)    
        # self.target_actor.load_state_dict(actor_state_dict, strict=False)                             
        

            

        
