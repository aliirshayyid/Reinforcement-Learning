import os
import numpy as np 
import torch as T
import torch.nn.functional as F
from Networks import CriticNet, ActorNet
from Replay_Buffer import ReplayBuffer

# T.manual_seed(0)
# np.random.seed(0)

class TD3Agent():
    def __init__(self, alpha, beta, input_dims, tau, env, n_actions=2,
            gamma=0.99, update_actor_interval=2,fc1Dms=400, fc2Dms=300,
            max_size=1000000, batch_size=100,warmup=1000, noise=0.1):
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.gamma = gamma 
        self.n_actions = n_actions
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.update_actor_iter = update_actor_interval

        self.critic_1 = CriticNet(beta, input_dims, n_actions, fc1Dms, fc2Dms,
                    name='Critic1')
        self.critic_2 = CriticNet(beta, input_dims, n_actions, fc1Dms, fc2Dms,
                    name='Critic2')
        self.actor = ActorNet(alpha, input_dims, n_actions, fc1Dms,
                    fc2Dms, name='actor')
        # target nets
        self.target_critic_1 = CriticNet(beta, input_dims, n_actions, fc1Dms, fc2Dms,
                    name='Target_critic1')
        self.target_critic_2 = CriticNet(beta, input_dims, n_actions, fc1Dms, fc2Dms,
                    name='Target_critic2')
        self.target_actor = ActorNet(alpha, input_dims, n_actions, fc1Dms,
                    fc2Dms, name='Target_actor')
        self.noise = noise 
        # set the target nets to be exactly as our nets
        self.update_network_parameters(tau=1)


    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward,
                                new_state, done)

    def choose_action(self, state):
        # check if we are in time after the warmup
        if self.time_step < self.warmup:
            # scale is the standard deviation
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)))
        else:
            # print("the warmup is done")
            state = T.tensor(state, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
            
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise),
                     dtype=T.float).to(self.actor.device)

        # we want to make sure the mu is not out of the max action the env can take
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step +=1
        
        # action.shape= (2,)
        # print(action)
        return mu_prime.cpu().detach().numpy()
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            # print("not learning")
            return 
        
        states, actions, rewards, new_states, dones = \
                self.memory.sampling(self.batch_size)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.actor.device)

        # regularization
        # might breake if elements of min and max are not equal
        target_action = self.target_actor.forward(new_states) + \
                T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_action = T.clamp(target_action, self.min_action[0], self.max_action[0])

        target_critic1_q = self.target_critic_1.forward(new_states, target_action)
        target_critic2_q = self.target_critic_2.forward(new_states, target_action)

        target_critic1_q[dones] = 0
        target_critic2_q[dones] = 0

        target_critic1_q = target_critic1_q.view(-1)
        target_critic2_q = target_critic2_q.view(-1)

        q1 = self.critic_1.forward(states, actions)
        q2 = self.critic_2.forward(states, actions)
        

        y = rewards + self.gamma * T.min(target_critic1_q, target_critic2_q)
        y = y.view(self.batch_size, 1)

        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(y, q1)
        q2_loss = F.mse_loss(y, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr +=1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return 
        # print("learning>>")
        self.actor.optimizer.zero_grad()
        actor_loss = self.critic_1.forward(states, self.actor.forward(states))
        actor_loss = -T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        target_actor_params = self.target_actor.named_parameters()

        critic1_params = self.critic_1.named_parameters()
        critic2_params = self.critic_2.named_parameters()
        target_critic1_params = self.target_critic_1.named_parameters()
        target_critic2_params = self.target_critic_2.named_parameters()

        critic1_state_dict = dict(critic1_params)
        critic2_state_dict = dict(critic2_params)
        target_critic1_state_dict = dict(target_critic1_params)
        target_critic2_state_dict = dict(target_critic2_params)

        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)

        # for name in target_actor_state_dict:
        #     target_actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
        #             (1-tau) * target_actor_state_dict[name].clone()

        # for name in target_critic1_state_dict:
        #     target_critic1_state_dict[name] = tau * critic1_state_dict[name].clone() +\
        #             (1-tau) * target_critic1_state_dict[name].clone()

        # for name in target_critic2_state_dict:
        #     target_critic2_state_dict[name] = tau * critic2_state_dict[name].clone() +\
        #             (1-tau) * target_critic2_state_dict[name].clone()

           
        # self.target_actor.load_state_dict(target_actor_state_dict) 
        # self.target_critic_1.load_state_dict(target_critic1_state_dict)
        # self.target_critic_2.load_state_dict(target_critic2_state_dict)


        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau) * target_actor_state_dict[name].clone()

        for name in critic1_state_dict:
            critic1_state_dict[name] = tau*critic1_state_dict[name].clone() + \
                    (1-tau) * target_critic1_state_dict[name].clone()

        for name in critic2_state_dict:
            critic2_state_dict[name] = tau*critic2_state_dict[name].clone() + \
                    (1-tau) * target_critic2_state_dict[name].clone()

           
        self.target_actor.load_state_dict(actor_state_dict) 
        self.target_critic_1.load_state_dict(critic1_state_dict)
        self.target_critic_2.load_state_dict(critic2_state_dict)


    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()















        
