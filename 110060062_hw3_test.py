# python related
import numpy as np
import random
from collections import deque

# training related
import torch

# gym related
import gym
import gym_multi_car_racing

# others
import cv2
import wandb

ACTION_SPACE_SIZE = 3
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.0003
BATCH_SIZE = 128  # Batch size for training
MEMORY_SIZE = 50000  # Size of the replay memory buffer

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class ReplayMemory_Per(object):
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity=MEMORY_SIZE, a=0.6, e=0.01):
        self.tree = SumTree(capacity)
        self.memory_size = capacity
        self.prio_max = 0.1
        self.a = a
        self.e = e

    def push(self, transition):
        p = (np.abs(self.prio_max) + self.e) ** self.a  # proportional priority
        self.tree.add(p, transition)

    def sample(self, batch_size):
        idxs = []
        priorities = []
        sample_datas = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            if not isinstance(data, tuple):
                print(idx, p, data, self.tree.write)
            idxs.append(idx)
            priorities.append(p)
            sample_datas.append(data)
        return idxs, priorities, sample_datas

    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries
    
class QNetwork(torch.nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=3)

        self.fc1 = torch.nn.Linear(16 * 7 * 7 + 3, 256) # instead of concate action after fc1
        self.fc2 = torch.nn.Linear(256, 1) # instead of concate action after fc1
        # self.fc3 = torch.nn.Linear(64, 1)

        self.init_weight()

    def init_weight(self):
        # Initialize weights using Xavier initialization
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv3.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        # Initialize biases to zero
        self.conv1.bias.data.fill_(0)
        self.conv2.bias.data.fill_(0)
        self.conv3.bias.data.fill_(0)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)

    def forward(self, state, action):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # print(x.shape)
        x = x.reshape(x.size(0), -1) # flatten
        x = torch.cat([x, action], 1) # concatenate with action
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc2(x)
        return x
    
class CriticNetwork(torch.nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.q_net_1 = QNetwork()
        self.q_net_2 = QNetwork()

    def forward(self, state, action):
        action = action.clone()
        q_value_1 = self.q_net_1.forward(state, action)
        q_value_2 = self.q_net_2.forward(state, action)
        return q_value_1, q_value_2
    
class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=3)

        self.fc1 = torch.nn.Linear(16 * 7 * 7, 256)
        self.fc_mean = torch.nn.Linear(256, 3)
        self.fc_log_std = torch.nn.Linear(256, 3)

        # self.init_weight()

    def init_weight(self):
        # Initialize weights using Xavier initialization
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.conv3.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.fc_mean.weight)
        torch.nn.init.xavier_uniform_(self.fc_log_std.weight)

        # Initialize biases to zero
        self.conv1.bias.data.fill_(0)
        self.conv2.bias.data.fill_(0)
        self.conv3.bias.data.fill_(0)
        self.fc1.bias.data.fill_(0)
        self.fc_mean.bias.data.fill_(0)
        self.fc_log_std.bias.data.fill_(0)

    def forward(self, state):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) # flatten
        x = torch.relu(self.fc1(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        # round_mean = [round(x, 2) for x in mean[0].tolist()]
        # round_std = [round(x, 2) for x in log_std[0].exp().tolist()]
        # if np.random.rand() < 0.01:
        #     print("mean:", round_mean, "log std:", round_std) # => always 0 and 1
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # the reparameterization trick, (x_t is tensor)
        
        y_t = torch.tanh(x_t)
        action = torch.zeros_like(y_t)
        action[:, 0] = y_t[:, 0]
        action[:, 1:] = (y_t[:, 1:] + torch.tensor(1.)) * torch.tensor(0.5) # shift for gas and brake

        # Enforcing Action Bound (pranz24 ver)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True) # [batch, 3] -> [batch, 1]
        
        # spinning up ver
        # log_prob = normal.log_prob(x_t)
        # log_prob -= (2*(np.log(2) - x_t - torch.nn.functional.softplus(-2*x_t)))
        # log_prob = log_prob.sum(axis=1, keepdim=True)
        # print(action.shape, log_prob.shape)

        # change action = sampled or tanh(mean) according to mode
        if not self.training:
            action = torch.tanh(mean)
        
        # if np.random.rand() < 0.01:
        #     round_action = [round(x, 2) for x in action[0].tolist()]
        #     print("action:", round_action) 
        
        return action, log_prob 

class Agent:
    def __init__(self):

        self.alpha = 0.2
        self.tau = 0.001
        self.update_interval = 1
        # critic network
        self.critic = CriticNetwork().to(device)
        self.critic_target = None
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

        # actor network
        self.policy = PolicyNetwork().to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        ## entropy
        self.target_entropy = -torch.prod(torch.Tensor(ACTION_SPACE_SIZE).to(device)).item()
        ## alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=LEARNING_RATE)

        # replay buffer
        # self.memory = deque(maxlen=MEMORY_SIZE)
        # self.memory = ReplayMemory_Per(capacity=MEMORY_SIZE)

        # others
        ## counters
        self.steps_counter = 0
        self.frames_counter = 0
        ## temps needed for replay()
        self.stacked_img = None
        self.stacked_img_buf = None
        self.prev_action = [0, 1, 0] # initialize as gas = 1
        self.pick_action_flag = False

        self.load_test("110060062_hw3_data.py")
        
    def init_target_model(self): # used only before training
        self.critic_target = CriticNetwork().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False
    
    def act(self, observation):
        # grayscale the image
        observation = np.squeeze(observation)
        observation = cv2.cvtColor(observation.astype(np.float32), cv2.COLOR_RGB2GRAY)
        observation = np.expand_dims(observation, axis=2)

        if self.frames_counter != 12:
            
            # stack image
            if self.frames_counter == 0:
                self.stacked_img = observation
            elif self.frames_counter % 4 == 0:
                self.stacked_img = np.concatenate((self.stacked_img, observation), axis=2)

            # update member variables
            self.pick_action_flag = False

            # update frames counter
            self.frames_counter += 1

        
        else: # self.frames_counter == 12

            # stack image
            self.stacked_img = np.concatenate((self.stacked_img, observation), axis=2)
            self.stacked_img = np.int8(self.stacked_img)
            self.stacked_img = torch.from_numpy(self.stacked_img).float() # change to float when inferencing
            self.stacked_img = self.stacked_img.permute(2, 0, 1)
            self.stacked_img = self.stacked_img.unsqueeze(0).to(device)

            # pick new action
            new_action, _ = self.policy.sample(self.stacked_img)

            # update member variables
            self.stacked_img_buf = self.stacked_img.squeeze(0)
            self.stacked_img = None
            self.prev_action = new_action.detach().cpu().numpy()[0]
            self.pick_action_flag = True
            # update frames counter
            self.frames_counter = 0

        return self.prev_action
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # self.memory.push((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        # idxs, priorities, minibatch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # # compute weights for loss update
        # weights = np.power(np.array(priorities) + self.memory.e, -self.memory.a)
        # weights /= weights.max()
        # weights = torch.from_numpy(weights).float().to(device)

        states = torch.stack(states).float().to(device)
        actions = torch.stack(actions).float().to(device)
        rewards = torch.FloatTensor(rewards).float().unsqueeze(1).to(device)
        next_states = torch.stack(next_states).float().to(device)
        dones = torch.FloatTensor(dones).float().unsqueeze(1).to(device)

        # update critic
        with torch.no_grad(): # according to paper equation 6
            next_states_action, next_states_log_pi = self.policy.sample(next_states) # a_t+1, log_pi(a_t+1)
            q1_next_target, q2_next_target = self.critic_target(next_states, next_states_action) # Q(s_t+1, a_t+1)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_states_log_pi # bootstrap = Q(s_t+1, a_t+1) - alpha * log_pi(a_t+1)
            next_q_value = rewards + GAMMA * min_q_next_target * (1 - dones) # y = r_t + gamma * bootstrap * (1-dones)
        q1, q2 = self.critic(states, actions) # Q(s_t, a_t)
        # q1_loss = (weights * torch.nn.MSELoss()(q1, next_q_value)).mean()
        # q2_loss = (weights * torch.nn.MSELoss()(q2, next_q_value)).mean()
        q1_loss = torch.nn.MSELoss()(q1, next_q_value)
        q2_loss = torch.nn.MSELoss()(q2, next_q_value)
        q_loss = q1_loss + q2_loss # another version is, they back prop separately
        
        wandb.log({"q loss": q_loss.item()})
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # update PER
        # td_errors = (torch.min(q1, q2) - next_q_value).detach().squeeze().tolist()
        # self.memory.update(idxs, td_errors)

        for p in self.critic.parameters(): # freeze q
            p.requires_grad = False

        # update policy
        pis, log_pis,  = self.policy.sample(states) # pi(s_t), log_pi(a_t)
        q1_pi, q2_pi = self.critic(states, pis)
        min_q_pi = torch.min(q1_pi, q2_pi) # Q(s_t, f)

        # with torch.autograd.set_detect_anomaly(True):
        policy_loss = ((self.alpha * log_pis) - min_q_pi).mean() # J_pi = E[(alpha * log_pi) - Q]
        wandb.log({"pi loss": policy_loss.item()})
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for p in self.critic.parameters(): # unfreeze q
            p.requires_grad = True

        # update alpha
        alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean() # not on paper
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        if self.steps_counter % self.update_interval == 0:
            self.soft_update(self.critic_target, self.critic, self.tau)
        
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def load_test(self, pi_name):
        self.policy.eval()
        self.policy.load_state_dict(torch.load(pi_name))
        # self.critic.load_state_dict(torch.load(name, map_location=torch.device('cpu')))

    def load_cont_train(self, pi_name, q_name):
        self.policy.load_state_dict(torch.load(pi_name))
        self.critic.load_state_dict(torch.load(q_name))

    def save(self, pi_name, q_name):
        policy_weights = self.policy.state_dict()
        torch.save(policy_weights, pi_name)
        critic_weights = self.critic.state_dict()
        torch.save(critic_weights, q_name)

if __name__ == '__main__':
    env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)
    obs = env.reset()

    agent = Agent()
    agent.load_test("car_racing_pi_half.pt")
    agent.policy.eval()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

    print("individual scores:", total_reward)
    env.close()