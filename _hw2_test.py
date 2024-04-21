# %%
# !pip install gym_super_mario_bros
# !pip install gym==0.23.1

# %%
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import torch
import math
import cv2


ACTION_SIZE = 12  # Number of valid actions in the game
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.0001  # Learning rate
BATCH_SIZE = 32  # Batch size for training
MEMORY_SIZE = 20000  # Size of the replay memory buffer

class NoisyLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = torch.nn.Parameter(torch.full((out_features, in_features), sigma_init), requires_grad=True)
        self.sigma_bias = torch.nn.Parameter(torch.full((out_features,), sigma_init), requires_grad=True)
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        # if self.training:
        weight = self.weight + self.sigma_weight * self.epsilon_weight.data.to(self.weight.device)
        bias = self.bias + self.sigma_bias * self.epsilon_bias.data.to(self.bias.device)
        # else:
        #     weight = self.weight
        #     bias = self.bias

        output = torch.nn.functional.linear(input, weight, bias)
        return output

    def reset_noise(self):
        epsilon_weight = torch.randn(self.out_features, self.in_features)
        epsilon_bias = torch.randn(self.out_features)
        self.epsilon_weight = torch.nn.Parameter(epsilon_weight, requires_grad=False)
        self.epsilon_bias = torch.nn.Parameter(epsilon_bias, requires_grad=False)

# Define the Double DQN model
class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, stride=3)
        self.conv3 = torch.nn.Conv2d(in_channels=12, out_channels=14, kernel_size=3, stride=2)
        self.fc1 = NoisyLinear(14 * 9 * 10, 256)
        self.value_stream = NoisyLinear(256, 1)
        self.advantage_stream = NoisyLinear(256, ACTION_SIZE)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        print("after conv1:", x.shape)
        x = torch.relu(self.conv2(x))
        print("after conv2:", x.shape)
        x = torch.relu(self.conv3(x))
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        value = self.value_stream(x)
        adv = self.advantage_stream(x)
        adv_average = torch.mean(adv, dim=1, keepdim=True)
        q_values = value + adv - adv_average
        return q_values

    def reset_noise(self):
        self.fc1.reset_noise()
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()

    def save_mu_state_dict(self):
        state_dict = self.state_dict()
        mu_state_dict = {}
        for key, value in state_dict.items():
            if 'sigma' not in key and 'epsilon' not in key:
                mu_state_dict[key] = value
            else:
                mu_state_dict[key] = torch.zeros_like(value)
        return mu_state_dict
    
    def load_mu_state_dict(self, state_dict, strict=True):
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if 'sigma' not in key and 'epsilon' not in key:
                filtered_state_dict.update({key: value})
                print(value.type)
            else:
                filtered_state_dict.update({key: torch.empty(value.shape)})
        super().load_state_dict(filtered_state_dict, strict)

class Agent:
    def __init__(self):
        
        # self.memory = module.ReplayMemory_Per(capacity=MEMORY_SIZE)
        self.model = DQN()#.to(device)
        self.target_model = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.steps_counter = 0

        self.frames_counter = 0

        self.stacked_img = None
        self.stacked_img_buf = None
        self.prev_action = 0 # initialize as NOOP
        self.pick_action_flag = False

        self.load("110060062_hw2_data.py")

    def init_target_model(self): # used only before training
        self.target_model = DQN()#.to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        for param in self.target_model.parameters():
            param.requires_grad = False

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def act(self, observation):
        # grayscale the image
        print(observation.shape)
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
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

            return self.prev_action
        
        else: # self.frames_counter == 12
            
            # stack image
            self.stacked_img = np.concatenate((self.stacked_img, observation), axis=2)
            self.stacked_img = np.int8(self.stacked_img)
            self.stacked_img = torch.from_numpy(self.stacked_img).float()
            self.stacked_img = self.stacked_img.permute(2, 0, 1)
            self.stacked_img = self.stacked_img.unsqueeze(0)#.to(device)
            
            # pick new action
            # if np.random.rand() <= self.epsilon:
            #     new_action = np.random.randint(0, 12)
            # else:
            q_values = self.model(self.stacked_img)
            new_action = q_values.max(1)[1].item()

            # update member variables
            self.stacked_img_buf = self.stacked_img.squeeze(0).to(torch.int8)
            self.stacked_img = None
            self.prev_action = new_action
            self.pick_action_flag = True

            # update frames counter
            self.frames_counter = 0

            return new_action

    def replay(self):
        if self.memory.size() < BATCH_SIZE:
            return

        idxs, priorities, sample_datas = self.memory.sample(BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*sample_datas)

        # compute weights for loss update
        weights = np.power(np.array(priorities) + self.memory.e, -self.memory.a)
        weights /= weights.max()
        weights = torch.from_numpy(weights).float().to(device)

        states = torch.stack(states).float()
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).float().to(device)
        next_states = torch.stack(next_states).float()
        dones = torch.FloatTensor(dones).float().to(device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = (rewards + GAMMA * next_q_values * (1 - dones))
        loss = (weights * torch.nn.MSELoss()(q_values, expected_q_values)).mean()
        
        wandb.log({"loss": loss.item()})
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1000, norm_type=2)
        self.optimizer.step()

        # update PER
        td_errors = (q_values - expected_q_values).detach().squeeze().tolist()
        self.memory.update(idxs, td_errors)

        if self.steps_counter % 10000 == 0:
            print("copy!")
            self.target_model.load_state_dict(self.model.state_dict())

        self.steps_counter += 1
        self.model.reset_noise()
        self.target_model.reset_noise()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        weights = self.model.state_dict()
        torch.save(weights, name)

if __name__ == '__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    state = env.reset()

    agent = Agent()
    
    agent.model.eval()
    done = False
    total_reward = 0
    for step in range(5000):
        if done:
            state = env.reset()
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            state = np.expand_dims(state, axis=2)
            print(total_reward)
            total_reward = 0
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

    env.close()
