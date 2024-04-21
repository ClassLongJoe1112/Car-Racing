# python related
import numpy as np
import random
from collections import deque

# training related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# gym related
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import wandb

from _hw2_test import Agent

n_episode = 1

# Environment and model hyperparameters
STATE_SIZE = [240, 256, 1]  # Dimensions of the game observation
ACTION_SIZE = 12  # Number of valid actions in the game
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.0001  # Learning rate
BATCH_SIZE = 32  # Batch size for training
MEMORY_SIZE = 20000  # Size of the replay memory buffer

LOG_FREQ = 100

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

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    wandb.init(project="mario-with-dqn", mode="disabled")
    # Create the environment
    # env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0', stages=['1-1'])
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    # Create the agent
    agent = Agent()
    agent.model.train()
    # agent.load("noisy_per_step_nogradtarget_.pt")
    agent.init_target_model()
    wandb.watch(agent.model, log_freq=LOG_FREQ)
    
    state_stack_temp = None
    action_temp = None
    reward_temp = None
    done_temp = None

    show_state = None
    show_next_state = None
    score_per_5_episode = 0
    # show_flag = True

    # Train the agent
    for episode in (range(n_episode)):
        state = env.reset()
        done = False
        score = 0
        first_action = True
        _4_frames_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            _4_frames_reward += reward

            if agent.pick_action_flag:
                if first_action:
                    
                    first_action = False

                    # just store
                    state_stack_temp = agent.stacked_img_buf
                    action_temp = action
                    reward_temp = _4_frames_reward
                    done_temp = done

                    _4_frames_reward = 0
                else:
                    # call remember() (to remember the last transition)
                    next_state_stack = agent.stacked_img_buf
                    action_temp = torch.tensor(action_temp).to(torch.int8)
                    reward_temp = torch.tensor(reward_temp).to(torch.int8)
                    done_temp = torch.tensor(done_temp).to(torch.int8)

                    agent.remember(state_stack_temp, \
                                    action_temp, \
                                    reward_temp, \
                                    next_state_stack, \
                                    done_temp)
                    
                    # for plotting
                    # if show_flag and np.random.rand() < 0.01:
                    #     show_flag = False
                    #     show_state = state_stack_temp.cpu().numpy()
                    #     show_next_state = next_state_stack.cpu().numpy()

                    # store
                    state_stack_temp = next_state_stack
                    action_temp = action
                    reward_temp = _4_frames_reward
                    done_temp = done

                    agent.replay()
                    _4_frames_reward = 0

            state = next_state
            score += reward
        print(f"Episode: {episode}, Score: {score}")

        
        # if agent.epsilon > EPSILON_END: # not used in noisy net
        #     agent.epsilon *= EPSILON_DECAY

        score_per_5_episode += score
        if episode % 5 == 0:
            wandb.log({"score per 5 epi": score_per_5_episode / 5})
            score_per_5_episode = 0

    # Save the trained model
    agent.save("noisy_per_step_nogradtarget____.pt")

    env.close()

    # matplotlib.use('TkAgg')
    # fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # for i in range(2):
    #     for j in range(4):
    #         if i == 0:
    #             axes[i, j].imshow(show_state[j], cmap='gray')
    #         else:
    #             axes[i, j].imshow(show_next_state[j], cmap='gray')
    # plt.axis('off')  # Turn off axis
    # plt.tight_layout()
    # plt.show()
    # Close the environment