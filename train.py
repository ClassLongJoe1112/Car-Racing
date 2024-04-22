import gym
import gym_multi_car_racing
import numpy as np
import torch

import wandb

from test import Agent

LOG_FREQ = 100

n_episode = 1000

if __name__ == '__main__':

    wandb.init(project="car-racing", mode="disabled")

    # declare the environment
    env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
            use_random_direction=True, backwards_flag=True, h_ratio=0.25,
            use_ego_color=False)
    
    # declare the agent
    agent = Agent()
    agent.policy.train()
    agent.init_target_model()
    wandb.watch(agent.critic, log_freq=LOG_FREQ)
    
    # For transition storing (temp = last 4 frames = "current state" in a trans)
    state_stack_temp = None
    action_temp = None
    reward_temp = None
    done_temp = None

    # For reward curve checking
    reward_per_5epi = 0

    for episode in (range(n_episode)):

        obs = env.reset()
        done = False
        first_action = True
        reward_per_episode = 0
        reward_per_4frames = 0

        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            reward_per_4frames += reward
            # Store transition every time the agent pick action
            '''
            logic:
                When agent picks action, that means a 4 frames stack is stored in the agent.
                if it is the first action among the whole episode, just store the frames to the temp,
                which will become the "state" of the transition when agent pick the next action

                if it's not the first action, 
                    "state" = the stored frames
                    "next_state" = the currently facing frames
                    at last, update "state" frames = "next_state" frames

            '''
            if agent.pick_action_flag:
                if first_action:

                    first_action = False

                    # just store to temp
                    state_stack_temp = agent.stacked_img_buf.to(torch.int8)
                    action_temp = action
                    reward_temp = reward_per_4frames
                    done_temp = done

                    reward_per_4frames = 0

                else:
                    # actually store and also let agent "remember"
                    next_state_stack = agent.stacked_img_buf.to(torch.int8)
                    action_temp = torch.tensor(action_temp)
                    reward_temp = torch.tensor(reward_temp).to(torch.int8)
                    done_temp = torch.tensor(done_temp).to(torch.int8)

                    agent.remember(state_stack_temp, \
                                    action_temp, \
                                    reward_temp, \
                                    next_state_stack, \
                                    done_temp)
                    
                    # update temp
                    state_stack_temp = next_state_stack
                    action_temp = action
                    reward_temp = reward_per_4frames
                    done_temp = done

                    agent.replay()
                    reward_per_4frames = 0
                    
            reward_per_episode += reward

        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Episode: {episode}, Score: {reward_per_episode}")

        reward_per_5epi += reward_per_episode
        if episode % 5 == 0:
            wandb.log({"score per 5 epi": reward_per_5epi / 5})
            reward_per_5epi = 0
            
        if episode % n_episode/3 == 0:
            print("saved checkpoint!")
            agent.save("car_racing.pt")

    # Save the trained model
    agent.save("car_racing.pt")

    env.close()
    # print("individual scores:", total_reward)