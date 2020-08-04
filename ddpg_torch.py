# %%
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# %%
torch.set_default_dtype(torch.float64)

# %%
problem = "Pendulum-v0"
env = gym.make(problem)

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

# %%
from ou_action_noise import OUActionNoise

# %%
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = torch.tensor(self.state_buffer[batch_indices])
        action_batch = torch.tensor(self.action_buffer[batch_indices])
        reward_batch = torch.tensor(self.reward_buffer[batch_indices])
        next_state_batch = torch.tensor(self.next_state_buffer[batch_indices])

        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        critic_optimizer.zero_grad()

        target_actions = target_actor(next_state_batch)
        y = reward_batch + gamma * target_critic([next_state_batch, target_actions])
        critic_value = critic_model([state_batch, action_batch])
        critic_loss = torch.mean(torch.square(y - critic_value))

        critic_loss.backward()
        critic_optimizer.step()

        actor_optimizer.zero_grad()

        actions = actor_model(state_batch)
        critic_value = critic_model([state_batch, actions])
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -torch.mean(critic_value)

        actor_loss.backward()
        actor_optimizer.step()

    def __len__(self):
        return self.buffer_counter


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(tau):
    for target_param, param in zip(target_critic.parameters(), critic_model.parameters()):
        target_param.data.copy_(
            param.data * tau + target_param.data * (1.0 - tau)
        )

    for target_param, param in zip(target_actor.parameters(), actor_model.parameters()):
        target_param.data.copy_(
            param.data * tau + target_param.data * (1.0 - tau)
        )

# %%
class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_states, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1),
            nn.Tanh()
        )

        self.model[-2].weight.data.uniform_(-0.003, 0.003)

    def forward(self, inputs):
        return self.model(inputs) * upper_bound


def get_actor():
    return Actor()

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.state_model = nn.Sequential(
            nn.Linear(num_states, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.action_model = nn.Sequential(
            nn.Linear(num_actions, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.out_model = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1)
        )

    def forward(self, inputs):
        model_input = self.state_model(inputs[0])
        action_input = self.action_model(inputs[1])

        return self.out_model(torch.cat([model_input, action_input], dim=1))

def get_critic():
    return Critic()

# %%
def policy(state, noise_object):
    actor_model.eval()
    sampled_actions = actor_model(state).squeeze()
    actor_model.train()

    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.detach().numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


# %%
# Hyperparameters
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.load_state_dict(actor_model.state_dict())
target_critic.load_state_dict(critic_model.state_dict())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = optim.Adam(critic_model.parameters(), lr=critic_lr)
actor_optimizer = optim.Adam(actor_model.parameters(), lr=actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)

# %%
# Training loop
# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 20 min to train
for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        env.render()

        torch_prev_state = torch.tensor(prev_state).unsqueeze(dim=0)

        action = policy(torch_prev_state, ou_noise)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

# %%
