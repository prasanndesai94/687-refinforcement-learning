import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
import numpy as np
import warnings
import multiprocessing
from collections import deque
import copy
import gymnasium as gym
import warnings
from gymnasium.wrappers import (
    AtariPreprocessing,
    AutoResetWrapper,
    FrameStack,
    RecordEpisodeStatistics,
    RecordVideo,
    TransformReward
)

NUM_CORES = multiprocessing.cpu_count()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Using {DEVICE} device and Cores = {NUM_CORES}")

# warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

# Create environment for Atari games with common preprocessing steps.
def create_atari_env(name, seed, num_envs=1, frame_stack=1, video_path=None, **kwargs):
    env = AtariPreprocessing(gym.make(name, **kwargs), scale_obs=True)
    env = FrameStack(env, num_stack=frame_stack)
    env = TransformReward(env, lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    env = RecordEpisodeStatistics(AutoResetWrapper(env))

    if video_path is not None:
        env = RecordVideo(env, video_folder=video_path)

    env.reset(seed=seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)

    return env

# Function to return action based on Epsilon greedy policy.
def epsilon_greedy_policy(epsilon, value_network, state, num_actions):
  choice = np.random.choice(["random","max"], p=[epsilon, 1-epsilon])
  if choice == "max":
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    q_values = value_network(state_tensor)
    action = torch.argmax(q_values).item()
  else:
    action = np.random.choice(list(range(num_actions)))
  return action

def preprocess(obs):
  return obs

# Function to create and initialize replay memory.
def create_replay_memory(env, replay_memory_size, batch_train_size):
  # Initialise replay memory with capacity N i.e. replay_memory_size
  replay_memory = deque(maxlen=replay_memory_size)

  # List of states which will later be used to track performance
  states_list = []

  # Execute a few actions and bootstrap the replay memory with sample data
  obs, info = env.reset(seed=42)
  state = preprocess(obs)
  states_list.append(state)

  for _ in range(batch_train_size*100):
    action = torch.tensor(env.action_space.sample())
    next_obs, reward, terminated, truncated, info = env.step(action)
    next_state = preprocess(next_obs)

    if terminated:
        replay_memory.append((state, action, -1, next_state, terminated, truncated))
    else:
        replay_memory.append((state, action, reward, next_state, terminated, truncated))
        state = next_state

    states_list.append(next_state)
    state = next_state

    if terminated or truncated:
        observation, info = env.reset()
        state = preprocess(observation)

  return replay_memory, states_list

# Function to anneal/decay epsilon during learning based on number of frames encountered.
def anneal_epsilon(initial_value, final_value, annealing_steps, current_step):
  if current_step < annealing_steps:
    ratio = 1.0 - current_step/annealing_steps
    value = (ratio * initial_value) + ((1-ratio) * final_value)
  else:
    value = final_value

  return value

# CNN-based architecture for Atari.
def create_atari_action_value_network(num_actions, frame_stack=1):
    def create_network(num_outputs):
        return nn.Sequential(
            nn.Conv2d(frame_stack, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, num_outputs, bias=True)
        )

    value_network = create_network(num_actions).to(DEVICE)

    # Initialize the weights using tf.variance_scaling_initializer with scale=2 based on recommendation
    for layer in value_network:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))

    return value_network

def dqn_with_experience_replay(env, value_network, replay_memory_size=1000000,
                               num_training_episodes=5, num_testing_episodes=20, num_epochs=100, num_timesteps_per_episode=10000,
                               epsilon=0.05, batch_train_size=32, annealing_steps=1000000,
                               learning_rate = 0.0003, gamma = 1):

  # Define replay memory with capacity N i.e. replay_memory_size
  replay_memory, states_list = create_replay_memory(env, replay_memory_size, batch_train_size)
  states_list = random.sample(states_list, 100)
  num_frames_encountered = 0
  avg_q_values = []
  avg_q_values.append(0)

  print(f"Replay memory current size: {len(replay_memory)}")

  # Initialise the action-value function and define some model components
  # Loss Function (MSE Loss)
  # loss_fn = nn.MSELoss()
  loss_fn = nn.SmoothL1Loss() # Huber loss recommended to improve training

  # Optimizer to define the method of updating model parameters. As per the DQN paper, they've used RMSProp
  optimizer = torch.optim.RMSprop(value_network.parameters(), lr=learning_rate)

  average_rewards_per_episode = []

  for epoch_id in tqdm(range(num_epochs)):
    episodic_rewards = []

    for episode_id in tqdm(range(num_training_episodes)):

      episode_rewards = 0
      terminated = False

      # Initial pre-processed state (can sometimes be a stack of frames)
      num_actions = env.action_space.n
      current_obs, _ = env.reset()
      current_state = preprocess(current_obs)
      num_frames_encountered += 1
      target_value_network = copy.deepcopy(value_network) if episode_id % 2 == 0 else target_value_network
      torch.no_grad()

      while not terminated:

        # Calculate the annealed value of epsilon based on number of frames encountered so far
        annealed_epsilon = anneal_epsilon(1, epsilon, annealing_steps, num_frames_encountered)

        # Calculate the action based on epsilon greediness
        action = epsilon_greedy_policy(annealed_epsilon, value_network, current_state, num_actions)

        # Calculate the next state by running the action in the env emulator to get the next state, reward.
        next_obs, reward, terminated, truncated, info = env.step(action) # or action.cpu()
        episode_rewards += reward
        next_state = preprocess(next_obs)
        num_frames_encountered += 1

        # Store transition in replay memory and update the current state
        replay_memory.append((current_state, action, reward, next_state, terminated, truncated))

        if terminated:
          # Store transition in replay memory and update the current state, -1 reward
          replay_memory.append((current_state, action, -1, next_state, terminated, truncated))
        else:
          # Store transition in replay memory and update the current state
          replay_memory.append((current_state, action, reward, next_state, terminated, truncated))
          current_state = next_state

        if truncated:
          break

        # Sample random minibatch of transitions from replay memory
        random_rm_minibatch = random.sample(replay_memory, batch_train_size)

        # Improve value_network
        torch.enable_grad()

        minibatch_dict = {}
        minibatch_dict['current_state'] = []
        minibatch_dict['action'] = []
        minibatch_dict['reward'] = []
        minibatch_dict['next_state'] = []
        minibatch_dict['done'] = []

        for item in random_rm_minibatch:
          minibatch_dict['current_state'].append(item[0])
          minibatch_dict['action'].append(item[1])
          minibatch_dict['reward'].append(item[2])
          minibatch_dict['next_state'].append(item[3])
          minibatch_dict['done'].append(item[4])

        state_batch = torch.tensor(minibatch_dict['current_state'], dtype=torch.float32).to(DEVICE)
        action_batch = torch.tensor(minibatch_dict['action'], dtype=torch.long).to(DEVICE)
        next_state_batch = torch.tensor(minibatch_dict['next_state'], dtype=torch.float32).to(DEVICE)
        reward_batch = torch.tensor(minibatch_dict['reward'], dtype=torch.float32).to(DEVICE)
        done_batch = torch.tensor(minibatch_dict['done'], dtype=torch.float32).to(DEVICE)

        # Compute Q-values and target Q-values
        q_values = value_network(state_batch).gather(1, action_batch.unsqueeze(1))
        target_q_values = reward_batch + gamma * (1 - done_batch) * torch.max(target_value_network(next_state_batch), dim=1).values.unsqueeze(1)

        # Compute the loss
        loss = loss_fn(q_values, target_q_values.detach())
        # print(f"Loss: {loss}")
        # Perform mini-batch gradient descent step and backward prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      print(f"Episode {episode_id+1} rewards: {episode_rewards}, Truncated:{truncated}, Terminated:{terminated}")
      episodic_rewards.append(episode_rewards)

    average_rewards_per_episode.append(np.mean(episodic_rewards))

    # Performance check
    q_values_predicted = []
    states_sample = torch.tensor(states_list, dtype=torch.float32).to(DEVICE)
    q_values_predicted = torch.max(value_network(states_sample), dim=1).values.unsqueeze(1).view(-1).cpu().detach().numpy()
    avg_q_value = np.mean(q_values_predicted)
    avg_q_values.append(avg_q_value)

    print(f"\nEpoch number={epoch_id}, Average={avg_q_value}, Number of frames={num_frames_encountered}")
    print(f"\nAvg_training_rewards={average_rewards_per_episode[-1]}")

  return value_network, avg_q_values, average_rewards_per_episode

game_name = "ALE/Breakout-v5"
frame_stack = 4

env = create_atari_env(game_name, seed=2023, num_envs=NUM_CORES, frame_stack=frame_stack, frameskip=1)
value_network = create_atari_action_value_network(env.action_space.n, frame_stack=frame_stack)

trained_network, avg_q_values_breakout, average_rewards_per_episode_breakout = dqn_with_experience_replay(env, value_network, replay_memory_size=100000,
                              num_training_episodes=50, num_testing_episodes=10, num_epochs=100, num_timesteps_per_episode=250,
                               epsilon=0.01, batch_train_size=32, annealing_steps=50000,
                               learning_rate = 0.0001, gamma = 0.99)

# To plot Average Q over time
game_name = "ALE/Breakout-v5"
metric="Average Q"
ymax=max(avg_q_values_breakout)
ylabel="Average Action Value (Q)"
cmapval = "YlOrBr"

plt.figure(figsize=(8, 6))
plt.scatter(range(len(avg_q_values_breakout)), avg_q_values_breakout, c=avg_q_values_breakout, cmap=cmapval, label=ylabel, marker='.')
plt.xticks(np.arange(0, 101, 10))
plt.yticks(np.arange(min(avg_q_values_breakout), ymax+0.5, 0.5))
plt.xlabel('Training Epochs')
plt.ylabel(f'{ylabel}')
plt.title(f'{metric} on {game_name.split("/")[1].split("-")[0]}')
plt.show()

# To plot Average Episodic Rewards
game_name = "ALE/Breakout-v5"
metric="Average Reward"
ymax=max(average_rewards_per_episode_breakout)
ylabel="Average Reward Per Episode"
cmapval = "PuRd"
metric_list = average_rewards_per_episode_breakout
plt.figure(figsize=(8, 6))
plt.plot(range(len(metric_list)), metric_list, c='#FF7940', label=ylabel, marker='.')
plt.xticks(np.arange(0, 101, 10))
plt.yticks(np.arange(-21, 10, 5))
plt.xlabel('Training Epochs')
plt.ylabel(f'{ylabel}')
plt.title(f'{metric} on {game_name.split("/")[1].split("-")[0]}')
plt.show()

