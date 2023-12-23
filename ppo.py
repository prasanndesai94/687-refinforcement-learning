import gymnasium as gym
import multiprocessing
import numpy as np
import pandas as pd
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from gymnasium.wrappers import (
    AtariPreprocessing,
    AutoResetWrapper,
    FrameStack,
    RecordEpisodeStatistics,
    RecordVideo,
    TransformReward,
)
from torch.utils.data import DataLoader, StackDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from torchviz import make_dot

warnings.filterwarnings("ignore", category=DeprecationWarning)
NUM_CORES = multiprocessing.cpu_count()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Create vectorized environments for Atari games with common preprocessing steps
def create_atari_env(name, seed, num_envs=1, frame_stack=1, video_path=None, **kwargs):
    wrappers = [
        RecordEpisodeStatistics,
        AutoResetWrapper,
        lambda env: AtariPreprocessing(env, scale_obs=True),
        lambda env: FrameStack(env, num_stack=frame_stack),
        lambda env: TransformReward(env, lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)),
    ]

    if video_path is not None:
        wrappers.append(lambda env: RecordVideo(env, video_folder=video_path))

    env = gym.vector.make(name, num_envs=num_envs, wrappers=wrappers, **kwargs)
    env.reset(seed=[seed + i for i in range(num_envs)])
    env.observation_space.seed(seed)
    env.single_observation_space.seed(seed)
    env.action_space.seed(seed)
    env.single_action_space.seed(seed)

    return env

# Based on recommendations from https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details
def init_layer(layer, std=np.sqrt(2.0), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer

class Agent(nn.Module):
    def __init__(self, num_actions, frame_stack=1):
        super(Agent, self).__init__()
        self.base_net = nn.Sequential(
            init_layer(nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)), # 32 x 20 x 20
            nn.ReLU(),
            init_layer(nn.Conv2d(32, 64, kernel_size=4, stride=2)), # 64 x 9 x 9
            nn.ReLU(),
            init_layer(nn.Conv2d(64, 64, kernel_size=3, stride=1)), # 64 x 7 x 7
            nn.ReLU(),
            nn.Flatten(),
            init_layer(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU()
        )
        self.policy_head = init_layer(nn.Linear(512, num_actions), std=0.01)
        self.value_head = init_layer(nn.Linear(512, 1), std=1.0)

    def get_policy_and_value(self, states):
        base = self.base_net(states)
        return self.policy_head(base), self.value_head(base)
    
    def get_value(self, states):
        return self.value_head(self.base_net(states))

def ppo(env, agent, iterations=1, steps=128, epochs=1, batch_size=32,
        lr=2.5e-04, clip_eps=0.1, gamma=0.99, lmbda=0.95, value_coeff=1.0,
        logger=None, log_tag=""):
    num_envs = env.num_envs
    state, _ = env.reset()
    episode_rewards = []
    log_tag = log_tag.replace("/", "_")
    
    if logger is None:
        log_dir = f"runs/{log_tag}_{int(time.time())}_envs_{num_envs}_steps_{steps}_epochs_{epochs}_batch_{batch_size}_lr_{lr}_clip_{clip_eps}"
        logger = SummaryWriter(log_dir, max_queue=num_envs*steps)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    # Train policy
    for cur_iter in tqdm(range(iterations)):
        states = []
        actions = []
        action_probs = []
        rewards = []
        is_done = []
        values = []

        # Simulate
        with torch.no_grad():
            for t in range(steps):
                # Sample an action using current policy
                raw_actions, value = agent.get_policy_and_value(torch.Tensor(state).float().to(DEVICE))
                probs = torch.softmax(raw_actions, dim=1)
                action = torch.multinomial(probs, 1).view(-1)

                states.append(state)
                actions.append(action)
                action_probs.append(probs[torch.arange(num_envs), action])
                values.append(value)

                state, r, terminated, truncated, infos = env.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                rewards.append(r)
                is_done.append(done)

                # Log episode termination info
                for idx, is_episode_done in enumerate(done):
                    if is_episode_done:
                        episode_reward = infos["final_info"][idx]["final_info"]["episode"]["r"][0]
                        log_idx = num_envs * (cur_iter * steps + t) + idx
                        logger.add_scalar(f"{log_tag}/episode_reward", episode_reward, log_idx)
                        episode_rewards.append((log_idx, episode_reward))
                    

            # states.append(state) # Use the last state to estimate advantages
            values.append(agent.get_value(torch.Tensor(state).float().to(DEVICE)))

            # Estimate advantages using GAE method
            values = torch.hstack(values).t()
            is_not_done = 1.0 - torch.Tensor(np.stack(is_done)).float().to(DEVICE)
            rewards = torch.Tensor(np.stack(rewards)).float().to(DEVICE)
            deltas = rewards + gamma * values[1:,:] * is_not_done - values[:-1,:]
            adv = torch.zeros_like(rewards).to(DEVICE)
            adv[steps - 1] = deltas[steps - 1]
            for i in range(steps - 2, -1, -1):
                adv[i] = deltas[i] + gamma * lmbda * is_not_done[i,:] * adv[i + 1]

            # Prepare training data
            states = torch.Tensor(np.concatenate(states)).float().to(DEVICE)
            actions = torch.stack(actions).reshape(-1)
            action_probs = torch.stack(action_probs).reshape(-1)
            adv = adv.reshape(-1)
            target_values = values[:-1, :].reshape(-1) + adv
            dataset = StackDataset(states, actions, action_probs, adv, target_values)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimize policy and value network
        with torch.enable_grad():
            total_policy_loss = 0.0
            total_value_loss = 0.0
            n_update = 0

            for _ in range(epochs):
                for b_states, b_actions, b_action_probs, b_adv, b_target_values in dataloader:
                    new_action_probs, new_values = agent.get_policy_and_value(b_states)
                    new_action_probs = torch.softmax(new_action_probs, dim=1)[
                        torch.arange(b_states.shape[0]),
                        b_actions,
                    ]

                    ratios = new_action_probs.view(-1) / b_action_probs
                    clipped_ratios = torch.clip(ratios, 1 - clip_eps, 1 + clip_eps)

                    policy_loss = -torch.minimum(ratios * b_adv, clipped_ratios * b_adv).mean()
                    value_loss = 0.5 * F.mse_loss(new_values.view(-1), b_target_values)

                    loss = policy_loss + value_coeff * value_loss
                    # make_dot(loss, show_attrs=True, show_saved=True).render("graph1", format="png", cleanup=True)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_policy_loss += policy_loss.cpu().item()
                    total_value_loss += value_loss.cpu().item()
                    n_update += 1

            avg_policy_loss = total_policy_loss / n_update
            avg_value_loss = total_value_loss / n_update
            log_idx = (cur_iter + 1) * steps * num_envs
            logger.add_scalar(f"{log_tag}/avg_policy_loss", avg_policy_loss, log_idx)
            logger.add_scalar(f"{log_tag}/avg_value_loss", avg_value_loss, log_idx)

    logger.close()
    return episode_rewards

def run_ppo(game, seed=1,
            num_envs=8, frame_stack=4, frameskip=1, 
            total_steps=10000000, steps=128,
            epochs=3, n_minibatches=4, **kwargs):
    set_seed(seed)
    
    env = create_atari_env(
        game, seed=seed, num_envs=num_envs,
        frame_stack=frame_stack, render_mode="rgb_array", frameskip=frameskip)

    agent = Agent(env.single_action_space.n, frame_stack).to(DEVICE)

    episode_rewards = ppo(env, agent, iterations=total_steps//(steps*num_envs),
                          steps=steps, epochs=epochs,
                          batch_size=num_envs*steps//n_minibatches, log_tag=game, **kwargs)

    env.close()
    return episode_rewards

def search_hyperparams(game, lrs, clips, **kwargs):
    res = []
    for lr in lrs:
        for clip_eps in clips:
            print(f"lr: {lr} | clip_eps: {clip_eps}")
            rewards = run_ppo(game, lr=lr, clip_eps=clip_eps, **kwargs)
            avg_reward = np.mean([r[1] for r in rewards])
            last_100_avg_reward = np.mean([r[1] for r in rewards[-100:]])
            res.append({
                'lr': lr, 'clip_eps': clip_eps,
                'overall_avg_reward': avg_reward,
                'last_100_eps_avg_reward': last_100_avg_reward,
                'rewards': rewards,
            })
            print(f"Overall avg reward: {avg_reward} | Last 100 eps avg reward {last_100_avg_reward}")

    df = pd.DataFrame.from_dict(res)
    df.to_csv(f"{game}_ppo_hyperparams_results.csv", index=False)
    return df

if __name__ == "__main__":
    # run_ppo("PongNoFrameskip-v4", seed=1, lr=3e-04, clip_eps=0.2)
    search_hyperparams("BreakoutNoFrameskip-v4", lrs=[1e-04, 3e-04, 1e-03], clips=[0.1, 0.2, 0.3])
