import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import wandb
import os
import random
import time
from copy import deepcopy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WandB:
    os.environ["WANDB_API_KEY"] = "721698f6f56dad25aaddab8144199e466d24202f"
    project_name = "ppocheetah"

set_seed(45)
os.environ["WANDB_API_KEY"] = "721698f6f56dad25aaddab8144199e466d24202f"
wandb.init(project=WandB.project_name)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        x = self.net(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        return self.net(state)

class PPO:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.old_actor = None
        
        # Modified hyperparameters
        self.clip_ratio = 0.2
        self.gamma = 0.99
        self.lam = 0.95
        self.batch_size = 64
        self.n_epochs = 10
        self.target_kl = 0.015  # Target KL divergence
        self.entropy_coef = 0.01  # Entropy coefficient
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.cumulative_samples = 0
        self.entropies = []
        
    def get_action(self, state):
        mean, std = self.actor(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().mean()  # Remove clamping to allow natural entropy values
        return action.detach().numpy()[0], log_prob.detach()[0], entropy.item()
    
    def update(self, states, actions, old_log_probs, advantages, returns):
        policy_losses = []
        value_losses = []
        kl_divs = []
        entropy_losses = []
        
        # Store current policy state before updates
        self.old_actor = deepcopy(self.actor)
        
        early_stop = False
        
        for epoch in range(self.n_epochs):
            if early_stop:
                break
                
            for idx in range(0, len(states), self.batch_size):
                batch_states = states[idx:idx + self.batch_size]
                batch_actions = actions[idx:idx + self.batch_size]
                batch_old_log_probs = old_log_probs[idx:idx + self.batch_size]
                batch_advantages = advantages[idx:idx + self.batch_size]
                batch_returns = returns[idx:idx + self.batch_size]
                
                # Get distributions
                old_mean, old_std = self.old_actor(batch_states)
                new_mean, new_std = self.actor(batch_states)
                
                old_dist = Normal(old_mean.detach(), old_std.detach())
                new_dist = Normal(new_mean, new_std)
                
                new_log_probs = new_dist.log_prob(batch_actions).sum(dim=-1)
                entropy = new_dist.entropy().mean()
                
                # Calculate KL divergence
                kl = torch.distributions.kl_divergence(old_dist, new_dist).mean().item()
                
                if kl > 1.5 * self.target_kl:
                    early_stop = True
                    break
                
                # Policy loss with entropy bonus
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # Value loss with clipping
                values = self.critic(batch_states).squeeze()
                value_loss = ((values - batch_returns) ** 2).mean()
                
                # Update networks with gradient clipping
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                kl_divs.append(kl)
                entropy_losses.append(entropy.item())
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'kl_div': np.mean(kl_divs),
            'entropy': np.mean(entropy_losses)
        }
    
    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    
    def train(self, max_episodes=1000, steps_per_episode=1000, seed=45, save_interval=50):
        os.makedirs('saved_models', exist_ok=True)
        training_start_time = time.time()
        
        for episode in range(max_episodes):
            state, _ = self.env.reset(seed=seed)
            episode_reward = 0
            episode_length = 0
            episode_entropies = []
            
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            
            for step in range(steps_per_episode):
                action, log_prob, entropy = self.get_action(state)
                value = self.critic(state).item()
                episode_entropies.append(entropy)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            self.cumulative_samples += episode_length
            advantages, returns = self.compute_gae(rewards, values, self.critic(state).item(), dones)
            
            states = torch.FloatTensor(np.array(states))
            actions = torch.FloatTensor(np.array(actions))
            old_log_probs = torch.FloatTensor(log_probs)
            
            train_info = self.update(states, actions, old_log_probs, advantages, returns)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.entropies.append(np.mean(episode_entropies))
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_entropy = np.mean(self.entropies[-10:])
                
                metrics = {
                    "reward": avg_reward,
                    "policy_loss": train_info['policy_loss'],
                    "value_loss": train_info['value_loss'],
                    "kl_div": train_info['kl_div'],
                    "entropy": avg_entropy,
                    "episode": episode, 
                    "cumulative_steps": self.cumulative_samples 
                }

                wandb.log(metrics, step=self.cumulative_samples)
                print(f"Episode {episode}, Steps: {self.cumulative_samples}, Average Reward: {avg_reward:.2f}, Entropy: {avg_entropy:.4f}")
            
            if episode > 0 and episode % save_interval == 0:
                self.save_model(episode)
        
        return self.episode_rewards

    def save_model(self, episode):
        """
        Save the actor and critic models
        
        Args:
            episode (int): Current episode number for filename
        """
        actor_path = f'saved_models/actor_ep{episode}.pth'
        critic_path = f'saved_models/critic_ep{episode}.pth'
        
        # Save actor model
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
        }, actor_path)
        
        # Save critic model
        torch.save({
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, critic_path)
        
        print(f"Models saved at episode {episode}")
        wandb.save(actor_path)
        wandb.save(critic_path)

   
    def load_model(self, actor_path, critic_path):
        """
        Load previously saved actor and critic models
        
        Args:
            actor_path (str): Path to saved actor model
            critic_path (str): Path to saved critic model
        """
        # Load actor model
        actor_checkpoint = torch.load(actor_path)
        self.actor.load_state_dict(actor_checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])
        
        # Load critic model
        critic_checkpoint = torch.load(critic_path)
        self.critic.load_state_dict(critic_checkpoint['model_state_dict'])
        self.critic_optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])
        
        print(f"Models loaded from {actor_path} and {critic_path}")
        
# Training
def main():
    wandb.init(
        project="ppocheetah",
        config={
            "algorithm": "PPO",
            "environment": "HalfCheetah-v4",
            "max_episodes": 1000,
            "steps_per_episode": 1000,
            "learning_rate_actor": 3e-4,
            "learning_rate_critic": 1e-3,
            "clip_ratio": 0.2,
            "gamma": 0.99,
            "lambda": 0.95,
            "entropy_coef": 0.01,
            "target_kl": 0.015
        }
    )
    
    env = gym.make('HalfCheetah-v4')
    set_seed(45)
    agent = PPO(env)
    rewards = agent.train(seed=45, max_episodes=1000, save_interval=50)
    wandb.finish()

if __name__ == "__main__":
    main()