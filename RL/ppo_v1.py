import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RolloutStorage:
    def __init__(self, num_steps, num_envs, obs_dim, device="cuda"):
        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, 2), device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.num_steps = num_steps
        self.step = 0
        self.device = device

    def insert(self, obs, actions, logprobs, rewards, dones, values):
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.logprobs[self.step] = logprobs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.step = (self.step + 1) % self.num_steps
        
    def reset(self):
        self.step = 0

class PPO:
    def __init__(
        self,
        agent,
        optimizer,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=4,
        mini_batch_size=32768,
        gamma=0.99,
        gae_lambda=0.95
    ):
        self.agent = agent
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def update(self, storage, next_value, next_done):
        # GAE Calculation
        with torch.no_grad():
            advantages = torch.zeros_like(storage.rewards)
            lastgaelam = 0
            for t in reversed(range(storage.num_steps)):
                if t == storage.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - storage.dones[t + 1]
                    nextvalues = storage.values[t + 1]
                
                delta = storage.rewards[t] + self.gamma * nextvalues * nextnonterminal - storage.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                
            returns = advantages + storage.values

        # Flatten
        b_obs = storage.obs.reshape(-1, storage.obs.shape[-1])
        b_logprobs = storage.logprobs.reshape(-1)
        b_actions = storage.actions.reshape(-1, 2)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = storage.values.reshape(-1)

        # Optimization
        b_inds = np.arange(b_obs.shape[0])
        total_loss = 0
        
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, b_obs.shape[0], self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    # approx_kl = ((ratio - 1) - logratio).mean()
                    # clipfracs += [((ratio - 1.0).abs() > CLIP_EPS).float().mean().item()]
                    pass
                    
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value Loss
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -self.clip_eps,
                    self.clip_eps,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - self.entropy_coef * entropy_loss + self.value_loss_coef * v_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                
        return total_loss / (self.ppo_epochs * (b_obs.shape[0] / self.mini_batch_size))
