import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticAgent(nn.Module):
    def __init__(self, obs_dim, action_dim_rudder=3, action_dim_fire=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.actor_rudder = nn.Linear(128, action_dim_rudder)
        self.actor_fire = nn.Linear(128, action_dim_fire)
        self.critic = nn.Linear(128, 1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits_rudder = self.actor_rudder(hidden)
        logits_fire = self.actor_fire(hidden)
        
        probs_rudder = Categorical(logits=logits_rudder)
        probs_fire = Categorical(logits=logits_fire)
        
        if action is None:
            action_rudder = probs_rudder.sample()
            action_fire = probs_fire.sample()
        else:
            action_rudder = action[:, 0].long()
            action_fire = action[:, 1].long()
            
        log_prob_rudder = probs_rudder.log_prob(action_rudder)
        log_prob_fire = probs_fire.log_prob(action_fire)
        
        return torch.stack([action_rudder, action_fire], dim=1), log_prob_rudder + log_prob_fire, probs_rudder.entropy() + probs_fire.entropy(), self.critic(hidden)
