"""
Training module for CatDiP-RL experiments.
Contains agent implementations and training loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from preprocess import TokenMLP, flatten_grads


class AgentBase:
    """Base class for RL agents."""
    
    def __init__(self, obs_dim: int, act_dim: int, lr=5e-4):
        self.policy = TokenMLP(obs_dim, act_dim)
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=lr, betas=(0.9, 0.999))
        self.act_dim = act_dim

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()


class REINFORCEAgent(AgentBase):
    """Vanilla REINFORCE with a learned baseline (value function)."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__(obs_dim, act_dim)
        self.v = TokenMLP(obs_dim, 1)
        self.opt = torch.optim.AdamW(list(self.policy.parameters()) + list(self.v.parameters()), lr=5e-4)

    def compute_loss(self, obs, action, reward):
        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        value = self.v(obs).squeeze(-1)
        advantage = reward - value.detach()
        reinforce = -(log_prob * advantage).mean()
        value_loss = 0.5 * (reward - value).pow(2).mean()
        return reinforce + value_loss


class CatDiPSurrogateAgent(AgentBase):
    """Simplified surrogate of CatDiP – uses Gumbel-Softmax + score-matching-style loss.
    This is NOT the full CatDiP; it is just enough to produce low-variance
    gradients for the bandit demo.
    """

    def compute_loss(self, obs, action, reward):
        logits = self.policy(obs)
        tau = 0.7  # fixed temperature
        gumbel_noise = (-torch.empty_like(logits).exponential_().log())
        y = ((logits + gumbel_noise) / tau).softmax(-1)  # relaxed one-hot

        target = F.one_hot(action, num_classes=self.act_dim).float() * reward.unsqueeze(-1)
        loss = 0.5 * (y - target).pow(2).sum(-1).mean()  # score-matching-style L2
        return loss


def train_agents(env, agents: dict, steps: int = 1000, batch_size: int = 256):
    """Train multiple agents and collect gradient statistics."""
    grad_buffers = {name: [] for name in agents.keys()}
    regret_cum = {name: 0.0 for name in agents.keys()}

    obs = env.reset()
    for t in range(steps):
        for name, agent in agents.items():
            obs_batch, act_batch, rew_batch = [], [], []
            for _ in range(batch_size):
                s = env.reset()
                a = agent.act(s)
                _, r, *_ = env.step(a)
                obs_batch.append(s)
                act_batch.append(a)
                rew_batch.append(r)
                regret_cum[name] += (1.0 - r.item())  # optimal reward =1
            obs_b = torch.stack(obs_batch)
            act_b = torch.stack(act_batch)
            rew_b = torch.stack(rew_batch)

            agent.opt.zero_grad()
            loss = agent.compute_loss(obs_b, act_b, rew_b)
            loss.backward()
            grad_vec = flatten_grads(agent.policy).cpu()
            grad_buffers[name].append(grad_vec)
            agent.opt.step()

    results = {"variance": {}, "regret": {}}
    for name in agents.keys():
        grads = torch.stack(grad_buffers[name])
        var = torch.var(grads, dim=0).mean().item()
        results["variance"][name] = var
        regret = regret_cum[name] / (steps * batch_size)
        results["regret"][name] = regret
        print(f"{name:<10}  variance={var:.4e} | average regret={regret:.4f}")

    return results
