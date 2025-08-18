"""
Data preprocessing for CatDiP-RL experiments.
Handles environment setup and data preparation.
"""

import time
import math
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flatten_grads(model: nn.Module) -> torch.Tensor:
    """Return a 1-D tensor containing the gradients of all parameters."""
    return torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])


class LargeVocabBandit:
    """Contextual K-armed bandit with 32 one-hot states and sparse optimal action."""

    def __init__(self, num_actions: int, seed: int = 7):
        set_seed(seed + num_actions)  # deterministic across |A|
        self.K = num_actions
        self.num_states = 32

        self.R = 0.1 * torch.ones(self.num_states, self.K)
        optimal_indices = torch.randint(low=0, high=self.K, size=(self.num_states,))
        self.R[torch.arange(self.num_states), optimal_indices] = 1.0

        self.state = None

    def reset(self) -> torch.Tensor:
        self.state = torch.randint(0, self.num_states, (1,)).item()
        return F.one_hot(torch.tensor(self.state), num_classes=self.num_states).float()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool, dict]:
        action = action.item() if torch.is_tensor(action) else int(action)
        reward_prob = self.R[self.state, action]
        reward = torch.bernoulli(reward_prob).item()
        done = True  # bandit → single-step episode
        info = {"opt": bool(reward_prob > 0.5)}
        next_state = self.reset()
        return next_state, torch.tensor(reward, dtype=torch.float32), done, info


class TokenMLP(nn.Module):
    """Shared policy network for discrete action spaces."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden = min(1024, int(4 * math.sqrt(output_dim)))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)  # logits


def prepare_environment(action_size: int, seed: int = 0) -> LargeVocabBandit:
    """Prepare the bandit environment for experiments."""
    set_seed(seed)
    return LargeVocabBandit(action_size, seed)


def check_gpu_memory():
    """Check GPU memory availability for T4 compatibility."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Total GPU memory: {total_memory / 1024**3:.1f} GB")
        return total_memory
    else:
        print("CUDA not available, using CPU")
        return 0
