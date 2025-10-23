import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Basic nature DQN agent."""

    def __init__(self, action_space, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, bias=bias)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, bias=bias)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, bias=bias)
        self.fc1 = nn.Linear(3136, 512, bias=bias)
        self.q = nn.Linear(512, action_space, bias=bias)

    def forward(self, x):
        x = F.relu(self.conv1(x / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.q(x)
        return x


def linear_schedule(start_e: float, end_e: float, duration: float, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def dqn_loss(
        q_network: QNetwork,
        target_network: QNetwork,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the double DQN loss."""
    with torch.no_grad():
        # Get value estimates from the target network
        target_vals = target_network.forward(next_obs)
        # Select actions through the policy network
        policy_actions = q_network(next_obs).argmax(dim=1)
        target_max = target_vals[range(len(target_vals)), policy_actions]
        # Calculate Q-target
        td_target = rewards.flatten() + gamma * target_max * (1 - dones.flatten())

    old_val = q_network(obs).gather(1, actions).squeeze()
    return F.mse_loss(td_target, old_val), old_val