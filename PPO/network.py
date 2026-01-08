import numpy as np
import scipy.signal
# import gymnasium

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
active_fn = F.relu


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    # n = len(x)
    # result = np.zeros_like(x)
    # result[-1] = x[-1]
    # for i in range(n - 2, -1, -1):
    #     result[i] = x[i] + discount * result[i + 1]
    # #
    # return result



class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, act_dim)

    def _distribution(self, obs):
        actro_ = active_fn(self.fc1(obs))
        actro_ = active_fn(self.fc2(actro_))
        logits = self.actor(actro_)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class ConvActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.actor = nn.Linear(512, act_dim)

    def _distribution(self, obs):
        v_ = active_fn(self.conv1(obs / 255.0))
        v_ = active_fn(self.conv2(v_))
        v_ = active_fn(self.conv3(v_))
        if len(v_.shape)==3:
            v_ = torch.flatten(v_, start_dim=0)
        else:
            v_ = torch.flatten(v_, start_dim=1)
        v_ = active_fn(self.fc1(v_))
        logits = self.actor(v_)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.actor = nn.Linear(hidden_sizes, act_dim)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))


    def _distribution(self, obs):

        actro_ = active_fn(self.fc1(obs))
        actro_ = active_fn(self.fc2(actro_))
        mu = self.actor(actro_)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        # self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

        self.fc1 = nn.Linear(obs_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.v = nn.Linear(hidden_sizes, 1)

    # def forward(self, obs):
    #     return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.
    def forward(self,obs):
        v_ = active_fn(self.fc1(obs))
        v_ = active_fn(self.fc2(v_))
        v = self.v(v_)

        return v

class ConvCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.v = nn.Linear(512, 1)

    def forward(self,obs):
        v_ = active_fn(self.conv1(obs/255.0))
        v_ = active_fn(self.conv2(v_))
        v_ = active_fn(self.conv3(v_))
        if len(v_.shape) == 3:
            v_ = torch.flatten(v_, start_dim=0)
        else:
            v_ = torch.flatten(v_, start_dim=1)
        v_ = active_fn(self.fc1(v_))
        v = self.v(v_)

        return v



class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=64, activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]


class ConvActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        self.pi = ConvActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = ConvCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]


def ortho_init(tensor, scale=1.0):
    shape = tensor.shape
    if len(shape) == 2:
        flat_shape = shape
    elif len(shape) == 4:  # Assumes NCHW format
        flat_shape = (shape[0] * shape[1] * shape[2], shape[3])
    else:
        raise NotImplementedError("Only supports dense and 4D conv layers")

    a = torch.randn(flat_shape, dtype=tensor.dtype, device=tensor.device)
    u, _, v = torch.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # Pick the one with the correct shape
    q = q.reshape(shape)
    with torch.no_grad():
        tensor.copy_(scale * q[:shape[0], :shape[1]])
