import numpy as np
import scipy.signal
# import gymnasium

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
active_fn = F.relu



class shareMLPActorCriticLR(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=64, activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.norm1 = nn.LayerNorm(hidden_sizes)
        self.norm2 = nn.LayerNorm(hidden_sizes)
        self.norm3 = nn.LayerNorm(hidden_sizes)
        self.norm4 = nn.LayerNorm(hidden_sizes)

        self.fc1 = nn.Linear(obs_dim, hidden_sizes)

        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc4 = nn.Linear(hidden_sizes, hidden_sizes)

        self.actor = nn.Linear(hidden_sizes, act_dim)


        self.value = nn.Linear(hidden_sizes, 1)

        # log_std = -1*torch.ones(act_dim)
        log_std = torch.zeros(act_dim)
        self.log_std = torch.nn.Parameter(log_std)


    def share(self, obs):
        v_1 = active_fn(self.norm1(self.fc1(obs)))
        v_2 = active_fn(self.norm2(self.fc2(v_1))) +v_1
        v_3 = active_fn(self.norm3(self.fc3(v_2))) +v_2
        share = active_fn(self.norm4(self.fc4(v_3)))+v_3

        return share

    def v(self, obs):
        share = self.share(obs)
        v = self.value(share)
        return v

    def pi(self,obs):
        share = self.share(obs)
        mu = self.actor(share)
        # log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def logp(self,obs, act):
        pi = self.pi(obs)
        logp_a = pi.log_prob(act).sum(axis=-1)

        return logp_a

    def step(self, obs):
        with torch.no_grad():

            pi = self.pi(obs)
            a = pi.sample()
            logp_a = pi.log_prob(a).sum(axis=-1)
            v = self.v(obs)

        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]


class shareMLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=64, activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.fc1 = nn.Linear(obs_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc4 = nn.Linear(hidden_sizes, hidden_sizes)

        self.actor = nn.Linear(hidden_sizes, act_dim)

        self.value = nn.Linear(hidden_sizes, 1)

        # log_std = -1*torch.ones(act_dim)
        log_std = torch.zeros(act_dim)
        self.log_std = torch.nn.Parameter(log_std)

    def share(self, obs):
        v_1 = active_fn(self.fc1(obs))
        v_2 = active_fn(self.fc2(v_1))
        v_3 = active_fn(self.fc3(v_2))
        share = active_fn(self.fc4(v_3))

        return share

    def v(self, obs):
        share = self.share(obs)
        v = self.value(share)
        return v

    def pi(self, obs):
        share = self.share(obs)
        mu = self.actor(share)
        # log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def logp(self, obs, act):
        pi = self.pi(obs)
        logp_a = pi.log_prob(act).sum(axis=-1)

        return logp_a

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi(obs)
            a = pi.sample()
            logp_a = pi.log_prob(a).sum(axis=-1)
            v = self.v(obs)

        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden):
        super(FeedForward, self).__init__()
        self.redo1 = nn.Linear(d_model, d_hidden, bias=False)  # 升维
        # self.act = nn.GeLU()  # 非线性激活（可以换成 GELU）
        self.w2 = nn.Linear(d_hidden, d_model,bias=False)  # 降维
        self.w3 = nn.Linear(d_model, d_hidden,bias=False) # 随机失活，防止过拟合

    def forward(self, x):
        return self.w2(F.relu(self.redo1(x)) * self.w3(x))


class shareMLPActorCriticFFN(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=64):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.redo1 = nn.Linear(obs_dim, 64)
        self.ffn1 = FeedForward(64, 256)
        self.redo1 = nn.Linear(64, 64)
        self.ffn2 = FeedForward(64, 256)

        self.actor = nn.Linear(64, act_dim)

        self.value = nn.Linear(64, 1)

        # log_std = -1*torch.ones(act_dim)
        log_std = torch.zeros(act_dim)
        self.log_std = torch.nn.Parameter(log_std)

    def share(self, obs):
        v_1 = active_fn(self.redo1(obs))
        v_2 = self.ffn1(v_1) +v_1
        v_3 = active_fn(self.fc3(v_2))
        share = active_fn(self.fc4(v_3))

        return share

    def v(self, obs):
        share = self.share(obs)
        v = self.value(share)
        return v

    def pi(self, obs):
        share = self.share(obs)
        mu = self.actor(share)
        # log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def logp(self, obs, act):
        pi = self.pi(obs)
        logp_a = pi.log_prob(act).sum(axis=-1)

        return logp_a

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi(obs)
            a = pi.sample()
            logp_a = pi.log_prob(a).sum(axis=-1)
            v = self.v(obs)

        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]