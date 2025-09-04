# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from pathlib import Path

import ale_py

import gymnasium as gym
import numpy as np
import torch

import torch.optim as optim
import tyro

from DQN.agent import QNetwork, linear_schedule
from buffer import ReplayBuffer
from config import Config
from redo import run_redo
from utils import lecun_normal_initializer, make_env, set_cuda_configuration
from evaluate import evaluate

from xinglog.xinglog import XLogger

from agent import dqn_loss


def main(cfg: Config) -> None:

    # To get deterministic pytorch to work
    if cfg.torch_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    device = set_cuda_configuration(cfg.gpu)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.env_id, cfg.seed + i) for i in range(cfg.num_envs)]
    )
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.env_id, cfg.seed+100 + i) for i in range(cfg.num_envs)]
    )
    # envs = make_atari_env(env_id=cfg.env_id,num_envs=1,)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"


    exp_path = '/vepfs-dev/xing/workspace/DNA/experiments'
    res_filename = f'{cfg.env_id}-seed_{cfg.seed}'
    if cfg.enable_muon:
        algo_name = f'DQN_muon'
    elif cfg.enable_redo:
        algo_name = f'DQN_redo'
    else:
        algo_name = 'DQN'
    rec_variable_name = ['eval_reward','expl_reward','td_loss','q_values',f"dormant_tau_{cfg.redo_tau}_fraction",f"dormant_tau_{cfg.redo_tau}_count"]
    xlog = XLogger(exp_path=exp_path, algo_dir=algo_name, res_filename=res_filename, record_variable_names=rec_variable_name)

    q_network = QNetwork(envs).to(device)
    if cfg.use_lecun_init:
        # Use the same initialization scheme as jax/flax
        q_network.apply(lecun_normal_initializer)
    if cfg.enable_muon is False:
        optimizer = optim.Adam(q_network.parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)
    if cfg.enable_muon:
        from muon import MuonWithAuxAdam
        paramters = q_network.named_modules()
        muon_name =['conv2', 'conv3','fc1']
        muon_weight = []
        adam_param = []
        for name, param in paramters:
            adam_param.append(param.bias)
            if name in muon_name:
                muon_weight.append(param.weight)
            else:
                adam_param.append(param.weight)
        param_groups = [
            dict(params=muon_weight, use_muon=True,
                 lr=0.02, weight_decay=0),
            dict(params=adam_param, use_muon=False,
                 lr=cfg.learning_rate, betas=(0.9, 0.999),eps=cfg.adam_eps, weight_decay=0),
        ]
        optimizer = MuonWithAuxAdam(param_groups)


    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        cfg.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    for global_step in range(cfg.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(cfg.start_e, cfg.end_e, cfg.exploration_fraction * cfg.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            expl_rewards = []
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                expl_rewards.append(info["episode"]["r"])

            xlog.update('expl_reward', np.mean(expl_rewards) if len(expl_rewards) > 0 else 0)


        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            if global_step % cfg.train_frequency == 0:
                data = rb.sample(cfg.batch_size)
                loss, old_val = dqn_loss(
                    q_network=q_network,
                    target_network=target_network,
                    obs=data.observations,
                    next_obs=data.next_observations,
                    actions=data.actions,
                    rewards=data.rewards,
                    dones=data.dones,
                    gamma=cfg.gamma,
                )
                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                xlog.update('td_loss', loss)
                xlog.update('q_values', old_val.mean().item())

            if global_step % cfg.redo_check_interval == 0:
                redo_samples = rb.sample(cfg.redo_bs)
                redo_out = run_redo(
                    redo_samples.observations,
                    model=q_network,
                    optimizer=optimizer,
                    tau=cfg.redo_tau,
                    re_initialize=cfg.enable_redo,
                    use_lecun_init=cfg.use_lecun_init,
                )

                xlog.update(f"dormant_tau_{cfg.redo_tau}_fraction", redo_out["dormant_fraction"])
                xlog.update(f"dormant_tau_{cfg.redo_tau}_count", redo_out["dormant_count"])

            # update target network
            if global_step % cfg.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        cfg.tau * q_network_param.data + (1.0 - cfg.tau) * target_network_param.data
                    )

    if global_step %1000==0:
        eval_reward = evaluate(
            envs=eval_envs,
            eval_episodes=5,
            model=QNetwork,
            device=cfg.device,
        )
        xlog.update('eval_reward', eval_reward)
        xlog.log()
        QNetwork.train()

    if cfg.save_model:
        model_path = Path(f"runs/{run_name}/{cfg.exp_name}")
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(q_network.state_dict(), model_path / ".cleanrl_model")
        print(f"model saved to {model_path}")

    envs.close()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
