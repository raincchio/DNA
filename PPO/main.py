
import torch

import share_network as core
import numpy as np
import random
import os
from config import Atari, Mujoco, wandb_mode
from atari_ppo import atari_ppo
from mujoco_ppo import mujoco_ppo
from env_maker import make_atari_env, make_mujoco_env

import wandb


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='DemonAttack-v4')
    parser.add_argument('--env', type=str, default='HalfCheetah-v4')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--deeper', action='store_true')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=4)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--resp', action='store_true')
    parser.add_argument('--reset_interval', type=int, default=10)
    parser.add_argument('--reset_tau', type=float, default=0.0025, help='mean is 0.015625 for 64 neurons')
    parser.add_argument('--svd', action='store_true')



    args = parser.parse_args()

    wandb.init(project="redo_ppo", name=args.env+'_'+'seed_'+str(args.seed), group=f"tau_{args.reset_tau}",mode=wandb_mode)
    wandb.config.update(
        {
            'hid':args.hid,
            'env':args.env,
            'layer_num':args.l,
            'seed':args.seed,
            'redo':args.resp,
            'tau':args.reset_tau
        }
    )

    if args.task:
        data_dir = os.path.expanduser('~') + '/experiments/' + 'PPO_' + args.task +'_'
    else:
        data_dir = os.path.expanduser('~') + '/experiments/' + 'PPO_'

    if args.deeper:
        data_dir += 'deeper_'

    if args.redo:
        data_dir += 'REDO_F_'+str(args.reset_interval)+'_tau_'+str(args.reset_tau)+'_H_'+str(args.hid)
    elif args.resp:
        data_dir += 'RESP_F_'+str(args.reset_interval)+'_tau_'+str(args.reset_tau)+'_H_'+str(args.hid)
    else:
        data_dir += 'H_'+str(args.hid)


    file_name = args.env+'_'+'seed_'+str(args.seed)
    os.makedirs(data_dir,exist_ok=True)

    log_writer = open(data_dir + '/{}'.format(file_name), 'w')
    log_content = ['eval_reward','policy_loss', 'value_loss']
    hid_l = args.l
    log_content.extend([f'layer{i}_frac' for i in range(1, hid_l+1)])
    log_content.extend([f'layer{i}_mean' for i in range(1,hid_l+1)])
    log_content.extend([f'layer{i}_var' for i in range(1, hid_l+1)])
    log_content.extend([f'layer{i}_srank' for i in range(1, hid_l+1)])

    log_writer.write(",".join(log_content) + '\n')


    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.env in Mujoco:
        env = make_mujoco_env(args.env)
        if args.deeper:
            mujoco_ppo(env, actor_critic=core.shareMLPActorCriticLR, log_writer=log_writer, args=args,
                       wandb_content=log_content)
        else:
            mujoco_ppo(env, actor_critic=core.shareMLPActorCritic,log_writer=log_writer, args=args, wandb_content=log_content)
    elif args.env in Atari:
        env = make_atari_env(env_id=args.env)
        atari_ppo(env, actor_critic=core.ConvActorCritic, log_writer=log_writer, args=args)
    else:
        print('Not implement')