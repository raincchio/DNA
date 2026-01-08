
import network as core
import torch
from reset import run_reset
import numpy as np

from buffer import PPOBuffer
from torch.optim import Adam
import wandb
import time

from torch.optim.lr_scheduler import LambdaLR


def learn(data, epoch_reuse, steps_per_epoch, mini_batch, clip_ratio, ac_optimizer, ac):
    obs, act, adv, logp_old, ret, old_v = data['obs'], data['act'], data['adv'], data['logp'], data['ret'], data['val']

    indices = np.arange(len(obs))

    for i in range(epoch_reuse):
        np.random.shuffle(indices)
        for start in range(0, steps_per_epoch, mini_batch):
            end = start + mini_batch
            indx_ = indices[start:end]
            obs_, act_, adv_, logp_old_, ret_, old_v_ = obs[indx_], act[indx_], adv[indx_], logp_old[indx_], ret[indx_], \
            old_v[indx_]

            ac_optimizer.zero_grad()

            logp = ac.logp(obs_, act_)
            ratio = torch.exp(logp - logp_old_)
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_
            pg_loss = -(torch.min(ratio * adv_, clip_adv)).mean()

            vpred = ac.v(obs_).squeeze()
            vpredclipped = old_v_ + torch.clamp(vpred - old_v_, -clip_ratio, clip_ratio)

            vf_losses1 = torch.square(vpred - ret_)
            vf_losses2 = torch.square(vpredclipped - ret_)
            vf_loss = .5 * (torch.max(vf_losses1, vf_losses2)).mean()

            # vf_loss = .5 *vf_losses1.mean()
            loss = pg_loss + vf_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            # print(ac.log_std)
            ac_optimizer.step()

    return pg_loss.item(), (vpred - ret_).abs().mean().item()



def mujoco_ppo(env, actor_critic=None, log_writer=None, args=None,wandb_content=None):

    steps_per_epoch = 1000

    epochs = 1000
    epoch_reuse = 10
    gamma = 0.99
    lam = 0.95
    max_ep_len = 1000

    mini_batch = 25
    clip_ratio = 0.2
    lr = 3e-4

    device=args.device

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space,hidden_sizes=args.hid)

    if device=='cuda':
        ac.to(device)

    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam, device=device)

    ac_optimizer = Adam(ac.parameters(), lr=lr)

    # Prepare for interaction with environment
    o, _ = env.reset(seed=args.seed)
    env.action_space.seed(seed=args.seed)
    ep_ret, ep_len = 0,0
    EpRet = 0
    lr_lambda = lambda step: 1 - step / epochs
    scheduler = LambdaLR(ac_optimizer, lr_lambda)
    # Main loop: collect experience in env and update/log each epoch
    wandb_dict = {}
    begin = time.time()
    for epoch in range(0,epochs):

        for t in range(steps_per_epoch):
            o = np.array(o)

            a, v, logp = ac.step(torch.FloatTensor(o).to(device))

            next_o, r, terminated, truncated, info = env.step(a)
            ep_ret += (info['reward_run']+info['reward_ctrl'])
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            # logger.store(VVals=v)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = terminated or timeout
            # terminal = terminated or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                # print(t)
                if timeout or epoch_ended:
                    o = np.array(o)
                    _, v, _ = ac.step(torch.FloatTensor(o).to(device))
                else:
                    v = 0
                buf.finish_path(v)
                EpRet = ep_ret
                o, _= env.reset()
                ep_ret, ep_len = 0, 0

        data = buf.get()

        loss_pi, loss_v = learn(data, epoch_reuse, steps_per_epoch, mini_batch, clip_ratio, ac_optimizer, ac)

        scheduler.step()

        sample_histroy = data['obs']

        layers_frac, means, vars, stable= run_reset(
            sample_histroy,
            model=ac,
            optimizer=ac_optimizer,
            args=args,
            epoch=epoch
        )
        elapse = (time.time() - begin)/60
        print('Epoch: {} EpRet: {:.2f} dormant count: {} sparse count: {}, time:{}'.format(epoch, EpRet, loss_pi, loss_v, elapse/(1+epoch)))

        log_content = [EpRet, loss_pi, loss_v, *layers_frac, *means, * vars, *stable]

        for key, value in zip(wandb_content, log_content):
            wandb_dict[key] = value

        wandb.log(wandb_dict)

        log_writer.write(",".join(map(str, log_content)) + '\n')
        log_writer.flush()
    # wandb.finish()
    o, _ = env.reset(seed=args.seed)
    env.action_space.seed(seed=args.seed)
    ep_ret, ep_len = 0,0
    EpRet = 0
    lr_lambda = lambda step: 1 - step / epochs
    scheduler = LambdaLR(ac_optimizer, lr_lambda)
    # Main loop: collect experience in env and update/log each epoch
    args.redo=False
    wandb_dict = {}
    begin = time.time()
    for epoch in range(0,epochs):

        for t in range(steps_per_epoch):
            o = np.array(o)

            a, v, logp = ac.step(torch.FloatTensor(o).to(device))

            next_o, r, terminated, truncated, info = env.step(a)
            ep_ret += (info['reward_run']+info['reward_ctrl'])
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            # logger.store(VVals=v)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = terminated or timeout
            # terminal = terminated or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                # print(t)
                if timeout or epoch_ended:
                    o = np.array(o)
                    _, v, _ = ac.step(torch.FloatTensor(o).to(device))
                else:
                    v = 0
                buf.finish_path(v)
                EpRet = ep_ret
                o, _= env.reset()
                ep_ret, ep_len = 0, 0

        data = buf.get()

        loss_pi, loss_v = learn(data, epoch_reuse, steps_per_epoch, mini_batch, clip_ratio, ac_optimizer, ac)

        scheduler.step()

        sample_histroy = data['obs']

        layers_frac, means, vars, stable= run_reset(
            sample_histroy,
            model=ac,
            optimizer=ac_optimizer,
            args=args,
            epoch=epoch
        )
        elapse = (time.time() - begin)/60
        print('Epoch: {} EpRet: {:.2f} dormant count: {} sparse count: {}, time:{}'.format(epoch, EpRet, loss_pi, loss_v, elapse/(1+epoch)))

        log_content = [EpRet, loss_pi, loss_v, *layers_frac, *means, * vars, *stable]

        for key, value in zip(wandb_content, log_content):
            wandb_dict[key] = value

        wandb.log(wandb_dict)

        log_writer.write(",".join(map(str, log_content)) + '\n')
        log_writer.flush()
    # wandb.finish()
