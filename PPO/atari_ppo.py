import network as core
import torch
from reset import run_reset
import numpy as np

from buffer import PPOBuffer
from torch.optim import Adam
import time


def atari_ppo(env, actor_critic=core.ConvActorCritic, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=3e-4, train_pi_iters=4, train_v_iters=4, lam=0.97, max_ep_len=1000,
        save_freq=10, log_writer=None, args=None, device='cpu'):

    steps_per_epoch = 1000
    epochs = 1000
    gamma = args.gamma

    # Random seed

    # Instantiate environment
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space)

    if device=='cuda':
        ac.to(device)

    # Count variables
    # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    # print('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam, device=device)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    # Todo


    def update(data):
        # pi_l_old, pi_info_old = compute_loss_pi(data)
        # pi_l_old = pi_l_old.item()
        # v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            # kl = pi_info['kl']

            #
            # if kl > 1.5 * target_kl:
            #     logger.log('Early stopping at step %d due to reaching max kl.' % i)
            #     break

            loss_pi.backward()
            # mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        # logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            # mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()
        return loss_pi.item(), loss_v.item()
        # Log changes from update
        # kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        # logger.store(LossPi=pi_l_old, LossV=v_l_old,
        #              KL=kl, Entropy=ent, ClipFrac=cf,
        #              DeltaLossPi=(loss_pi.item() - pi_l_old),
        #              DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        o, _ = env.reset(seed=args.seed)
        ep_ret, ep_len = 0, 0
        EpRet = 0

        for t in range(steps_per_epoch):
            o = np.array(o)
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device))

            next_o, r, terminated, truncated, info = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = terminated or timeout
            # terminal = terminated or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    o = np.array(o)
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:


                    EpRet = ep_ret
                    o, _= env.reset()
                    ep_ret, ep_len = 0, 0

        # Save model

        # Perform PPO update!
        data = buf.get()
        loss_pi, loss_v = update(data)

        dormant_fraction = run_redo(
            data['obs'],
            model=ac.v,
            optimizer=vf_optimizer,
            tau=0.025,
            args=args,
            epoch=epoch
        )

        print('Epoch: {} EpRet: {:.2f} dormant: {}%'.format(epoch, EpRet, dormant_fraction))
        log_content = [EpRet, dormant_fraction, loss_pi, loss_v]
        log_writer.write(",".join(map(str, log_content)) + '\n')
        log_writer.flush()
        # Log info about epoch
        # logger.log_tabular('Epoch', epoch)
        # logger.log_tabular('EpRet', with_min_and_max=True)
        # logger.log_tabular('EpLen', average_only=True)
        # logger.log_tabular('VVals', with_min_and_max=True)
        # logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        # logger.log_tabular('LossPi', average_only=True)
        # logger.log_tabular('LossV', average_only=True)
        # logger.log_tabular('DeltaLossPi', average_only=True)
        # logger.log_tabular('DeltaLossV', average_only=True)
        # logger.log_tabular('Entropy', average_only=True)
        # logger.log_tabular('KL', average_only=True)
        # logger.log_tabular('ClipFrac', average_only=True)
        # logger.log_tabular('StopIter', average_only=True)
        # logger.log_tabular('Time', time.time() - start_time)
        # logger.dump_tabular()
