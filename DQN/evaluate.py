import torch
import numpy as np
from copy import deepcopy
import time

def evaluate(global_step,
             eval_envs,
             eval_episodes,
             eval_state,
             eval_q_network,
             xlog=None,
             device='cuda',
             ):

    start_time = time.time()

    # device = torch.device('cuda')
    old_log = deepcopy(xlog.data_log)

    eval_q_network.load_state_dict(eval_state)

    obs,_ = eval_envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:

        q_values = eval_q_network(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs,_ ,_, _, infos = eval_envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    episodic_returns.append(info["episode"]["r"])

        obs = next_obs
    old_log['eval_reward'] = np.mean(episodic_returns)
    data = list(old_log.values())
    xlog.log_file(log_value=data)

    end_time = time.time()

    # 计算程序运行时间
    elapsed_time = end_time - start_time

    print(f"glb_stp: {global_step} , expl_rwd: {old_log['expl_reward']}, eval_rwd: {old_log['eval_reward']}, drmt_pct: {int(data[4])} % elapsed_time: {elapsed_time:.2f} s")

