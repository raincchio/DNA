
import torch
import numpy as np


def evaluate(
        envs,
        eval_episodes,
        model,
        device,
):

    # model.eval()
    obs,_ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:

        q_values = model(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs,_ ,_, _, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" in info.keys():
                    episodic_returns.append(info["episode"]["r"])
        obs = next_obs
    return np.mean(episodic_returns)
