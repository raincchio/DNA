
import torch
import numpy as np


def evaluate(
        envs,
        eval_episodes,
        model,
        device,
):

    model.eval()
    obs = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:

        q_values = model(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, infos = envs.step(actions)
        for info in infos:
            if "episode" in info.keys():
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return np.mean(episodic_returns)
