from datetime import datetime
import os
import numpy as np
from toto_benchmark.sim.dm_pour import DMWaterPouringEnv
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

import torch
import yaml

import toto_benchmark
from toto_benchmark.agents import init_agent_from_config
from toto_benchmark.vision import load_model, load_transforms
from toto_benchmark.scripts.utils import Namespace
from toto_benchmark.scripts.test_stub_env import get_args


def save_frames_as_gif(frames, frame_rate_divider=1):
    fname = datetime.now().strftime('%m-%d-%Y-%H-%M-%S.gif')
    save_path = os.path.join(os.path.dirname(toto_benchmark.__file__), "sim", fname)

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(save_path, writer='imagemagick', fps=60 / frame_rate_divider)
    print("Saved gif to", save_path)


def eval_agent(agent_predict_fn):
    n_rollouts = 100
    env = DMWaterPouringEnv(has_viewer=False)
    rewards = []
    env.seed(0)

    for i in range(n_rollouts):
        print(f'Evaluating Traj {i} ...')
        obs = env.reset()

        frames = []
        gif_frame_rate_divider = 15
        while not env.done:
            a = agent_predict_fn(obs)
            obs, reward, _, _ = env.step(a)

            # In first eval rollout, same frames for gif
            if i == 0 and env.timestep % gif_frame_rate_divider == 0:
                frames.append(obs['image'])

        if i == 0:
            save_frames_as_gif(frames, frame_rate_divider=gif_frame_rate_divider)

        rewards.append(reward)
        print(f'Traj {i} Final reward: {reward}')

    print(f'Finished evaluating {n_rollouts} trajectories.')

    mean_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    success_rate = sum(np.array(rewards) > 0) / len(rewards)
    print(f'Mean reward: {mean_reward}')
    print(f'Max reward: {max_reward}')
    print(f'Success rate: {success_rate}')


def create_agent_predict_fn(agent, cfg):
    img_transform_fn = load_transforms(cfg)
    model = load_model(cfg)
    model = model.eval().to(cfg.training.device)

    def agent_predict_fn(obs):
        image = torch.stack((img_transform_fn(Image.fromarray(obs['image'])),))
        embed = model(image.to(cfg.training.device)).to('cpu').data.numpy()

        o = torch.from_numpy(obs['proprioception'])[None].float()
        obs = np.hstack([o, embed])

        return agent.predict({'inputs': obs})
    return agent_predict_fn


def load_agent_from_args():
    args = get_args()

    with open(os.path.join(args.agent_path, 'hydra.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Overwriting to use local
    cfg['saved_folder'] = args.agent_path
    cfg = Namespace(cfg)

    agent, img_transform_fn = init_agent_from_config(cfg, cfg.training.device)

    agent_predict_fn = create_agent_predict_fn(agent, cfg)
    return agent_predict_fn

if __name__ == "__main__":
    # load_agent_from_args() is an example that loads a BC agent from train.py
    # TODO(optional): replace agent_predict_fn with your custom agent predict function
    agent_predict_fn = load_agent_from_args()

    eval_agent(agent_predict_fn)
