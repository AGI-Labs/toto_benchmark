import os
import random
from multiprocessing import set_start_method

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from data_with_embeddings import precompute_embeddings
from toto_benchmark.vision import load_transforms


def shift_window(arr, window, np_array=True):
    nparr = np.array(arr) if np_array else list(arr)
    nparr[:-window] = nparr[window:]
    nparr[-window:] = nparr[-1] if np_array else [nparr[-1]] * window
    return nparr

class FrankaDatasetTraj(Dataset):
    def __init__(self, data, cfg, sim=True):
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        self.demos = data
        self.cfg = cfg
        self.logs_folder = cfg.data.logs_folder
        self.subsample_period = cfg.data.subsample_period
        self.im_h = cfg.data.images.im_h
        self.im_w = cfg.data.images.im_w
        self.obs_dim = cfg.data.in_dim
        self.H = cfg.data.H
        self.top_k = cfg.data.top_k
        self.device = cfg.training.device
        self.cameras = cfg.data.images.cameras or []
        self.img_transform_fn = load_transforms(cfg)
        self.noise = cfg.data.noise
        self.crop_images = cfg.data.images.crop
        self.pick_high_reward_trajs()

        self.subsample_demos()

        if sim:
            self.demos = precompute_embeddings(self.cfg, self.demos, from_files=False)
        elif len(self.cameras) > 0:
            self.load_imgs()

        self.process_demos()

    def pick_high_reward_trajs(self):
        original_data_size = len(self.demos)
        if self.top_k is None: # assumed using all successful traj (reward > 0)
            self.demos = [traj for traj in self.demos if traj['rewards'][-1] > 0]
            print(f"Using {len(self.demos)} number of successful trajs. Total trajs: {original_data_size}")
        elif self.top_k == 1: # using all data
            pass
        else:
            self.demos = sorted(self.demos, key=lambda x: x['rewards'][-1], reverse=True)
            top_idx_thres = int(self.top_k * len(self.demos))
            print(f"picking top {self.top_k * 100}% of trajs: {top_idx_thres} from {original_data_size}")
            self.demos = self.demos[:top_idx_thres]
        random.shuffle(self.demos)


    def subsample_demos(self):
        for traj in self.demos:
            for key in traj.keys():
                if key == 'observations':
                    traj[key] = traj[key][:, :self.obs_dim]
                if key == 'rewards':
                    rew = traj[key][-1]
                    traj[key] = traj[key][::self.subsample_period]
                    traj[key][-1] = rew
                else:
                    if key not in ('traj_id', 'material', 'normalized_reward'):
                        traj[key] = traj[key][::self.subsample_period]

    def process_demos(self):
        inputs, labels = [], []
        for traj in self.demos:
            traj['observations'] = np.hstack([traj['observations'], traj['embeddings']])

            if traj['actions'].shape[0] > self.H:
                for start in range(traj['actions'].shape[0] - self.H + 1):
                    inputs.append(traj['observations'][start])
                    labels.append(traj['actions'][start : start + self.H, :])
            else:
                extended_actions = np.vstack([traj['actions'], np.tile(traj['actions'][-1], [self.H - traj['actions'].shape[0], 1])]) # pad short trajs with the last action
                inputs.append(traj['observations'][0])
                labels.append(extended_actions)

        inputs = np.stack(inputs, axis=0)
        labels = np.stack(labels, axis=0)
        if self.cameras:
            images = []
            for traj in self.demos:
                if traj['actions'].shape[0] > self.H:
                    for start in range(traj['actions'].shape[0] - self.H + 1):
                        images.append(traj['images'][start])
                else:
                    images.append(traj['images'][0])
            self.images = images

        self.inputs = torch.from_numpy(inputs).float()
        self.labels = torch.from_numpy(labels).float()
        self.labels = self.labels.reshape([self.labels.shape[0], -1]) # flatten actions to (#trajs, H * action_dim)

    def load_imgs(self):
        print("Start loading images...")
        for path in self.demos:
            path['images'] = []
            for i in range(path['observations'].shape[0]):
                img_path = os.path.join(self.logs_folder, os.path.join('data', path['traj_id'], path['cam0c'][i]))
                path['images'].append(img_path)
        print("Finished loading images.")


    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        datapoint = {
                'inputs': self.inputs[idx],
                'labels': self.labels[idx],
                }
        if self.noise:
            datapoint['inputs'] += torch.randn_like(datapoint['inputs']) * self.noise

        if self.cameras:
            for _c in self.cameras:
                try:
                    img = Image.open(self.images[idx])
                except:
                    print("\n***Image path does not exist. Set the image directory as logs_folder in the config.")
                    raise
                datapoint[_c] = img.crop((200, 0, 500, 400)) if self.crop_images else img
                datapoint[_c] = (self.img_transform_fn(datapoint[_c]) if self.img_transform_fn
                                 else datapoint[_c])
                img.close()
        return datapoint
