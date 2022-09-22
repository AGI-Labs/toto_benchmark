from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import itertools
from PIL import Image

def np_to_tensor(nparr, device):
    return torch.from_numpy(nparr).float()

def shift_window(arr, window, np_array=True):
    nparr = np.array(arr) if np_array else list(arr)
    nparr[:-window] = nparr[window:]
    nparr[-window:] = nparr[-1] if np_array else [nparr[-1]] * window
    return nparr

class FrankaDatasetTraj(Dataset):

    def __init__(self, data,
            logs_folder='./',
            subsample_period=1,
            im_h=480,
            im_w=640,
            obs_dim=7,
            action_dim=7,
            H=50, ##########
            device='cpu',
            cameras=None,
            img_transform_fn=None,
            noise=None):
        from multiprocessing import set_start_method
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        self.logs_folder = logs_folder
        self.subsample_period = subsample_period
        self.im_h = im_h
        self.im_w = im_w
        self.obs_dim = obs_dim
        self.action_dim = action_dim # not used yet
        self.H = H
        self.demos = data
        self.device = device
        self.cameras = cameras or []
        self.img_transform_fn = img_transform_fn
        self.noise = noise
        self.subsample_demos()
        if len(self.cameras) > 0:
            self.load_imgs()
        self.process_demos()

    def subsample_demos(self):
        for traj in self.demos:
            for key in ['cam0c', 'observations', 'actions', 'terminated', 'rewards']:
                if key == 'observations':
                    traj[key] = traj[key][:, :self.obs_dim] ##############
                traj[key] = traj[key][::self.subsample_period]

    def process_demos(self): # each datapoint is a single (st, at)!
        inputs, labels = [], []
        for traj in self.demos:
            if traj['actions'].shape[0] > self.H: # ignore short trajs
                for start in range(traj['actions'].shape[0] - self.H + 1):
                    inputs.append(traj['observations'][start])
                    labels.append(traj['actions'][start : start + self.H, :]) # sliding window
        inputs = np.stack(inputs, axis=0).astype(np.float64)
        labels = np.stack(labels, axis=0).astype(np.float64)
        if self.cameras:
            images = []
            for traj in self.demos:
                if traj['actions'].shape[0] > self.H: # ignore short trajs
                    for start in range(traj['actions'].shape[0] - self.H + 1):
                        images.append(traj['images'][start])
            # images = np.concatenate(images, axis=0) ### TODO: sanity check
            self.images = images
        self.inputs = np_to_tensor(inputs, self.device)
        self.labels = np_to_tensor(labels, self.device)

        self.labels = self.labels.reshape([self.labels.shape[0], -1]) # flatten actions to (#trajs, H * action_dim)


    def load_imgs(self):
        print("Start loading images...")
        cnt = 0
        for path in self.demos:
            print(cnt, "  ", path['traj_id'])
            path['images'] = []
            for i in range(path['observations'].shape[0]):
                img_path = os.path.join(self.logs_folder, os.path.join('data', path['traj_id'], path['cam0c'][i]))
                path['images'].append(img_path)
            cnt += 1
        print("Finished loading images...")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        datapoint = {
                'inputs': self.inputs[idx],
                'labels': self.labels[idx],  ######
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
                datapoint[_c] = (self.img_transform_fn(img) if self.img_transform_fn
                                 else datapoint[_c])
                img.close()
        return datapoint
