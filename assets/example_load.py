import pickle
import numpy as np
from PIL import Image
import os

paths = pickle.load(open('parsed.pkl', 'rb'))
s = np.concatenate([p['observations'][:-1] for p in paths])
a = np.concatenate([p['actions'][:-1] for p in paths])
sp = np.concatenate([p['observations'][1:] for p in paths])
r = np.concatenate([p['rewards'][:-1] for p in paths])
rollout_score = np.mean([np.sum(p['rewards']) for p in paths])  ### avg of sum of rewards (recorded) of a traj in the expert demos
num_samples = np.sum([p['rewards'].shape[0] for p in paths])

for path in paths:
    path['images'] = []
    path['depth_images'] = []
    for i in range(path['observations'].shape[0]):
        img = Image.open(os.path.join('data', path['traj_id'], path['cam0c'][i]))  # Assuming RGB for now
        path['images'].append(np.asarray(img))
        img.close()
        depth_img = Image.open(os.path.join('data', path['traj_id'], path['cam0d'][i])) 
        path['depth_images'].append(np.asarray(depth_img))
        depth_img.close()
    path['images'] = np.array(path['images']) # (horizon, img_h, img_w, 3)
    path['depth_images'] = np.array(path['depth_images']) # (horizon, img_h, img_w)
import pdb; pdb.set_trace()
