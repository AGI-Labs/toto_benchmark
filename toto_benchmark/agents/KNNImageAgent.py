import numpy as np
import os
import pickle
from scipy import special
from sklearn.neighbors import KDTree
import torch
from torchvision import transforms as T
from toto_benchmark.vision import load_model, load_transforms

class KNNImageAgent(object):
    def __init__(self, k, pickle_fn, vision_model, H, feature_key, device='cuda:0'):
        with open(pickle_fn, 'rb') as f:
            paths = pickle.load(f)
        
        self.key = feature_key
        self.H = H

        # for open loop KNN
        self.traj_id = None 
        self.action_idx = 0 
        self.start_action_idx = None
        
        actions = []
        representations = []
        img_paths = []
        traj_ids = []
        count = 0
        print("Process KNN data...")
        for path in paths:
            if path['rewards'][-1] > 0: # ignore failed trajs
                actions.append(path['actions'])
                representations.append(path[self.key]) 
                img_paths.extend([os.path.join(path['traj_id'], fname) for fname in path['cam0c']])
                traj_ids.extend([(count, i) for i in range(len(path['cam0c']))])
                count += 1

        self.representations = np.vstack(representations)
        self.actions_raw = actions
        self.actions = np.vstack(actions)
        self.img_paths = img_paths
        self.traj_ids = traj_ids
        self.KDTree = KDTree(data=self.representations)
        self.vision_model = vision_model
        self.k = k
        print("KNN initialized.")

    def predict(self, sample):
        state = self.vision_model(torch.unsqueeze(sample['cam0c'], dim=0)).detach().numpy()
        state = state.reshape([1, -1])
        if self.key == 'observations': # jointstates + embeddings
            state = np.hstack([sample['inputs'].reshape([1, -1]), state]) 
        if self.H == 1:
            knn_dis, knn_idx = self.KDTree.query(state, k=self.k)
            print("Closest image:",self.img_paths[knn_idx[0][0]])
            actions = [self.actions[i] for i in knn_idx[0]]
            weights = [-1*(knn_dis[0][i]) for i in range(self.k)]
            weights = special.softmax(weights)
            return_action = np.zeros(7)
            for i in range(self.k):
                return_action += weights[i] * actions[i]
        else: 
            if self.traj_id == None or self.action_idx >= self.H:
                knn_dis, knn_idx = self.KDTree.query(state, k=self.k)
                self.traj_id, self.start_action_idx = self.traj_ids[knn_idx[0][0]]
                self.action_idx = 0
                print("Closest image:",self.img_paths[knn_idx[0][0]])
                print("traj number in dataset: ", self.traj_id, self.action_idx)
                
            if self.action_idx + self.start_action_idx < self.actions_raw[self.traj_id].shape[0]:
                return_action = self.actions_raw[self.traj_id][self.action_idx + self.start_action_idx]
                self.action_idx += 1
            else:
                print("Finished KNN pred. ")
                return_action = self.actions_raw[self.traj_id][self.action_idx + self.start_action_idx - 1] # the last valid action

        return return_action

def _init_agent_from_config(config, device='cpu'):
    print("Loading KNNImageAgent with",config.data.pickle_fn)
    vision_model = load_model(config)
    transforms = load_transforms(config)
    knn_agent = KNNImageAgent(config.knn.k, config.data.pickle_fn, vision_model, config.agent.H, config.agent.feature_key)
    return knn_agent, transforms