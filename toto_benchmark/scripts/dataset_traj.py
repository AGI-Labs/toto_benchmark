import os
import tqdm
import random
import numpy as np
import torch
from torch.utils.data import Dataset
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
            H=50,
            top_k=None,
            device='cpu',
            cameras=None,
            img_transform_fn=None,
            noise=None,
            crop_images=False,
            sim=True):
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
        self.action_dim = action_dim # not used
        self.H = H
        self.top_k = top_k
        self.demos = data
        self.device = device
        self.cameras = cameras or []
        self.img_transform_fn = img_transform_fn
        self.noise = noise
        self.crop_images = crop_images
        self.pick_high_reward_trajs()

        self.subsample_demos()
        if sim:
            self.embed_sim_images()
        elif len(self.cameras) > 0:
            self.load_imgs()
        self.process_demos()
        #self.process_simulation_demos()

    def pick_high_reward_trajs(self):
        original_data_size = len(self.demos)
        if self.top_k == None: # assumed using all successful traj (reward > 0)
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

    def embed_sim_images(self):
        #from toto_benchmark.vision import load_model, load_transforms
        from toto_benchmark.vision.pvr_model_loading import load_pvr_model, load_pvr_transforms
        vision_model_name = 'moco_conv5_robocloud'
        model = load_pvr_model(vision_model_name)[0]
        device = 'cuda:0'
        model = model.eval().to(device) ## assume this model is used in eval
        transforms = load_pvr_transforms(vision_model_name)[1]

        with torch.no_grad():
            for traj in self.demos:
                print("Starting traj of ",len(self.demos))
                #traj['observations'] = []
                path_len = len(traj['images'])
                batch_size = 128
                embeddings = []
                for b in range((path_len // batch_size + 1)):
                    if b * batch_size < path_len:
                        image_batch = traj['images'][b * batch_size:min(batch_size * (b + 1), path_len)]
                        image_batch = [transforms(Image.fromarray(img).crop((200, 0, 500, 400))) for img in image_batch]
                                                
                        chunk = torch.stack(image_batch)
                        chunk_embed = model(chunk.to(device))
                        embeddings.append(chunk_embed.to('cpu').data.numpy())
                    print("Finished batch",b,"of",path_len//batch_size)
                traj['embeddings'] = np.vstack(embeddings)
                traj['observations'] = np.hstack([traj['proprioception'], traj['embeddings']])
                
                #for img in traj['images']:
                #    img = (Image.fromarray(img).crop((200, 0, 500, 400)) if self.crop_images else img)
                #    img = transforms(img).to(device)
                #    img = img[None, :]
                #    traj['observations'].append(img_encoder(img))
                #    if len(traj['observations']) > 100:
                #        break
                    #imgs_out = [ for image in traj['images']]
                    #concat_inputs = torch.cat([sample['inputs']] + imgs_out, dim=-1)
                    #return self.models['decoder'](concat_inputs)
                #traj['observations'] = np.asarray(traj['observations'])

    def subsample_demos(self):
        for traj in self.demos:
            #for key in ['cam0c', 'observations', 'actions', 'terminated', 'rewards']:
            for key in traj.keys():
                if key == 'observations':
                    traj[key] = traj[key][:, :self.obs_dim]
                if key == 'rewards':
                    rew = traj[key][-1]
                    traj[key] = traj[key][::self.subsample_period]
                    traj[key][-1] = rew
                else:
                    traj[key] = traj[key][::self.subsample_period]

    def process_demos(self):
        inputs, labels = [], []
        cnt = 0
        for traj in self.demos:
            if traj['actions'].shape[0] > self.H:
                for start in range(traj['actions'].shape[0] - self.H + 1):
                    inputs.append(traj['observations'][start])
                    labels.append(traj['actions'][start : start + self.H, :]) 
            else:
                extended_actions = np.vstack([traj['actions'], np.tile(traj['actions'][-1], [self.H - traj['actions'].shape[0], 1])]) # pad short trajs with the last action
                inputs.append(traj['observations'][0])
                labels.append(extended_actions)
        inputs = np.stack(inputs, axis=0).astype(np.float64)
        labels = np.stack(labels, axis=0).astype(np.float64)
        if self.cameras:
            images = []
            for traj in self.demos:
                if traj['actions'].shape[0] > self.H:
                    for start in range(traj['actions'].shape[0] - self.H + 1):
                        images.append(traj['images'][start])
                else:
                    images.append(traj['images'][0])
            self.images = images
        self.inputs = np_to_tensor(inputs, self.device)
        self.labels = np_to_tensor(labels, self.device)
        self.labels = self.labels.reshape([self.labels.shape[0], -1]) # flatten actions to (#trajs, H * action_dim)

    # This not being used currently
    def process_simulation_demos(self):
        inputs, labels = [], []
        ac_chunk = 1 # TODO
        random.shuffle(self.demos)

        for traj in tqdm.tqdm(self.demos):
            imgs, acs = traj['images'], traj['actions']
            #assert len(obs) == len(acs) and len(acs) == len(imgs), "All time dimensions must match!"
            
            # pad camera dimension if needed
            #if len(imgs.shape) == 4:
            #    imgs = imgs[:,None]

            #for t in range(len(imgs) - ac_chunk):
                #i_t, o_t = imgs[t], obs[t]
                #i_t_prime, o_t_prime = imgs[t+ac_chunk], obs[t+ac_chunk]
                #a_t = acs[t:t+ac_chunk]
                #self.s_a_sprime.append(((i_t, o_t), a_t, (i_t_prime, o_t_prime)))
                #inputs.append(traj['images'][t])
                #labels.append(traj['actions'][t])
            inputs.extend(traj['images'])
            labels.extend(traj['actions'])
            if len(inputs) > 1000:
                break

        inputs = np.asarray(inputs).astype(np.float64)
        labels = np.asarray(labels).astype(np.float64)

        #inputs = inputs[:,None]
        #inputs = np.stack(inputs, axis=0).astype(np.float64)
        #labels = np.stack(labels, axis=0).astype(np.float64)
        self.inputs = np_to_tensor(inputs, self.device)
        self.labels = np_to_tensor(labels, self.device)
        print(self.labels.shape, self.inputs.shape)
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
