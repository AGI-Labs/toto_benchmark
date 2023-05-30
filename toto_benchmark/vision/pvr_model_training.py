import torch, torchvision, torchvision.transforms as T
import numpy as np, os, pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from toto_benchmark.vision import preprocess_image

# ===========================
# Utilities
# ===========================

def compute_accuracy(dataloader, model, config):
    # since we're not training, we don't need to calculate the gradients for our outputs
    model = model.eval()
    model = model.to(config['device'])
    mse_loss = torch.nn.MSELoss()
    loss, counter = 0.0, 0
    with torch.no_grad():
        for idx in range(config['eval_steps']):
            batch  = next(dataloader)
            out    = model(batch)
            tar    = batch['actions'].to(config['device'])
            b_loss = mse_loss(out, tar)
            loss += b_loss.item()
    return loss / config['eval_steps']


def get_time_chunk(path, key, t, window):
    if t >= window:
        return path[key][(t-window+1):(t+1)].astype(np.float32)
    else:
        return np.array([path[key][max(k, 0)] for k in range(t-window+1, t+1)]).astype(np.float32)


# ===========================
# Policies
# ===========================

class FrozenEmbeddingPolicy(torch.nn.Module):
    def __init__(self, embedding_dim: int,
                       joint_dim: int,
                       history_window: int,
                       hidden_dim: int,
                       act_dim: int,
                       mask_embedding: bool = False,
                       *args, **kwargs):
        super(FrozenEmbeddingPolicy, self).__init__()
        self.embedding_dim = embedding_dim
        self.joint_dim = joint_dim
        self.history_window = history_window
        self.mask_embedding = mask_embedding
        self.device = 'cpu'

        self.visual_bn   = torch.nn.BatchNorm1d(embedding_dim)
        self.visual_fc_1 = torch.nn.Linear(embedding_dim, hidden_dim)
        self.visual_fc_2 = torch.nn.Linear(hidden_dim * history_window, hidden_dim)

        self.joint_bn    = torch.nn.BatchNorm1d(joint_dim)
        self.joint_fc_1  = torch.nn.Linear(joint_dim, hidden_dim)
        self.joint_fc_2  = torch.nn.Linear(hidden_dim * history_window, hidden_dim)

        self.action_fc_1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.action_fc_2 = torch.nn.Linear(hidden_dim, act_dim)

        self.relu = torch.nn.ReLU()
    
    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, inp_dict: dict):
        """
            Expected keys for input dictionary are: "embeddings" and "joints"
        """
        embeddings, joints = inp_dict["embeddings"], inp_dict["joints"] 
        if type(embeddings) == np.ndarray:
            embeddings = torch.Tensor(embeddings).to(self.device)  # (B, T, E)
        else:
            embeddings = embeddings.to(self.device)
        if type(joints) == np.ndarray:
            joints = torch.Tensor(joints).to(self.device)          # (B, T, J)
        else:
            joints = joints.to(self.device)
        # if self.mask_embedding:
        #     embeddings = embeddings * 0.0
        z_viz = [ self.visual_bn(embeddings[:,k,:]) for k in range(self.history_window) ]
        z_viz = [ self.visual_fc_1(z) for z in z_viz ]
        z_viz = torch.cat([ self.relu(z) for z in z_viz ], dim=1)
        z_jnt = [ self.joint_bn(joints[:,k,:]) for k in range(self.history_window) ]
        z_jnt = [ self.joint_fc_1(z) for z in z_jnt ]
        z_jnt = torch.cat([ self.relu(z) for z in z_jnt ], dim=1)
        z_viz = self.relu(self.visual_fc_2(z_viz))
        z_jnt = self.relu(self.joint_fc_2(z_jnt))
        z_act = torch.cat([z_viz, z_jnt], dim=1)
        z_act = self.relu(self.action_fc_1(z_act))
        action = self.action_fc_2(z_act)
        return action

class VisuoMotorPolicy(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, 
                       policy: torch.nn.Module):
        super(VisuoMotorPolicy, self).__init__()
        self.base_model = base_model
        self.policy = policy
        self.history_window = self.policy.history_window
        self.embedding_dim = self.policy.embedding_dim
        self.joint_dim = self.policy.joint_dim
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.base_model = self.base_model.to(device)
        self.policy = self.policy.to(device)
        return super().to(device)

    def forward(self, inp_dict: dict):
        """
            Expected keys for input dictionary are: "images" and "joints"
            Images shape: (B, history_window, 3, 224, 224)
            Joints shape: (B, history_window, joint_dim)
            Example at inference time:
                Images shape: (1, 3 (history), 3 (RGB), 224 (H), 224 (W))
                Joints shape: (1, 3 (history), 9 or 12)
        """
        images = torch.Tensor(inp_dict['images']).to(self.device)
        joints = torch.Tensor(inp_dict['joints']).to(self.device)
        batch_size = joints.shape[0]
        assert images.shape[0] == joints.shape[0]
        assert images.shape[1:] == (self.history_window, 3, 224, 224)
        assert joints.shape[1:] == (self.history_window, self.joint_dim)
        # reshape images for a fast batched forward pass
        embeddings = self.base_model(images.reshape(-1, 3, 224, 224)) # (B*T, E)
        embeddings = embeddings.view(-1, self.history_window, self.embedding_dim)
        return self.policy(dict(embeddings=embeddings, joints=joints))


# ===========================
# Datasets and Loading
# ===========================

class FrozenEmbeddingDataset(torch.utils.data.IterableDataset):
    def __init__(self, paths, history_window, config):
        self.paths = paths
        self.num_paths = len(paths)
        self.history_window = history_window
        self.config = config
    
    def _sample(self):
        # randomly pick a path
        p_idx = np.random.choice(self.num_paths)
        path  = self.paths[p_idx]
        path_len = path[self.config['joint_key']].shape[0]
        # randomly pick a time index
        t_idx = np.random.choice(path_len)
        # obtain timewindow from path
        embeddings = get_time_chunk(path, 'embeddings', t_idx, self.history_window)
        joints = get_time_chunk(path, self.config['joint_key'], t_idx, self.history_window)
        actions = path['actions'][t_idx].astype(np.float32)
        return {'embeddings': embeddings, 'joints': joints, 'actions': actions}

    def __iter__(self):
        while True:
            yield self._sample()


def precompute_embeddings(model, paths, config, transforms):
    # data parallel mode to use both GPUs
    model = model.to(config['device'])
    model = torch.nn.DataParallel(model)
    model = model.eval()
    data_path = config['data_dir'] + '/data/'
    batch_size = config['batch_size']
    print("Total number of paths : %i" % len(paths))
    for idx, path in tqdm(enumerate(paths)):
        path_images = []
        for t in range(path[config['joint_key']].shape[0]):
            img = Image.open(os.path.join(data_path, path['traj_id'], path['cam0c'][t]))  # Assuming RGB for now
            img = preprocess_image(img, transforms)
            path_images.append(img)
        # compute the embedding
        path_len = len(path_images)
        embeddings = []
        with torch.no_grad():
            for b in range((path_len // batch_size + 1)):
                if b * batch_size < path_len:
                    chunk = torch.stack(path_images[b*batch_size:min(batch_size*(b+1), path_len)])
                    chunk_embed = model(chunk.to(config['device']))
                    embeddings.append(chunk_embed.to('cpu').data.numpy())
            embeddings = np.vstack(embeddings)
            assert embeddings.shape == (path_len, chunk_embed.shape[1])
        path['embeddings'] = embeddings.copy()
    return paths


# ===========================
# Model training
# ===========================

def train_policy_freeze_embedding(config, run=None):
    # construct the visuomotor policy
    from vision.pvr_model_loading import load_pvr_model
    base_model, embedding_dim, transforms = load_pvr_model(config['base_model'])
    control_policy = FrozenEmbeddingPolicy(embedding_dim=embedding_dim, joint_dim=config['joint_dim'],
                                            history_window=config['history_window'],
                                            hidden_dim=config['hidden_dim'],
                                            act_dim=config['action_dim'],
                                            mask_embedding=config['mask_embedding'])
    visuo_motor_policy = VisuoMotorPolicy(base_model=base_model, policy=control_policy)
    
    # load the dataset
    paths = pickle.load(open(config['data_dir'] + config['data_file'], 'rb'))
    # filter the paths
    paths = [p for p in paths if p['rewards'].sum() >= config['filter_reward']]

    print("Number of trajectories considered after filtering : %i" % len(paths))

    # split training and test sets
    paths = precompute_embeddings(visuo_motor_policy.base_model, paths, config, transforms)
    shuffle_idx = np.random.permutation(len(paths))
    split_idx = int(0.9 * len(paths))
    tr_paths  = [paths[idx] for idx in shuffle_idx[:split_idx]]
    te_paths  = [paths[idx] for idx in shuffle_idx[split_idx:]]

    tr_data = FrozenEmbeddingDataset(tr_paths, config['history_window'], config)
    te_data = FrozenEmbeddingDataset(te_paths, config['history_window'], config)

    trainloader = iter(torch.utils.data.DataLoader(tr_data, batch_size=config['batch_size'], num_workers=2))
    testloader  = iter(torch.utils.data.DataLoader(te_data, batch_size=config['batch_size'], num_workers=2))
    
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(visuo_motor_policy.policy.parameters(), lr=config['lr'])
    
    # train loop
    visuo_motor_policy = visuo_motor_policy.to(config['device'])
    visuo_motor_policy.base_model = visuo_motor_policy.base_model.eval()
    visuo_motor_policy.policy = visuo_motor_policy.policy.train()
    # visuo_motor_policy.policy = torch.nn.DataParallel(visuo_motor_policy.policy)
    log_step_ctr = 0
    for idx in range(config['train_steps']):
        batch = next(trainloader)
        # import ipdb; ipdb.set_trace()
        out   = visuo_motor_policy.policy(batch)
        tar   = batch['actions'].to(config['device'])
        
        optimizer.zero_grad()
        loss  = loss_func(out, tar)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()

        if idx % int(config['log_frequency']) == 0 and idx > 0:
            print(f'train_step: %i | loss: %2.4f' % (idx, loss_val))
            if run is not None:
                run.log({'train/step': idx, 'train/loss': loss.item()}, step=log_step_ctr)
            log_step_ctr += 1

        if idx % int(config['eval_frequency']) == 0 and idx > 0:
            eval_loss = compute_accuracy(testloader, visuo_motor_policy.policy, config)
            print(f'train_step: %i | loss: %2.4f' % (idx, eval_loss))
            if run is not None:
                run.log({'eval/loss': eval_loss}, step=log_step_ctr)

            save_file = '%s_policy.pickle' % config['base_model']
            torch.save(visuo_motor_policy.policy, save_file)
