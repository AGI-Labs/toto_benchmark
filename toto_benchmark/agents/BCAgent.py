import os
import pickle
import torch
from torch import nn
from .BaseAgent import BaseAgent

class BCAgent(BaseAgent):
    def __init__(self, models, learning_rate, device, H=1):
        super(BCAgent, self).__init__(models, learning_rate, device)

        self.H, self.t, self.cache = H, 0, None

    def forward(self, sample):
        return self.models['decoder'](sample['inputs'])

    def compute_loss(self, sample):
        output = self.forward(sample)
        labels = sample['labels']
        losses = self.loss_fn(output.view(-1, output.size(-1)), labels)
        self.loss = self.loss_reduction(losses)

    def predict(self, sample):
        if self.H == 1:
            [m.eval() for m in self.models.values()]
            with torch.no_grad():
                sample = self.pack_one_batch(sample)
                output = self.forward(sample)[0].to('cpu').detach().numpy()
                return output
        else:
            index = self.t % self.H
            if index == 0:
                [m.eval() for m in self.models.values()]
                with torch.no_grad():
                    sample = self.pack_one_batch(sample)
                    output = self.forward(sample)[0].to('cpu').detach().numpy()
                    self.cache = output.reshape([self.H, -1])
            self.t += 1
            return self.cache[index]

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

def get_stats(arr):
    arr_std = arr.std(0)
    arr_std[arr_std < 1e-4] = 1
    return len(arr_std), arr.mean(0), arr_std

class DeepMLPBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, dropout_p=0.1):
        super().__init__()
        self.fc = nn.Linear(inp_dim, out_dim)
        self.drop = nn.Dropout(p=dropout_p)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.drop(self.relu(self.fc(x)))

class Policy(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim=128):
        super(Policy, self).__init__()
        self.fc1 = DeepMLPBlock(inp_dim, hidden_dim)
        self.fc2 = DeepMLPBlock(hidden_dim, hidden_dim)
        self.final = nn.Linear(hidden_dim, out_dim)
        self.register_buffer("inp_mean", torch.zeros(inp_dim))
        self.register_buffer("inp_std", torch.ones(inp_dim))
        self.register_buffer("out_mean", torch.zeros(out_dim))
        self.register_buffer("out_std", torch.ones(out_dim))

    def set_stats(self, dataset):
        inp_dim, inp_mean, inp_std = get_stats(dataset.inputs)
        _, out_mean, out_std = get_stats(dataset.labels)

        self.inp_mean[:inp_dim].copy_(inp_mean)
        self.inp_std[:inp_dim].copy_(inp_std)
        self.out_mean.copy_(out_mean)
        self.out_std.copy_(out_std)

    def save_stats(self, foldername, filename='policy_stats.pkl'):
        policy_stats = {
            'inp_mean': self.inp_mean,
            'inp_std': self.inp_std,
            'out_mean': self.out_mean,
            'out_std': self.out_std
        }
        with open(os.path.join(foldername, filename), 'wb') as handle:
            pickle.dump(policy_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_stats(self, foldername, filename='policy_stats.pkl'):
        policy_stats = pickle.load(open(os.path.join(foldername, filename), 'rb'))
        self.inp_mean.copy_(policy_stats['inp_mean'])
        self.inp_std.copy_(policy_stats['inp_std'])
        self.out_mean.copy_(policy_stats['out_mean'])
        self.out_std.copy_(policy_stats['out_std'])

    def forward(self, observations):
        h = (observations - self.inp_mean) / self.inp_std
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.final(h)
        actions = self.out_mean + self.out_std * h
        return actions

def _init_agent_from_config(config, device='cpu', normalization=None):
    if 'H' in config.data.keys():
        H = config.data.H
    else:
        H = 1

    if 'hidden_dim' in config['agent']:
        hidden_dim = config.agent.hidden_dim
    else:
        hidden_dim = 128
    models = {
        'decoder': Policy(
            config.data.in_dim,
            config.data.out_dim * H, 
            hidden_dim)
    }

    if normalization is not None:
        models['decoder'].set_stats(normalization)
    else:
        assert os.path.isfile(os.path.join(config.saved_folder, 'policy_stats.pkl'))
        models['decoder'].load_stats(config.saved_folder)

    for k,m in models.items():
        m.to(device)
        if k=="img_encoder" and config.model.use_resnet:
            print("*** Resnet image encoder, do not init weight")
        else:
            m.apply(init_weights)

    bc_agent = BCAgent(models, config.training.lr, device, H)
    # load weights if exist (during inference)
    if os.path.isfile(os.path.join(config.saved_folder, 'Agent.pth')):
        bc_agent.load(config.saved_folder)
    return bc_agent, None # image transforms is None for BCAgent