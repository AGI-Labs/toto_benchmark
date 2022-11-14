import os
from PIL import Image
import torch
from torch import nn
import torchvision.models as models
from toto_benchmark.vision import load_model, load_transforms
from .Agent import Agent

class BCImageAgent(Agent):
    def __init__(self, models, learning_rate, device, cameras):
        self.models = models
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        self.loss_reduction = torch.mean
        self.parameters = sum([list(m.parameters()) for m in self.models.values()], [])
        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)
        self.device = device
        self.epoch = 0
        self.item_losses = None
        self.cameras = cameras
    
    def compute_loss(self, sample):
        output = self.forward(sample)
        labels = sample['labels']
        losses = self.loss_fn(output.view(-1, output.size(-1)), labels)
        self.loss = self.loss_reduction(losses)
    
    def eval(self, sample):
        [m.eval() for m in self.models.values()]
        with torch.no_grad():
            self.compute_loss(sample)
            return self.loss.item()
    
    def forward(self, sample):
        imgs = [sample[_c] for _c in self.cameras]
        imgs_out = [self.models['img_encoder'](img) for img in imgs]
        concat_inputs = torch.cat([sample['inputs']] + imgs_out, dim=-1)
        return self.models['decoder'](concat_inputs)

    def load(self, foldername, device=None, filename='Agent.pth'):
        if device is not None:
            self.device = device
        checkpoint = torch.load(os.path.join(foldername, filename), map_location=torch.device(self.device))
        self.epoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for mname, m in self.models.items():
            if mname in checkpoint:
                m.load_state_dict(checkpoint[mname])
                m = m.to(self.device)
            else:
                m = m.to(self.device)
                print(f"Not loading {mname} from checkpoint")


    def pack_one_batch(self, sample):
        for k, v in sample.items():
            t = v if torch.is_tensor(v) else torch.from_numpy(v)
            sample[k] = t.float().unsqueeze(0).to(self.device)
        return sample

    def predict(self, sample):
        [m.eval() for m in self.models.values()]
        with torch.no_grad():
            sample = self.pack_one_batch(sample)
            output = self.forward(sample)[0].to('cpu').detach().numpy()
            return output
    
    def save(self, foldername, filename='Agent.pth'):
        state = {'epoch': self.epoch,
                 'optimizer': self.optimizer.state_dict(),
                 }
        for mname, m in self.models.items():
            if mname != 'img_encoder': 
                state[mname] = m.state_dict()
        torch.save(state, os.path.join(foldername, filename))

    def train(self, sample):
        [m.train() for m in self.models.values()]
        self.zero_grad()
        self.compute_loss(sample)
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()

    def zero_grad(self):
        [m.zero_grad() for m in self.models.values()] 
        self.optimizer.zero_grad()

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
    def __init__(self, inp_dim, out_dim):
        super(Policy, self).__init__()
        self.fc1 = DeepMLPBlock(inp_dim, 128)
        self.fc2 = DeepMLPBlock(128, 128)
        self.final = nn.Linear(128, out_dim)
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

    def forward(self, observations):
        h = (observations - self.inp_mean) / self.inp_std
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.final(h)
        actions = self.out_mean + self.out_std * h
        return actions

def _init_agent_from_config(config, device='cpu', normalization=None):
    assert len(config.data.images.cameras) > 0
    img_encoder = load_model(config)
    transforms = load_transforms(config)

    # Pass in empty image to get model output dimensions
    dummy_image = transforms(Image.new('RGB', (config.data.images.im_h, config.data.images.im_w)))
    dummy_out = img_encoder(torch.unsqueeze(dummy_image, 0))

    img_input_size = len(config.data.images.cameras) * dummy_out.shape[1]
    input_dim = config.data.in_dim

    models = {
        'img_encoder': img_encoder,
        'decoder': Policy(
            input_dim + img_input_size,
            config.data.out_dim
            )
    }

    if normalization is not None:
        models['decoder'].set_stats(normalization)

    for k,m in models.items():
        m.to(device)
        if k == "img_encoder":
            print("*** Resnet image encoder, init weight only on FC layers")
            m.fc.apply(init_weights)
        else:
            m.apply(init_weights)

    agent = BCImageAgent(models, config.training.lr, device,
        config.data.images.cameras)
    
    return agent, transforms
