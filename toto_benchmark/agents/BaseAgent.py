import os
import torch
from .Agent import Agent

class BaseAgent(Agent):
    def __init__(self, models, learning_rate=1e-3, device='cpu'):
        self.models = models
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        self.loss_reduction = torch.mean
        self.parameters = sum([list(m.parameters()) for m in self.models.values()], [])
        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)
        self.device = device
        self.epoch = 0
        self.item_losses = None

    def save(self, foldername, filename='Agent.pth'):
        state = {'epoch': self.epoch,
                 'optimizer': self.optimizer.state_dict(),
                 }
        for mname, m in self.models.items():
            state[mname] = m.state_dict()
            m.save_stats(foldername)
        torch.save(state, os.path.join(foldername, filename))

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

    def zero_grad(self):
        [m.zero_grad() for m in self.models.values()] 
        self.optimizer.zero_grad()

    def train(self, sample):
        [m.train() for m in self.models.values()]
        self.zero_grad()
        self.compute_loss(sample)
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()

    def eval(self, sample):
        [m.eval() for m in self.models.values()]
        with torch.no_grad():
            self.compute_loss(sample)
            return self.loss.item()

    def pack_one_batch(self, sample):
        for k, v in sample.items():
            t = v if torch.is_tensor(v) else torch.from_numpy(v)
            sample[k] = t.float().unsqueeze(0).to(self.device)
        return sample

    # For query of a single datapoint query only!
    def predict(self, sample):
        raise NotImplementedError

    def compute_loss(self, sample):
        raise NotImplementedError

