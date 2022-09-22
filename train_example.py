"""Train a RoboCloud agent.

Example command:
python train_example.py --config-name train_bcimage.yaml data.logs_folder=/RoboCloud/cloud-dataset-scooping-v0

Hyperparameters can be set in confs/train_bcimage.yaml
"""

import logging
import os
import pickle
import numpy
import torch

from omegaconf import DictConfig, OmegaConf, open_dict
import baselines
import hydra

from torch.utils.data import DataLoader, random_split
from agents import init_agent_from_config
from dataset_traj import FrankaDatasetTraj
from vision import load_transforms

log = logging.getLogger(__name__)

def global_seeding(seed=0):
    torch.manual_seed(seed)
    numpy.random.seed(seed)

@hydra.main(config_path="conf", config_name="train_bc")
def main(cfg : DictConfig) -> None:
    with open_dict(cfg):
        cfg['saved_folder'] = os.getcwd()

    print(OmegaConf.to_yaml(cfg, resolve=True))

    with open(os.path.join(os.getcwd(), 'hydra.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    global_seeding(cfg.training.seed)

    try:
        with open(os.path.join(cfg.data.logs_folder, cfg.data.pickle_fn), 'rb') as f:
            data = pickle.load(f)
    except:
        print("\n***Pickle does not exist. Make sure the pickle is in the logs_folder directory.")
        raise

    dset = FrankaDatasetTraj(data,
        logs_folder=cfg.data.logs_folder,
        subsample_period=cfg.data.subsample_period,
        im_h=cfg.data.images.im_h,
        im_w=cfg.data.images.im_w,
        obs_dim=cfg.data.in_dim,
        action_dim=cfg.data.out_dim,
        H=cfg.data.H,
        device=cfg.training.device,
        cameras=cfg.data.images.cameras,
        img_transform_fn=load_transforms(cfg),
        noise=cfg.data.noise)

    agent, _ = init_agent_from_config(cfg, cfg.training.device, normalization=dset)

    split_sizes = [int(len(dset) * 0.8), len(dset) - int(len(dset) * 0.8)]
    train_set, test_set = random_split(dset, split_sizes)

    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.training.batch_size)
    train_metric, test_metric = baselines.Metric(), baselines.Metric()

    for epoch in range(cfg.training.epochs):
        acc_loss = 0.
        train_metric.reset()
        test_metric.reset()
        batch = 0
        for data in train_loader:
            for key in data:
                data[key] = data[key].to(cfg.training.device)
            agent.train(data)
            acc_loss += agent.loss
            train_metric.add(agent.loss.item())
            print('epoch {} \t batch {} \t train {:.6f}'.format(epoch, batch, agent.loss.item()), end='\r')
            batch += 1

        for data in test_loader:
            for key in data:
                data[key] = data[key].to(cfg.training.device)
            test_metric.add(agent.eval(data))
        log.info('epoch {} \t train {:.6f} \t test {:.6f}'.format(epoch, train_metric.mean, test_metric.mean))

        log.info(f'Accumulated loss: {acc_loss}')
        if epoch % cfg.training.save_every_x_epoch == 0:
            agent.save(os.getcwd())

    agent.save(os.getcwd())
    print("Saved agent to",os.getcwd())

if __name__ == '__main__':
    main()
