"""Train a TOTO agent.

Example command:
python train.py --config-name train_bc.yaml 

Hyperparameters can be set in corresponding .yaml files in confs/
"""

import baselines
import hydra
import logging
import numpy
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import pickle
import torch
from torch.utils.data import DataLoader, random_split
import wandb

from dataset_traj import FrankaDatasetTraj
from toto_benchmark.agents import init_agent_from_config
from toto_benchmark.vision import load_transforms, EMBEDDING_DIMS

log = logging.getLogger(__name__)

def global_seeding(seed=0):
    torch.manual_seed(seed)
    numpy.random.seed(seed)

@hydra.main(config_path="../conf", config_name="train_bc")
def main(cfg : DictConfig) -> None:
    with open_dict(cfg):
        cfg['saved_folder'] = os.getcwd()
        print("Model saved dir: ", cfg['saved_folder'])

        if 'crop' not in cfg['data']['images']:
            cfg['data']['images']['crop'] = False
        if 'H' not in cfg['data']:
            cfg['data']['H'] = 1
        cfg['data']['logs_folder'] = os.path.dirname(cfg['data']['pickle_fn']) 

    if cfg.agent.type in ['bcimage', 'bcimage_pre']:
        cfg['data']['images']['per_img_out'] = EMBEDDING_DIMS[cfg['agent']['vision_model']]
        if cfg.agent.type == 'bcimage_pre':
            # assume in_dim is without adding the image embedding dimensions
            cfg['data']['in_dim'] = cfg['data']['in_dim'] + cfg['data']['images']['per_img_out']

    print(OmegaConf.to_yaml(cfg, resolve=True))
    with open(os.path.join(os.getcwd(), 'hydra.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    global_seeding(cfg.training.seed)
    print(hydra.utils.get_original_cwd(),cfg.data.pickle_fn)

    agent_name = cfg['saved_folder'].split('outputs/')[-1]
    flat_dict = {}
    for key in ['data', 'agent', 'training']:
        flat_dict.update(dict(cfg[key]))
    wandb.init(project="toto-bc", config=flat_dict)
    wandb.run.name = "{}".format(agent_name)

    try:
        with open(os.path.join(hydra.utils.get_original_cwd(), cfg.data.pickle_fn), 'rb') as f:
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
        top_k=cfg.data.top_k,
        device=cfg.training.device,
        cameras=cfg.data.images.cameras,
        img_transform_fn=load_transforms(cfg),
        noise=cfg.data.noise,
        crop_images=cfg.data.images.crop)
    del data
    split_sizes = [int(len(dset) * 0.8), len(dset) - int(len(dset) * 0.8)]
    train_set, test_set = random_split(dset, split_sizes)

    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.training.batch_size)
    agent, _ = init_agent_from_config(cfg, cfg.training.device, normalization=dset)
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

        wandb.log({"Train Loss": train_metric.mean, "Epoch": epoch})
        wandb.log({"Test Loss": test_metric.mean, "Epoch": epoch})
        wandb.log({"Acc Train Loss": acc_loss, "Epoch": epoch})

    agent.save(os.getcwd())
    log.info("Saved agent to {}".format(os.getcwd()))

if __name__ == '__main__':
    main()
