import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import yaml, argparse, os, pickle
from toto_benchmark.scripts.utils import Namespace
from toto_benchmark.vision import load_model, load_transforms, preprocess_image

def precompute_embeddings(cfg, paths, data_path=None, from_files=True):
    device = 'cuda:0'
    model = load_model(cfg)
    model.to(device)
    model = model.eval()
    transforms = load_transforms(cfg)
    batch_size = 128
    print("Total number of paths : %i" % len(paths))
    for idx, path in tqdm(enumerate(paths)):
        path_images = []
        for t in range(path['observations'].shape[0]):
            if from_files:
                assert data_path is not None
                img = Image.open(os.path.join(data_path, path['traj_id'], path['cam0c'][t]))
            else:
                img = path['images'][t]
            img = preprocess_image(img, transforms)
            path_images.append(img)
        embeddings = []
        path_len = len(path_images)
        with torch.no_grad():
            for b in range((path_len // batch_size + 1)):
                if b * batch_size < path_len:
                    chunk = torch.stack(path_images[b * batch_size:min(batch_size * (b + 1), path_len)])
                    chunk_embed = model(chunk.to(device))
                    embeddings.append(chunk_embed.to('cpu').data.numpy())
            embeddings = np.vstack(embeddings)
            assert embeddings.shape == (path_len, chunk_embed.shape[1])
        path['embeddings'] = embeddings.copy()
    return paths

def precompute_embeddings_byol(cfg, paths, data_path):
    device = 'cuda:0'
    model = load_model(cfg)
    model.to(device)
    model = model.eval()
    byol_transforms = load_transforms(cfg)
    batch_size = 1
    print("Total number of paths : %i" % len(paths))
    for idx, path in tqdm(enumerate(paths)):
        path_images = []
        for t in range(path['observations'].shape[0]):
            img = Image.open(os.path.join(data_path, path['traj_id'], path['cam0c'][t]))  
            if cfg['data']['images']['crop']:
                img = img.crop((200, 0, 500, 400))
            img = preprocess_image(img, byol_transforms)
            path_images.append(img)
        embeddings = []
        path_len = len(path_images)
        with torch.no_grad():
            for b in range((path_len // batch_size + 1)):
                if b * batch_size < path_len:
                    chunk = torch.stack(path_images[b * batch_size:min(batch_size * (b + 1), path_len)])
                    chunk_embed = model(chunk.to(device))
                    embeddings.append(chunk_embed.to('cpu').data.numpy())
            embeddings = np.vstack(embeddings)
            assert embeddings.shape == (path_len, chunk_embed.shape[1])
        path['embeddings'] = embeddings.copy()
        path['observations'] = np.hstack([path['observations'], path['embeddings']])
    return paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_folder', type=str, default='assets/cloud-dataset-scooping/')
    parser.add_argument('-e', '--vision_model', type=str, default='moco_conv5')
    args = parser.parse_args()

    # default to use train_bc.yaml config
    with open(os.path.join(os.getcwd(), 'conf', 'train_bc.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = Namespace(cfg)

    cfg['agent']['vision_model_path'] = os.path.join(os.getcwd(), cfg['agent']['vision_model_path'])
    cfg['agent']['vision_model'] = args.vision_model
    # assume the file to be processed is parsed.pkl
    paths = pickle.load(open(os.path.join(os.getcwd(), args.data_folder, 'parsed.pkl'), 'rb')) 
    data_path = os.path.join(args.data_folder, 'data')

    if cfg['agent']['vision_model'] == 'byol':
        # support batch_size = 1 and img cropping
        paths_with_embeddings = precompute_embeddings_byol(cfg, paths, data_path)
    else:
        # support batch_size >= 1
        paths_with_embeddings = precompute_embeddings(cfg, paths, data_path=data_path)

    with open(os.path.join(args.data_folder, f'parsed_with_embeddings_{args.vision_model}.pkl'), 'wb') as f:
        pickle.dump(paths_with_embeddings, f)
