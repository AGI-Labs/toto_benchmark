from pathlib import Path
import torch
from torch import nn
from torchvision import models
import toto_benchmark

base_path = Path(toto_benchmark.__file__).parent.parent
CHECKPOINT_DIR = f'{base_path}/assets'

class Identity(nn.Module):
    '''
    Author: Janne Spijkervet
    url: https://github.com/Spijkervet/SimCLR
    '''
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def _load_model(config):
    vision_model = models.resnet18(pretrained=False)
    if config.agent.vision_model == 'byol_scoop':
        encoder_state_dict = torch.load(f'{CHECKPOINT_DIR}/BYOL_18_scoop_100.pt', map_location=torch.device('cpu'))
    else:
        encoder_state_dict = torch.load(f'{CHECKPOINT_DIR}/BYOL_18_pour_100.pt', map_location=torch.device('cpu'))
    vision_model.load_state_dict(encoder_state_dict['model_state_dict'])
    vision_model.fc = Identity()
    return vision_model

def _load_transforms(config):
    from torchvision import transforms as T
    img_transforms = T.Compose([T.ToTensor(),
                                T.Resize((224,224)),
                                T.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    return img_transforms
