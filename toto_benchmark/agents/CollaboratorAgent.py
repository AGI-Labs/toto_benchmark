"""
If you are contributing a new agent, implement your 
agent initialization & predict functions here. 

Next, (optionally) update your agent's name in agent/__init__.py.
"""

import numpy as np
import torchvision.transforms as T
from .Agent import Agent

NUM_JOINTS = 7

class CollaboratorAgent(Agent):
    def __init__(self, min=-0.1, max=0.1, step=0.01):
        # TODO: replace with your init function (including init arguments)
        self.delta_range = np.arange(min, max, step)

    def predict(self, observation: dict):
        # TODO: replace with your predict function
        return np.random.choice(self.delta_range, NUM_JOINTS)

def _init_agent_from_config(config, device='cpu'):
    # TODO: replace with your init_agent_from_config function
    agent = CollaboratorAgent()
    transforms = lambda x: x
    # if (optionally) choose to use the provided vision models or adopt 
    # the same structure for your customized vision model, the model
    # and the transforms can be directly initialized as follows:
    #
    # from vision import load_model, load_transforms
    # img_encoder = load_model(config)
    # transforms = load_transforms(config)
    return agent, transforms