from .Agent import Agent

import torchvision.transforms as T
import numpy as np

NUM_JOINTS = 7

class CollaboratorAgent(Agent):
    def __init__(self, min=-0.1, max=0.1, step=0.01):
        # TODO: replace with your init function
        self.delta_range = np.arange(min, max, step)

    def predict(self, observation: dict):
        # TODO: replace with your predict function
        return np.random.choice(self.delta_range, NUM_JOINTS)

def _init_agent_from_config(config, device='cpu'):
    # TODO: replace with your init_agent_from_config function
    agent = CollaboratorAgent()
    return agent, lambda x: x