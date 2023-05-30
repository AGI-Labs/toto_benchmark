from torch import Tensor
from abc import ABC, abstractmethod

class Agent(ABC):
    """Abstract class for the agent
    """
    @abstractmethod
    def __init__(self, **kwargs):
        """Setup the agent with the necessary state and config here.
        """
        pass

    @abstractmethod
    def predict(self, observation: dict) -> Tensor:
        """Predicts the next action based on current observation

        Args:
            observation: A dict containing the following keys
            "inputs": A numpy array which contains the current robot joint state (7, ).
            "cam0c": Output after applying agent-specific image_transforms.
                     i.e. image_transforms(img) where img is a PIL image initialized 
                     with an numpy array of shape (480, 640, 3) 
            "cam0d": A numpy array which contains the depth image.

        Returns:
            A numpy array of shape (7, ) which is the action to be executed on the robot
        """
        pass

def _init_agent_from_config(config, device='cpu'):
    """Initializes an agent using a custom config
    
    Args:
        config: A hydra config which contains all the required and optional parameters for the agent. 
                A starting file has been provided at outputs/collaborator_agent/hydra.yaml
        device: The device on which to load the agent

    Returns:
        A Tuple which contains (agent, image_transforms)
        agent: an agent that conforms to the Agent abstract class.
        image_transforms: a function containing the transformations to be applied to the camera image. E.g. a torchvision transform
        Note: Set image_transforms to be the identity function (lambda x: x) if no explicit image transformation is needed
    """
    pass