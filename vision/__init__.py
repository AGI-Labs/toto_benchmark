def load_model(config):
    model_type = config.agent.vision_model
    if model_type == 'resnet':
        from .Resnet import _load_model
    return _load_model(config)


def load_transforms(config):
    model_type = config.agent.vision_model
    if model_type == 'resnet':
        from .Resnet import _load_transforms
    return _load_transforms(config)