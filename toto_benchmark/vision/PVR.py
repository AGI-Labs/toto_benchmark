from toto_benchmark.vision.pvr_model_loading import load_pvr_model, load_pvr_transforms

def _load_model(config):
    vision_model_name = config.agent.vision_model
    model = load_pvr_model(vision_model_name)[0]
    model = model.eval() ## assume this model is used in eval
    return model

def _load_transforms(config):
    vision_model_name = config.agent.vision_model
    return load_pvr_transforms(vision_model_name)[1]
