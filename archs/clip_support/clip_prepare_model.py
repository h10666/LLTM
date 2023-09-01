# from torch._C import Module
from CLIP.clip.model import *
from CLIP.clip import clip
import os
import warnings


if torch.__version__.split(".") < ["1", "7", "1"]:
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

def clip_prepare_model(name, device, jit, download_root = None):
    if name in clip._MODELS:
        # model_path = clip._download(clip._MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
        model_path = "%s/%s.pt" % (download_root, name)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {clip.available_models()}")
    # exit(0)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location= device if jit else "cpu").eval()
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")
    
    return model.state_dict()

