import fsspec
import torch

def pt_load(file_path, map_location=None):
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location, weights_only=False)
    return out

def msa_model_key_fix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            new_key = k[len('encoder.'):]  # 'encoder.' 제거
        else:
            new_key = k
        new_state_dict[new_key] = v

    return new_state_dict