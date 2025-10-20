import re
from collections import defaultdict

image_block_pattern = "^blocks\.(\d+)"
exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n    # 차원이 1 또는 scalar 값, batch_norm, layer_norm, bias를 분리 

def clip_layer_wise_lr_decay(nameparams, pattern, initial_lr, decay):
    for p in nameparams:
        # if not p['params'].requires_grad: continue
        block_match = re.search(pattern, p['name'])
        if block_match:
            block_num = int(block_match.group(1))
            p['lr'] = initial_lr * decay**block_num
        else:
            p['lr'] = initial_lr

    return nameparams

def exclude_weight_decay(nameparams, weight_decay):
    for p in nameparams:
        if exclude(p['name'], p['params']):
            p['weight_decay'] = 0
        else:
            p['weight_decay'] = weight_decay

    return nameparams

def bind_group(nameparams, base_lr, base_wd):
    grouped_params = defaultdict(list)
    
    for p in nameparams:
        p['lr'] = p.get('lr', base_lr)
        p['weight_decay'] = p.get('weight_decay', base_wd)
        
        key = (p['lr'], p['weight_decay'])
        
        grouped_params[key].append(p['params'])
        
    param_groups = [{'params': params, 'lr': lr, 'weight_decay': wd} 
                    for (lr, wd), params in grouped_params.items()]
    
    return param_groups

def make_lwld_params_group(model, base_lr, decay, weight_decay):
    grad_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    rest_params = [{"name": name, "params": param, 'lr': base_lr} for name, param in grad_params if 'blocks' not in name]
    block_params = [{"name": name, "params": param} for name, param in grad_params if 'blocks' in name]
    block_params = clip_layer_wise_lr_decay(block_params, image_block_pattern, base_lr, decay)
    all_params = rest_params + block_params

    all_params = exclude_weight_decay(all_params, weight_decay)

    param_groups = bind_group(all_params, base_lr, weight_decay)

    return param_groups


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, block_depth=12):
    depthness = block_depth + 1
    if (
        "pos_embed" in name
        or "patch_embed" in name
        or "mask_token" in name
        or "cls_token" in name
        or "reg_token" in name
        or "decoder_embed" in name
    ):
        depthness = 0
    elif "blocks." in name and "residual." not in name:
        depthness = int(name[name.find("blocks.") :].split(".")[1]) + 1

    return lr_decay_rate ** (block_depth + 1 - depthness)

def get_transformer_lr_decay_rate(name, lr_decay_rate=1.0, block_depth=12):
    depthness = block_depth + 1
    if (
        "pos_embed" in name
        or "patch_embed" in name
        or "mask_token" in name
        or "cls_token" in name
        or "reg_token" in name
        or "token_embedding" in name
        or "positional_embedding" in name
    ):
        depthness = 0
    elif "blocks." in name and "residual." not in name:
        depthness = int(name[name.find("blocks.") :].split(".")[1]) + 1

    return lr_decay_rate ** (block_depth + 1 - depthness)

def fuse_params_groups(all_params_groups, keys=("lr_multiplier", "wd_multiplier")):
    fused_params_groups = defaultdict(lambda: {"params": []})
    for d in all_params_groups:
        identifier = ""
        for k in keys:
            identifier += k + str(d[k]) + "_"

        for k in keys:
            fused_params_groups[identifier][k] = d[k]
        fused_params_groups[identifier]["params"].append(d["params"])

    return fused_params_groups.values()


def make_param_groups(model, lr_decay_rate=1.0, patch_embed_lr_mult=1.0):
    param_groups = []

    block_depth = len(model.visual.blocks)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        decay_rate = get_transformer_lr_decay_rate(name, lr_decay_rate, block_depth)
        d = {"params": param, "lr_multiplier": decay_rate, "wd_multiplier": 1.0, "name": name}

        if name.endswith(".bias") or "norm" in name or "gamma" in name or "logit" in name:
            d.update({"wd_multiplier": 0.0})

        if "patch_embed" in name:
            d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})
        
        param_groups.append(d)
     
    param_groups = fuse_params_groups(param_groups)
    for g in param_groups:
        g["foreach"] = True

    return list(param_groups)

def make_mae_param_groups(model, lr_decay_rate=1.0, patch_embed_lr_mult=1.0):
    param_groups = []

    vit_depth = len(model.visual.blocks)
    decoder_depth = len(model.decoder.decoder_blocks)
    for name, param in model.named_parameters():
        if "visual" in name:
            if not param.requires_grad:
                continue

            decay_rate = get_transformer_lr_decay_rate(name, lr_decay_rate, vit_depth)
            d = {"params": param, "lr_multiplier": decay_rate, "wd_multiplier": 1.0, "name": name}

            if name.endswith(".bias") or "norm" in name or "gamma" in name or "logit" in name:
                d.update({"wd_multiplier": 0.0})

            if "patch_embed" in name:
                d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})
            
            param_groups.append(d)

        elif "decoder" in name:
            if not param.requires_grad:
                continue

            decay_rate = get_transformer_lr_decay_rate(name, lr_decay_rate, decoder_depth)
            d = {"params": param, "lr_multiplier": decay_rate, "wd_multiplier": 1.0, "name": name}

            if name.endswith(".bias") or "norm" in name or "gamma" in name or "logit" in name:
                d.update({"wd_multiplier": 0.0})

            if "patch_embed" in name:
                d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})
            
            param_groups.append(d)
        else:
            if not param.requires_grad:
                continue

            d = {"params": param, "lr_multiplier": 1.0, "wd_multiplier": 1.0, "name": name}

            if name.endswith(".bias") or "norm" in name or "gamma" in name or "logit" in name:
                d.update({"wd_multiplier": 0.0})

            if "patch_embed" in name:
                d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})
            
            param_groups.append(d)

    param_groups = fuse_params_groups(param_groups)
    for g in param_groups:
        g["foreach"] = True

    return list(param_groups)

def make_clip_param_groups(model, image_lr, text_lr, lr_decay_rate=1.0, patch_embed_lr_mult=1.0):
    param_groups = []

    image_depth = len(model.visual.trunk.blocks)
    text_depth = len(model.text.transformer.resblocks)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "visual." in name:
            decay_rate = get_transformer_lr_decay_rate(name, lr_decay_rate, image_depth)
            d = {"params": param, "lr_multiplier": decay_rate, "wd_multiplier": 1.0, 'lr': image_lr, "name": name}

            if name.endswith(".bias") or "norm" in name or "gamma" in name or "logit" in name:
                d.update({"wd_multiplier": 0.0})

            if "patch_embed" in name:
                d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})
            
            param_groups.append(d)

        elif "text." in name:
            decay_rate = get_transformer_lr_decay_rate(name, lr_decay_rate, text_depth)
            d = {"params": param, "lr_multiplier": decay_rate, "wd_multiplier": 1.0, 'lr': text_lr, "name": name}

            if name.endswith(".bias") or "norm" in name or "gamma" in name or "logit" in name:
                d.update({"wd_multiplier": 0.0})

            if "patch_embed" in name:
                d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})
            
            param_groups.append(d)

        else:
            decay_rate = get_transformer_lr_decay_rate(name, lr_decay_rate, image_depth)
            d = {"params": param, "lr_multiplier": decay_rate, "wd_multiplier": 1.0, 'lr': image_lr, "name": name}

            if name.endswith(".bias") or "norm" in name or "gamma" in name or "logit" in name:
                d.update({"wd_multiplier": 0.0})

            if "patch_embed" in name:
                d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})
            
            param_groups.append(d)

    param_groups = fuse_params_groups(param_groups)
    for g in param_groups:
        g["foreach"] = True

    return list(param_groups)