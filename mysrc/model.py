import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np

import timm
from timm.models.eva import Eva, checkpoint_filter_fn
from timm.models._manipulate import checkpoint
from timm.models._builder import load_pretrained, resolve_pretrained_cfg
from timm.layers import LayerNorm

class CustomEva(Eva):
    def __init__(
            self, img_size = 224, 
            patch_size = 16, 
            in_chans = 3, 
            num_classes = 1000, 
            global_pool = 'avg', 
            embed_dim = 768, 
            depth = 12, 
            num_heads = 12, 
            qkv_bias = True, 
            qkv_fused = True, 
            mlp_ratio = 4, 
            swiglu_mlp = False, 
            scale_mlp = False, 
            scale_attn_inner = False, 
            attn_type = 'eva', 
            drop_rate = 0, 
            pos_drop_rate = 0, 
            patch_drop_rate = 0, 
            proj_drop_rate = 0, 
            attn_drop_rate = 0, 
            drop_path_rate = 0, 
            norm_layer: Callable = LayerNorm,
            init_values = None, 
            class_token = True, 
            num_reg_tokens = 0, 
            no_embed_class = False, 
            use_abs_pos_emb = True, 
            use_rot_pos_emb = False, 
            rope_mixed_mode = False, 
            rope_grid_offset = 0, 
            rope_grid_indexing = 'ij', 
            rope_temperature = 10000, 
            use_post_norm = False, 
            use_pre_transformer_norm = False, 
            use_post_transformer_norm = None, 
            use_fc_norm = None, 
            attn_pool_num_heads = None, 
            attn_pool_mlp_ratio = None, 
            dynamic_img_size = False, 
            dynamic_img_pad = False, 
            ref_feat_shape = None, 
            head_init_scale = 0.001):
        super().__init__(img_size, patch_size, in_chans, num_classes, global_pool, embed_dim, depth, num_heads, qkv_bias, qkv_fused, mlp_ratio, swiglu_mlp, scale_mlp, scale_attn_inner, attn_type, drop_rate, pos_drop_rate, patch_drop_rate, proj_drop_rate, attn_drop_rate, drop_path_rate, norm_layer, init_values, class_token, num_reg_tokens, no_embed_class, use_abs_pos_emb, use_rot_pos_emb, rope_mixed_mode, rope_grid_offset, rope_grid_indexing, rope_temperature, use_post_norm, use_pre_transformer_norm, use_post_transformer_norm, use_fc_norm, attn_pool_num_heads, attn_pool_mlp_ratio, dynamic_img_size, dynamic_img_pad, ref_feat_shape, head_init_scale)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))


    def forward_features(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """Forward pass through feature extraction layers.

        Args:
            x: Input tensor.

        Returns:
            Feature tensor.
        """
        x = self.patch_embed(x)

        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x, rot_pos_embed = self._pos_embed(x)
        x = self.norm_pre(x)

        if getattr(self, 'rope_mixed', False) and rot_pos_embed is not None:
            # Handle depth-dependent embeddings for mixed mode
            # pos embed has shape (depth, num_heads, H*W, dim) or (depth, batch_size, num_heads, H*W, dim)
            for i, blk in enumerate(self.blocks):
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(blk, x, rope=rot_pos_embed[i])
                else:
                    x = blk(x, rope=rot_pos_embed[i])
        else:
            # Standard path for non-mixed mode
            for blk in self.blocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(blk, x, rope=rot_pos_embed)
                else:
                    x = blk(x, rope=rot_pos_embed)

        x = self.norm(x)
        return x
    
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.forward_features(x, mask)
        x = self.forward_head(x)
        return x


class CustomVisual(nn.Module):
    def __init__(
            self,
            model_name: str,
            model_kargs: dict = {},
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = create_eva(model_name, pretrained=True, **model_kargs)
        # self.visual = timm.create_model(model_name, pretrained=True, **model_kargs)

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def encode_image(self, image, mask=None, normalize: bool = False):
        features = self.visual(image, mask)
        return F.normalize(features, dim=-1) if normalize else features
    
    def inference(
            self,
            image: Optional[torch.Tensor] = None,
            normalize: bool = True,
    ):
        image_features = self.encode_image(image, normalize=normalize) if image is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, self.logit_scale.exp(), self.logit_bias
        return image_features, self.logit_scale.exp()
    
    def evaluate(
            self,
            image1: Optional[torch.Tensor] = None,
            image2: Optional[torch.Tensor] = None,
            image3: Optional[torch.Tensor] = None,
    ):
        image_features1 = self.encode_image(image1, normalize=True) if image1 is not None else None
        image_features2 = self.encode_image(image2, normalize=True) if image2 is not None else None
        image_features3 = self.encode_image(image3, normalize=True) if image3 is not None else None

        if self.output_dict:
            out_dict = {
                "image_features1": image_features1,
                "image_features2": image_features2,
                "image_features3": image_features3,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features1, image_features2, image_features3, self.logit_scale.exp(), self.logit_bias
        return image_features1, image_features2, image_features3, self.logit_scale.exp()


    def forward(
            self,
            image1: torch.Tensor,
            image2: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ):
        image_features1 = self.encode_image(image1, mask, normalize=True) if image1 is not None else None
        image_features2 = self.encode_image(image2, normalize=True) if image2 is not None else None

        if self.output_dict:
            out_dict = {
                "image_features1": image_features1,
                "image_features2": image_features2,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features1, image_features2, self.logit_scale.exp(), self.logit_bias
        return image_features1, image_features2, self.logit_scale.exp()


def create_eva(name, pretrained: bool = False, **kwargs):
    name, pretrained_cfg = name.split(".")
    kwargs['pretrained_cfg'] = pretrained_cfg
    if name == "eva02_base_patch14_224":
        model = eva02_base_patch14_224(pretrained, **kwargs)

    return model
    


def eva02_base_patch14_224(pretrained: bool = False, **kwargs) -> CustomEva:
    """EVA02 Base https://arxiv.org/abs/2303.11331"""
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    pretrained_cfg = kwargs.pop("pretrained_cfg")

    model = CustomEva(**dict(model_args, **kwargs))
    variant = "eva02_base_patch14_224"

    pretrained_cfg = resolve_pretrained_cfg(
        variant,
        pretrained_cfg=pretrained_cfg,
        # pretrained_cfg_overlay=pretrained_cfg_overlay
    )
    pretrained_cfg = pretrained_cfg.to_dict()
    
    load_pretrained(
        model,
        pretrained_cfg=pretrained_cfg,
        num_classes=kwargs['num_classes'],
        in_chans=kwargs.get('in_chans', 3),
        filter_fn=checkpoint_filter_fn,
        strict=False,
        # cache_dir=cache_dir,
    )
    return model