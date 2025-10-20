import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np

import timm
from timm.models.eva import Eva, EvaBlock, checkpoint_filter_fn
from timm.models.vision_transformer import Block
from timm.models._manipulate import checkpoint
from timm.models._builder import load_pretrained, resolve_pretrained_cfg
from timm.layers import LayerNorm, Mlp, GluMlp, SwiGLU, resample_abs_pos_embed


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
        nn.init.trunc_normal_(self.mask_token, std=0.02)
    
    def forward_features(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """Forward pass through feature extraction layers.

        Args:
            x: Input tensor.

        Returns:
            Feature tensor.
        """
        x = self.patch_embed(x) # NCHW -> NLC, Dynamic_image_size => NCHW -> NHWC

        if mask is not None:
            mask = mask.reshape(*x.shape[:-1])
            x = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x, rot_pos_embed = self._pos_embed(x)
        self.rot = rot_pos_embed
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

        # if not self.global_pool: # global_pool을 cls_token 방식으로 처리할 예정 
        #     return {"x_clstoken": x[:, 0], "x_patchtokens": x[:, self.num_prefix_tokens:]}
        
        return x

class StyleEmbeddingHead(nn.Module):
    def __init__(
            self, 
            dim,
            mlp_ratio,
            depth=1,
            swiglu_mlp=False,
            scale_mlp=False,
            proj_drop: float = 0,
            norm_layer: Callable = LayerNorm,
            act_layer: Callable = nn.GELU,
            **kwargs,
        ):
        super().__init__()
        hidden_features = int(dim * mlp_ratio)

        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlps = nn.ModuleList([
                    SwiGLU(
                        in_features=dim,
                        hidden_features=hidden_features,
                        norm_layer=norm_layer if scale_mlp else None,
                        drop=proj_drop,
                    )
                    for i in range(depth)])
                
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP

                self.mlps = nn.ModuleList([
                    GluMlp(
                        in_features=dim,
                        hidden_features=hidden_features * 2,
                        norm_layer=norm_layer if scale_mlp else None,
                        act_layer=nn.SiLU,
                        gate_last=False,
                        drop=proj_drop,
                    )
                    for i in range(depth)])
        else:
            self.mlps = nn.ModuleList([
                Mlp(
                    in_features=dim,
                    hidden_features=hidden_features,
                    act_layer=act_layer,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
                for i in range(depth)])
            
        self.norm = norm_layer(dim)

        if isinstance(self.norm, nn.LayerNorm):
            nn.init.constant_(self.norm.bias, 0)
            nn.init.constant_(self.norm.weight, 1.0)
            
    def forward(self, x, normalize=True):
        for mlp in self.mlps:
            x = mlp(x)
        x = self.norm(x)

        return F.normalize(x, dim=-1) if normalize else x
    
class CustomVisual(nn.Module):
    def __init__(
            self,
            model_name: str,
            vision_kargs: dict = {},
            style_embed_head_kargs= {},
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            compute_style=False,
            output_dict: bool = False,
    ):
        super().__init__()
        self.compute_style = compute_style
        self.output_dict = output_dict
        self.visual, visual_args = create_eva(model_name, pretrained=True, **vision_kargs)

        style_embed_head_kargs['dim'] = visual_args['embed_dim']
        style_embed_head_kargs['mlp_ratio'] = visual_args['mlp_ratio']
        style_embed_head_kargs['swiglu_mlp'] = visual_args['swiglu_mlp']
        style_embed_head_kargs['scale_mlp'] = visual_args['scale_mlp']
        self.style_embed_head = StyleEmbeddingHead(**style_embed_head_kargs)

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
            mask1: Optional[torch.Tensor] = None,
            mask2: Optional[torch.Tensor] = None,
    ):
        im_feat1 = self.encode_image(image1, mask1, normalize=True) if image1 is not None else None
        im_feat2 = self.encode_image(image2, mask2, normalize=True) if image2 is not None else None
        im_feat3 = self.encode_image(image3, normalize=True) if image3 is not None else None

        styled_features1, content_features1 = torch.chunk(im_feat1[:, 0], 2, dim=0)
        styled_features2, content_features2 = torch.chunk(im_feat2[:, 0], 2, dim=0)
        styled_features3, content_features3 = torch.chunk(im_feat3[:, 0], 2, dim=0)

        if self.compute_style:
            style_features1 = self.style_embed_head((styled_features1 - content_features1), normalize=True)
            style_features2 = self.style_embed_head((styled_features2 - content_features2), normalize=True)
            style_features3 = self.style_embed_head((styled_features3 - content_features3), normalize=True)

        if self.output_dict:
            out_dict = {
                "styled_features1": styled_features1,
                "styled_features2": styled_features2,
                "styled_features3": styled_features3,
                "content_features1": content_features1,
                "content_features2": content_features2,
                "content_features3": content_features3,
                "logit_scale": self.logit_scale.exp(),
                # "im_feat1_patches": im_feat1[:, self.visual.num_prefix_tokens:],
                # "im_feat2_patches": im_feat2[:, self.visual.num_prefix_tokens:],
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            if self.compute_style:
                out_dict['style_features1'] = style_features1
                out_dict['style_features2'] = style_features2
                out_dict['style_features3'] = style_features3
            return out_dict

        if self.logit_bias is not None:
            return im_feat1, im_feat2, im_feat3, self.logit_scale.exp(), self.logit_bias
        return im_feat1, im_feat2, im_feat3, self.logit_scale.exp()
    
    def forward(
            self,
            image1: torch.Tensor,
            image2: torch.Tensor,
            mask1: Optional[torch.Tensor] = None,
            mask2: Optional[torch.Tensor] = None,
    ):
        im_feat1 = self.encode_image(image1, mask1, normalize=True) if image1 is not None else None
        im_feat2 = self.encode_image(image2, mask2, normalize=True) if image2 is not None else None

        styled_features1, content_features1 = torch.chunk(im_feat1[:, 0], 2, dim=0)
        styled_features2, content_features2 = torch.chunk(im_feat2[:, 0], 2, dim=0)

        if self.compute_style:
            style_features1 = self.style_embed_head((styled_features1 - content_features1), normalize=True)
            style_features2 = self.style_embed_head((styled_features2 - content_features2), normalize=True)

        # proj_style_features = F.normalize(self.style_head(style_features), dim=-1)
     
        if self.output_dict:
            out_dict = {
                "styled_features1": styled_features1,
                "styled_features2": styled_features2,
                "content_features1": content_features1,
                "content_features2": content_features2,
                "logit_scale": self.logit_scale.exp(),
                # "im_feat1_patches": im_feat1[:, self.visual.num_prefix_tokens:],
                # "im_feat2_patches": im_feat2[:, self.visual.num_prefix_tokens:],
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            if self.compute_style:
                out_dict['style_features1'] = style_features1
                out_dict['style_features2'] = style_features2
            return out_dict

        if self.logit_bias is not None:
            return im_feat1, im_feat2, self.logit_scale.exp(), self.logit_bias
        return im_feat1, im_feat2, self.logit_scale.exp()


def create_eva(name, pretrained: bool = False, **kwargs):
    name, pretrained_cfg = name.split(".")
    kwargs['pretrained_cfg'] = pretrained_cfg
    if name == "eva02_base_patch14_224":
        model, model_args = eva02_base_patch14_224(pretrained, **kwargs)

    return model, model_args
    
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

    model_args.update(kwargs)
    model = CustomEva(**model_args)
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
    return model, model_args

def create_decoder(name, kwargs):
    model = globals()[name](kwargs)

    return model

def eva_decoder(kwargs):
    model_args = dict(
        decoder_dim=768,
        depth=8,
        num_heads=12,
        qkv_fused=False,
        qkv_bias = True,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
    )

    model_args.update(kwargs)
    model = EvaDecoder(**model_args)
    return model