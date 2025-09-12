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
from timm.layers import LayerNorm, Mlp, GluMlp, SwiGLU
from timm.models.layers import trunc_normal_

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

        # if not self.global_pool: # global_pool을 cls_token 방식으로 처리할 예정 
        #     return {"x_clstoken": x[:, 0], "x_patchtokens": x[:, self.num_prefix_tokens:]}
        
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
        self.visual, visual_args = create_eva(model_name, pretrained=True, **model_kargs)
        # self.visual = timm.create_model(model_name, pretrained=True, **model_kargs)

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def encode_image(self, image, mask=None, normalize: bool = False):
        features = self.visual(image, mask)

        if isinstance(features, dict):
            if normalize:
                features['x_clstoken'] = F.normalize(features['x_clstoken'], dim=-1)
                features['x_patchtokens'] = F.normalize(features['x_patchtokens'], dim=-1)
                return features
            
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
        im_feat1 = self.encode_image(image1, normalize=True) if image1 is not None else None
        im_feat1_masked = self.encode_image(image1, mask1, normalize=True) if image1 is not None else None
        im_feat2 = self.encode_image(image2, normalize=True) if image2 is not None else None
        im_feat2_masked = self.encode_image(image2, mask2, normalize=True) if image2 is not None else None
        im_feat3 = self.encode_image(image3, normalize=True) if image3 is not None else None

        if self.output_dict:
            out_dict = {
                "im_feat1": im_feat1,
                "im_feat1_masked": im_feat1_masked,
                "im_feat2": im_feat2,
                "im_feat2_masked": im_feat2_masked,
                "im_feat3": im_feat3,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return im_feat1, im_feat1_masked, im_feat2, im_feat2_masked, im_feat3, self.logit_scale.exp(), self.logit_bias
        return im_feat1, im_feat1_masked, im_feat2, im_feat2_masked, im_feat3, self.logit_scale.exp()

    def forward(
            self,
            image1: torch.Tensor,
            image2: torch.Tensor,
            mask1: Optional[torch.Tensor] = None,
            mask2: Optional[torch.Tensor] = None,
    ):
        im_feat1 = self.encode_image(image1, normalize=True) if image1 is not None else None
        im_feat1_masked = self.encode_image(image1, mask1, normalize=True) if image1 is not None else None
        im_feat2 = self.encode_image(image2, normalize=True) if image2 is not None else None
        im_feat2_masked = self.encode_image(image2, mask2, normalize=True) if image2 is not None else None

        if self.output_dict:
            out_dict = {
                "im_feat1": im_feat1,
                "im_feat1_masked": im_feat1_masked,
                "im_feat2": im_feat2,
                "im_feat2_masked": im_feat2_masked,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return im_feat1, im_feat1_masked, im_feat2, im_feat2_masked, self.logit_scale.exp(), self.logit_bias
        return im_feat1, im_feat1_masked, im_feat2, im_feat2_masked, self.logit_scale.exp()


class MAEDecoder(nn.Module):
    def __init__(
            self,
            in_chans,
            patch_size,
            num_patches,
            num_prefix_tokens,
            encoder_dim, 
            decoder_dim=512, 
            depth=8, 
            num_heads=16, 
            mlp_ratio=4., 
            norm_layer=nn.LayerNorm, 
        ):
        super().__init__()
        self.patch_size = patch_size
        self.num_prefix_tokens = num_prefix_tokens
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + num_prefix_tokens, decoder_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.decoder_norm = norm_layer(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # nn.Linear, nn.LayerNorm 초기화는 여기서
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3) => patch_size**2 *3: 이 사이즈는 dim=768이라는 게 아니라 패치 크기에 x 채널을 한 것.
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    # 코드 확인 필요!!!
    def forward(self, x, mask, mask_indices_list, none_mask_indices_list):
        B, N, D = x.shape   # B N D

        x_keep = torch.index_select(x[:, self.num_prefix_tokens:, :].flatten(0, 1), dim=0, index=none_mask_indices_list)   # B N D => BN' D
        x_keep = torch.cat((x[:, :self.num_prefix_tokens, :].flatten(0, 1), x_keep), dim=0)
        x_keep = self.decoder_embed(x_keep) # BN' D'

        # 패치들의 원래 위치 복원 및 마스킹
        x_ordered = torch.zeros((B * N, x_keep.shape[-1]), device=x_keep.device, dtype=x_keep.dtype) # BN D'
        x_ordered[none_mask_indices_list] = x_keep[B * self.num_prefix_tokens:, :]
        x_ordered[mask_indices_list] = self.mask_token.to(dtype=x_keep.dtype)
        x_ordered = torch.cat((x_keep[:B * self.num_prefix_tokens, :], x_ordered), dim=0)

        x = x_ordered.view(B, N, -1)    # BN D' => B N D'
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        return x


class EmbeddingHead(nn.Module):
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
            
    def forward(self, x):
        for mlp in self.mlps:
            x = mlp(x)
        x = self.norm(x)
        return x


class CustomMaeVisual(nn.Module):
    def __init__(
            self,
            model_name: str,
            model_kargs: dict = {},
            decoder_kargs: dict = {},
            embed_head_kargs: dict = {},
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual, visual_args = create_eva(model_name, pretrained=True, **model_kargs)

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

        decoder_kargs['img_size'] = visual_args['img_size']
        decoder_kargs['in_chans'] = self.visual.patch_embed.proj.in_channels
        decoder_kargs['patch_size'] = visual_args['patch_size']
        decoder_kargs['encoder_dim'] = visual_args['embed_dim']
        decoder_kargs['num_patches'] = self.visual.patch_embed.num_patches
        decoder_kargs['num_prefix_tokens'] = self.visual.num_prefix_tokens
        self.mae_decoder = mae_vit_base_decoder(**decoder_kargs)

        embed_head_kargs['dim'] = visual_args['embed_dim']
        embed_head_kargs['mlp_ratio'] = visual_args['mlp_ratio']
        embed_head_kargs['swiglu_mlp'] = visual_args['swiglu_mlp']
        embed_head_kargs['scale_mlp'] = visual_args['scale_mlp']
        self.embed_head = EmbeddingHead(**embed_head_kargs)

        

    def encode_image(self, image, mask=None, normalize: bool = False):
        features = self.visual(image, mask)

        if isinstance(features, dict):
            if normalize:
                features['x_clstoken'] = F.normalize(features['x_clstoken'], dim=-1)
                features['x_patchtokens'] = F.normalize(features['x_patchtokens'], dim=-1)
                return features
            
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
            mask_indices_list = None,
            none_mask_indices_list = None,
    ):
        im_feat1 = self.encode_image(image1, normalize=True) if image1 is not None else None
        im_feat2 = self.encode_image(image2, normalize=True) if image2 is not None else None
        im_feat3 = self.encode_image(image3, normalize=True) if image3 is not None else None

        cat_patches = torch.concat((im_feat1[:, self.visual.num_prefix_tokens:], im_feat2[:, self.visual.num_prefix_tokens:]), dim=0)
        cat_mask = torch.concat((mask1, mask2), dim=0)

        if mask_indices_list is not None:
            pred = self.mae_decoder(cat_patches, cat_mask, mask_indices_list, none_mask_indices_list)

        im_feat1 = self.embed_head(im_feat1)
        im_feat2 = self.embed_head(im_feat2)
        im_feat3 = self.embed_head(im_feat3)

        if self.output_dict:
            out_dict = {
                "im_feat1_cls": im_feat1[:, 0],
                "im_feat1_patches": im_feat1[:, self.visual.num_prefix_tokens:],
                "im_feat2_cls": im_feat2[:, 0],
                "im_feat2_patches": im_feat2[:, self.visual.num_prefix_tokens:],
                "im_feat3": im_feat3[:, 0],
                "logit_scale": self.logit_scale.exp(),
            }

            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias

            if mask_indices_list is not None:
                out_dict['pred'] = pred
                out_dict['mask'] = cat_mask

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
            mask_indices_list = None,
            none_mask_indices_list = None,
    ):
        im_feat1 = self.encode_image(image1, mask1, normalize=True) if image1 is not None else None
        im_feat2 = self.encode_image(image2, mask2, normalize=True) if image2 is not None else None
        # im_feat1 = self.encode_image(image1, normalize=True) if image1 is not None else None
        # im_feat2 = self.encode_image(image2, normalize=True) if image2 is not None else None
        # im_feat1_masked = self.encode_image(image1, mask1, normalize=True) if image1 is not None else None
        # im_feat2_masked = self.encode_image(image2, mask2, normalize=True) if image2 is not None else None

        # cat_patches = torch.concat((im_feat1[:, self.visual.num_prefix_tokens:], im_feat2[:, self.visual.num_prefix_tokens:]), dim=0)
        # cat_masked_patches = torch.concat((im_feat1_masked[:, self.visual.num_prefix_tokens:], im_feat2_masked[:, self.visual.num_prefix_tokens:]), dim=0)
        
        # if mask_indices_list is not None:
        #     label_patches = torch.index_select(cat_patches.flatten(0, 1), dim=0, index=mask_indices_list)
        #     masked_patches = torch.index_select(cat_masked_patches.flatten(0, 1), dim=0, index=mask_indices_list)

        #     _label_patches = torch.index_select(cat_patches.flatten(0, 1), dim=0, index=none_mask_indices_list)
        #     _masked_patches = torch.index_select(cat_masked_patches.flatten(0, 1), dim=0, index=none_mask_indices_list)
        #     masked_patches @ label_patches.T
        #     _masked_patches @ _label_patches.T

        cat_feates = torch.concat((im_feat1, im_feat2), dim=0)
        cat_mask = torch.concat((mask1, mask2), dim=0)

        if mask_indices_list is not None:
            pred = self.mae_decoder(cat_feates, cat_mask, mask_indices_list, none_mask_indices_list)

        im_feat1 = self.embed_head(im_feat1)
        im_feat2 = self.embed_head(im_feat2)

        if self.output_dict:
            out_dict = {
                "im_feat1_cls": im_feat1[:, 0],
                "im_feat1_patches": im_feat1[:, self.visual.num_prefix_tokens:],
                "im_feat2_cls": im_feat2[:, 0],
                "im_feat2_patches": im_feat2[:, self.visual.num_prefix_tokens:],
                "logit_scale": self.logit_scale.exp(),
            }

            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias

            if mask_indices_list is not None:
                out_dict['pred'] = pred
                out_dict['mask'] = cat_mask

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


def mae_vit_base_decoder(**kwargs):

    model_args = dict(
        decoder_dim=512,
        num_heads=16,
        mlp_ratio=4.,
        depth=8,
    )
    model_args.update(kwargs)
    model = MAEDecoder(**model_args)

    checkpoint = torch.load("pretrain/mae_pretrain_vit_base.pth")
    
    return model