import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import timm
from timm.models.eva import Eva
import numpy as np
import math
import torch
import logging

from open_clip.model import CLIPTextCfg
from open_clip.transformer import TextTransformer
from open_clip.factory import load_checkpoint, _get_hf_config, parse_model_name
from open_clip.timm_model import TimmModel

from timm.layers import LayerNorm, Mlp, GluMlp, SwiGLU, trunc_normal_
from timm.models.eva import EvaBlock

class CustomTextTransformer(TextTransformer):
    def __init__(
            self, 
            context_length = 77, 
            vocab_size = 49408, 
            width = 512, 
            heads = 8, 
            layers = 12, 
            mlp_ratio = 4, 
            ls_init_value = None, 
            output_dim = 512, 
            embed_cls = False, 
            no_causal_mask = False, 
            use_pad_mask = False, 
            correct_cls_mask = False, 
            pad_id = 0, 
            eos_id = 2, 
            pool_type = 'argmax', 
            proj_type = 'linear', 
            proj_bias = False, 
            act_layer = nn.GELU, 
            norm_layer = LayerNorm,
            output_tokens = False, 
            block_type = None, 
            qk_norm = False, 
            scaled_cosine_attn = False, 
            scale_heads = False, 
            scale_attn_inner = False, 
            scale_attn = False, 
            scale_fc = False
        ):
        
        super().__init__(
            context_length, vocab_size, width, heads, 
            layers, mlp_ratio, ls_init_value, output_dim, 
            embed_cls, no_causal_mask, use_pad_mask, correct_cls_mask, 
            pad_id, eos_id, pool_type, proj_type, 
            proj_bias, act_layer, norm_layer, output_tokens, 
            block_type, qk_norm, scaled_cosine_attn, scale_heads, 
            scale_attn_inner, scale_attn, scale_fc

        )        
    
    def forward(self, text):
        x, attn_mask = self._embeds(text)

        x = self.transformer(x, attn_mask=attn_mask)

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
            tokens = x[:, :-1]
        else:
            x = self.ln_final(x)
            pooled = text_global_pool(x, text, pool_type=self.pool_type, eos_token_id=getattr(self, "eos_id", None))
            tokens = x

        if self.output_tokens:
            return pooled, tokens

        return pooled



def text_global_pool(
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        pool_type: str = 'argmax',
        eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    if pool_type == 'first':
        pooled = x[:, 0]
    elif pool_type == 'last':
        pooled = x[:, -1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
    elif pool_type == 'eos':
        # take features from tokenizer specific eos
        assert text is not None
        assert eos_token_id is not None
        idx = (text == eos_token_id).int().argmax(dim=-1)
        pooled = x[torch.arange(x.shape[0], device=x.device), idx]
    elif pool_type == 'avg':
        assert text is not None
        assert eos_token_id is not None
        B, T, D = x.shape
        eos_pos = (text == eos_token_id).int().argmax(dim=-1)  # (B,)

        # eos까지 포함한 마스크 생성
        range_tensor = torch.arange(T, device=text.device).unsqueeze(0)  # (1, T)
        mask = range_tensor <= eos_pos.unsqueeze(1)  # (B, T) eos_token 위치까지 True 나머지 False
        mask_f = mask.float().unsqueeze(-1)  # (B, T, 1) broadcasting False은 부분의 피쳐 버림

        # masked sum과 count로 mean pooling
        summed = (x * mask_f).sum(dim=1)  # (B, D)
        count = mask_f.sum(dim=1)  # (B, 1)
        pooled = summed / count.clamp(min=1.0)
    elif pool_type == 'sty_eos':
        assert text is not None
        assert eos_token_id is not None
        eos_pos = (text == eos_token_id).int().argmax(dim=-1)
        sty_pos = eos_pos - 1
        eos_weight = 2.0
        sty_weight = 1.0
        
        eos_features = x[torch.arange(text.shape[0]), eos_pos]
        sty_features = x[torch.arange(text.shape[0]), sty_pos]

        pooled = (eos_weight * eos_features + sty_weight * sty_features) / (eos_weight + sty_weight)
    else:
        pooled = x

    return pooled

class TimmModel(nn.Module):
    """ compatible dummy for open_clip architecture
    """
    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk

    def forward(self, image):
        return self.trunk(image)
       

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


class CustomCLIP(nn.Module):
    def __init__(
            self,
            model_cfg=None,
            visual = None,
            vision_cfg=None,
            text = None,
            text_cfg=None,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            compute_style:bool = False,
            output_dict: bool = False,
            embed_head_kargs={},
    ):
        super().__init__()
        self.output_dict = output_dict
        self.compute_style = compute_style
        self.visual = visual
        self.text = text
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.eos_id = self.text.eos_id
        self.pool_type = model_cfg['custom_pool_type']

        # embed_head_kargs['dim'] = vision_cfg['num_classes']
        # embed_head_kargs['mlp_ratio'] = vision_cfg['mlp_ratio']
        # embed_head_kargs['swiglu_mlp'] = vision_cfg['swiglu_mlp']
        # embed_head_kargs['scale_mlp'] = vision_cfg['scale_mlp']
        # self.style_embed_head = StyleEmbeddingHead(**embed_head_kargs)


        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

        self.style_logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.style_logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.style_logit_bias = None

        # self.im_sty_head = nn.Linear(visual.trunk.num_classes, visual.trunk.num_classes)
        # self.im_sty_head = nn.Parameter(torch.empty(visual.trunk.num_classes, visual.trunk.num_classes))
        # self.txt_sty_head = nn.Linear(self.text.width, self.text.output_dim)
        # self.txt_sty_head = nn.Parameter(torch.empty(self.text.width, self.text.output_dim))
        # self.init_parameters()

    def init_parameters(self):
        if self.txt_sty_head is not None:
            if isinstance(self.txt_sty_head, nn.Linear):
                nn.init.normal_(self.txt_sty_head.weight, std=self.text.transformer.width ** -0.5)
                if self.txt_sty_head.bias is not None:
                    nn.init.zeros_(self.txt_sty_head.bias)
            else:
                nn.init.normal_(self.txt_sty_head, std=self.text.transformer.width ** -0.5)

        if self.im_sty_head is not None:
            trunc_normal_(self.im_sty_head.weight, std=.02)
            if self.im_sty_head.bias is not None:
                nn.init.zeros_(self.im_sty_head.bias)


    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features
    
    def encode_text(self, text, normalize: bool = False):
        # features = self.text(text)
        # features = F.normalize(features, dim=-1) if normalize else features

        # return features

        tokens_features = self.text(text)
        pooled = text_global_pool(tokens_features, text, self.pool_type, self.eos_id)

        if self.text.text_projection is not None:
            if isinstance(self.text.text_projection, nn.Linear):
                pooled = self.text.text_projection(pooled)
            else:
                pooled = pooled @ self.text.text_projection

        pooled = F.normalize(pooled, dim=-1) if normalize else pooled
        return pooled, tokens_features

    def forward(
            self,
            image: torch.Tensor,
            text: torch.Tensor,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        image_styled, image_content = torch.chunk(image_features, 2, dim=0)
        image_sty_feat = image_styled - image_content
        # image_sty_feat = torch.cat((image_sty_feat, image_content), dim=0)
        # image_sty_feat = self.im_sty_head(image_sty_feat)
        # image_sty_feat = image_sty_feat @ self.im_sty_head
        image_sty_feat = F.normalize(image_sty_feat, dim=-1)
        
        text_features, tokens_features = self.encode_text(text, normalize=True) if text is not None else None
        text_sty_feat = tokens_features[torch.nonzero(text > self.eos_id, as_tuple=True)]
        # text_styled, text_content = torch.chunk(image_features, 2, dim=0)
        # text_sty_feat = self.txt_sty_head(text_sty_feat)
        # text_sty_feat = text_sty_feat @ self.txt_sty_head
        text_sty_feat = F.normalize(text_sty_feat, dim=-1)

        # styled_features, content_features = torch.chunk(image_features, 2, dim=0)
        # if self.compute_style:
        #     style_features = F.normalize((style_features - content_features), dim=-1) 
        #     # style_features = self.style_embed_head(style_features, normalize=True)

        # proj_style_features = F.normalize(self.style_head(style_features), dim=-1)

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "image_sty_feat": image_sty_feat,
                "text_features": text_features,
                "text_sty_feat": text_sty_feat,
                "logit_scale": self.logit_scale.exp(),
                "style_logit_scale": self.style_logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
                out_dict['style_logit_bias'] = self.style_logit_bias

            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


def create_clip(name, model_cfg ={}, vision_cfg = {}, text_cfg = {}, pretrain: str = None, **kwargs):
    if name == "eva02_base_patch14_clip_224":
        visual, vision_cfg = eva02_base_patch14_clip_224(**vision_cfg)
        text, text_cfg = text_encoder(**text_cfg)

    kwargs['vision_cfg'] = vision_cfg
    kwargs['text_cfg'] = text_cfg

    model = CustomCLIP(model_cfg=model_cfg, visual=visual, text=text, init_logit_scale=math.log(20), init_logit_bias=-10, **kwargs)
    if pretrain is not None:
        checkpoint = torch.load(pretrain, weights_only=False)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        missing, unexpected = model.load_state_dict(checkpoint, strict=False)

        logging.info(f"Loaded pretrain model from {pretrain}")
        logging.info(f"missing_key: {len(missing)}: {missing[:10]}")
        logging.info(f"unexpected_key: {len(unexpected)}: {unexpected[:10]}")

    return model

def eva02_base_patch14_clip_224(**kwargs) -> Eva:
    """An EVA-CLIP specific variant that adds additional attn scale layer-norm to eva02_base"""
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
        scale_attn_inner=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
        global_pool=kwargs.pop('global_pool', 'token'),
    )

    model_args.update(kwargs)
    model = Eva(**model_args)
    model = TimmModel(model)

    return model, model_args

def text_encoder(**kwargs):
    model_args = dict(
        context_length=77,
        vocab_size=49408,
        width=512,
        heads=8,
        layers=12,
    )
    model_args.update(kwargs)

    # model = TextTransformer(**model_args)
    model = CustomTextTransformer(**model_args)

    return model, model_args