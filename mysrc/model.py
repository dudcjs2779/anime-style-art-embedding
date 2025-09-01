import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import timm

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
        self.visual = timm.create_model(model_name, **model_kargs)

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
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
            image1: Optional[torch.Tensor] = None,
            image2: Optional[torch.Tensor] = None,
    ):
        image_features1 = self.encode_image(image1, normalize=True) if image1 is not None else None
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
