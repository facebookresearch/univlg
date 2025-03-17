# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Set, Tuple, Union, Any, Iterable
from types import MethodType
import torch
import torch.nn.functional as F
from torch import nn
import torch
from torch import nn
import einops as E
from univlg.modeling.meta_arch.cross_view_attention import CrossViewPAnet
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
import fvcore.nn.weight_init as weight_init
from detectron2.layers import (
    ShapeSpec,
)
from functools import partial
from contextlib import nullcontext
from dataclasses import dataclass
from typing import NamedTuple


import ipdb
st = ipdb.set_trace



@dataclass
class ModelInfo:
    model_class: str
    model_subtype: str


class InterFeatState(NamedTuple):
    y: torch.Tensor
    alpha: torch.Tensor

class IntermediateFeatureNormalizerBase(nn.Module):
    def forward(self, x: torch.Tensor, index: int, rot_index: int = None, skip: Optional[int] = None) -> InterFeatState:
        raise NotImplementedError()


class NullIntermediateFeatureNormalizer(IntermediateFeatureNormalizerBase):
    instance = None

    def __init__(self, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.register_buffer('alpha', torch.tensor(1, dtype=dtype, device=device))

    @staticmethod
    def get_instance(dtype: torch.dtype, device: torch.device):
        if NullIntermediateFeatureNormalizer.instance is None:
            NullIntermediateFeatureNormalizer.instance = NullIntermediateFeatureNormalizer(dtype, device)
        return NullIntermediateFeatureNormalizer.instance

    def forward(self, x: torch.Tensor, index: int, rot_index: int = None, skip: Optional[int] = None) -> InterFeatState:
        return InterFeatState(x, self.alpha)


from typing import Union, Tuple

import torch
from torch import nn

norm_t = Union[Tuple[float, float, float], torch.Tensor]

def _to_tensor(v: norm_t):
    return torch.as_tensor(v, dtype=torch.float32).view(-1, 1, 1)

class InputConditioner(nn.Module):
    def __init__(self,
                 input_scale: float,
                 norm_mean: norm_t,
                 norm_std: norm_t,
                 dtype: torch.dtype = None,
    ):
        super().__init__()

        self.dtype = dtype

        self.register_buffer("norm_mean", _to_tensor(norm_mean) / input_scale)
        self.register_buffer("norm_std", _to_tensor(norm_std) / input_scale)

    def forward(self, x: torch.Tensor):
        y = (x - self.norm_mean) / self.norm_std
        if self.dtype is not None:
            y = y.to(self.dtype)
        return y


def _take_indices(
        num_blocks: int,
        n: Optional[Union[int, List[int], Tuple[int]]],
) -> Tuple[Set[int], int]:
    if isinstance(n, int):
        assert n >= 0
        take_indices = {x for x in range(num_blocks - n, num_blocks)}
    else:
        take_indices = {num_blocks + idx if idx < 0 else idx for idx in n}
    return take_indices, max(take_indices)


def forward_intermediates(
        model: nn.Module,
        patch_extractor: Callable[[torch.Tensor], torch.Tensor],
        norm: nn.Module,
        num_summary_tokens: int,
        num_cls_tokens: int,
        x: torch.Tensor,
        indices: Optional[Union[int, List[int], Tuple[int]]] = None,
        return_prefix_tokens: bool = False,
        stop_early: bool = False,
        output_fmt: str = 'NCHW',
        intermediates_only: bool = False,
        aggregation: Optional[str] = "sparse",
        inter_feature_normalizer: Optional[Any] = None,
        norm_alpha_scheme = "post-alpha",
        allow_grad_last_n_layers: Optional[int] = None,
) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """ Forward features that returns intermediates.

    The Dense layer aggregation method is inspired from the paper: "Dense Connector for MLLMs"
    by Yao, Huanjin et al. (2024). arXiv preprint arXiv:2405.13800}

    Args:
        x: Input image tensor
        indices: Take last n blocks if int, select matching indices if sequence
        return_prefix_tokens: Return both prefix and spatial intermediate tokens
        norm: Apply norm layer to all intermediates
        stop_early: Stop iterating over blocks when last desired intermediate hit
        output_fmt: Shape of intermediate feature outputs
        intermediates_only: Only return intermediate features
        aggregation: intermediate layer aggregation method (sparse or dense)
        norm_alpha_scheme: apply alpha before ("pre-alpha") or after accumulation ("post-alpha")
    Returns:
    """
    assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
    assert aggregation in ('sparse', 'dense'), 'Aggregation must be one of sparse or dense.'
    reshape = output_fmt == 'NCHW'
    intermediates = []

    blocks = model.blocks

    take_indices, max_index = _take_indices(len(blocks), indices)
    take_indices = sorted(take_indices)
    # forward pass
    B, _, height, width = x.shape

    x = patch_extractor(x)

    if stop_early:
        blocks = blocks[:max_index + 1]

    assert inter_feature_normalizer is None
    inter_feature_normalizer = NullIntermediateFeatureNormalizer.get_instance(x.dtype, x.device)

    assert norm_alpha_scheme in ('none', 'pre-alpha', 'post-alpha'), f'Unsupported alpha scheme: {norm_alpha_scheme}'
    post_alpha_scheme = norm_alpha_scheme == 'post-alpha'

    accumulator = 0
    alpha_sum = 0
    num_accumulated = 0

    take_off = 0

    for i, blk in enumerate(blocks):
        with torch.set_grad_enabled(allow_grad_last_n_layers is not None and i >= len(blocks) - allow_grad_last_n_layers) if allow_grad_last_n_layers is not None else nullcontext():
            x = blk(x)
        if aggregation == "dense":
            # Arbitrarily use the rotation matrix from the final layer in the dense group
            y, alpha = inter_feature_normalizer(x, i, rot_index=take_indices[take_off], skip=num_summary_tokens)
            if post_alpha_scheme:
                accumulator = accumulator + y
                alpha_sum = alpha_sum + alpha
            else:
                accumulator = accumulator + (alpha * y)
                alpha_sum += 1
            num_accumulated += 1
        if i == take_indices[take_off]:
            if aggregation == "dense":
                alpha = alpha_sum / num_accumulated
                x_ = alpha * accumulator / num_accumulated
                num_accumulated = 0
                accumulator = 0
                alpha_sum = 0
            else:
                 y, alpha = inter_feature_normalizer(x, i, skip=num_summary_tokens)
                 x_ = alpha * y
            # normalize intermediates with final norm layer if enabled
            intermediates.append(norm(x_))
            take_off = min(take_off + 1, len(take_indices) - 1)

    # process intermediates

    # split prefix (e.g. class, distill) and spatial feature tokens
    prefix_tokens = [y[:, :num_cls_tokens] for y in intermediates]
    intermediates = [y[:, num_summary_tokens:] for y in intermediates]

    if reshape:
        # reshape to BCHW output format
        H = height // model.patch_size
        W = width // model.patch_size
        intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
    if not torch.jit.is_scripting() and return_prefix_tokens:
        # return_prefix not support in torchscript due to poor type handling
        intermediates = list(zip(prefix_tokens, intermediates))
    if intermediates_only:
        return intermediates
    x = norm(x)
    return x, intermediates


class DinoWrapper(nn.Module):
    def __init__(self, dino_model: nn.Module):
        super().__init__()
        self.inner = dino_model
        dino_model.blocks = nn.Sequential(*dino_model.blocks)

    @property
    def embed_dim(self):
        return self.inner.embed_dim

    @property
    def vision_encoder(self):
        return self.inner

    @property
    def patch_size(self):
        return self.inner.patch_size

    @property
    def num_cls_tokens(self):
        return getattr(self.inner, 'num_tokens', 1)

    @property
    def num_registers(self):
        return getattr(self.inner, 'num_register_tokens', 0)

    @property
    def num_summary_tokens(self):
        return self.num_cls_tokens + self.num_registers

    @property
    def blocks(self):
        return self.inner.blocks

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        parts = self.inner.forward_features(*args, **kwargs)

        cls_token = parts['x_norm_clstoken']
        features = parts['x_norm_patchtokens']

        return cls_token, features

    def forward_features(self, x: torch.Tensor):
        x = self.inner.prepare_tokens_with_masks(x)
        x = self.inner.blocks(x)
        x_norm = self.inner.norm(x)

        return x_norm[:, 0], x_norm[:, self.num_summary_tokens:]

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.prepare_tokens_with_masks(x)

    def forward_intermediates(self,
        x: torch.Tensor,
        norm: bool = False,
        **kwargs,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        return forward_intermediates(
            self,
            patch_extractor=self.inner.prepare_tokens_with_masks,
            num_summary_tokens=self.num_summary_tokens,
            num_cls_tokens=self.num_cls_tokens,
            norm=self.inner.norm if norm else lambda y: y,
            x=x,
            **kwargs,
        )

def load_model(version: str, device: torch.device = None, **kwargs):
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model = torch.hub.load('facebookresearch/dinov2', version, pretrained=True, force_reload=False, **kwargs)
    model = DinoWrapper(model)

    preprocessor = InputConditioner(1.0, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    info = ModelInfo(model_class='DINOv2', model_subtype=version.replace('dinov2_', ''))

    if device is not None:
        model.to(device=device)

    return model, preprocessor, info


def auto_batch(batch_size, fn, x):
    if x.shape[0] < batch_size:
        return fn(x)
    else:
        outputs = []
        for i in x.split(batch_size):
            outputs.append(fn(i))

        if isinstance(outputs[0], list):
            return [torch.cat([o[i] for o in outputs]) for i in range(len(outputs[0]))]
        else:
            return torch.cat(outputs)



def center_padding(images, patch_size):
    _, _, h, w = images.shape
    diff_h = h % patch_size
    diff_w = w % patch_size

    if diff_h == 0 and diff_w == 0:
        return images

    pad_h = patch_size - diff_h
    pad_w = patch_size - diff_w

    pad_t = pad_h // 2
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_b = pad_h - pad_t

    images = F.pad(images, (pad_l, pad_r, pad_t, pad_b))
    return images


def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
    if output_type == "cls":
        assert cls_token is not None
        output = cls_token
    elif output_type == "gap":
        output = dense_tokens.mean(dim=1)
    elif output_type == "dense":
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        output = dense_tokens.contiguous()
    elif output_type == "dense-cls":
        assert cls_token is not None
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        cls_token = cls_token[:, :, None, None].repeat(1, 1, h, w)
        output = torch.cat((dense_tokens, cls_token), dim=1).contiguous()
    else:
        raise ValueError()

    return output

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

@BACKBONE_REGISTRY.register()
class DINOv2(Backbone):
    """
    Backbone definition for the DINOv2 model with optional special 3D layers.

    Args:
        cfg (CfgNode): Configuration options.
    """

    def __init__(
        self,
        cfg,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Get model from TorchHub.
        version = cfg.DINO_VERSION
        output = "dense"
        
        freeze_vit = True

        self.dtype = torch.bfloat16
        self.version = version
        self.checkpoint_name = f"{version}"
        self.unfreeze_layers = cfg.DINO_UNFREEZE_LAYERS
        self.cfg = cfg

        assert cfg.MODEL.FREEZE_BACKBONE == False
        assert cfg.USE_GENERIC_DINO

        self.intermediate_aggregation = "sparse"
        self.dinov2, self.preprocessor, self.info = load_model(cfg.DINO_VERSION)
        assert freeze_vit, "Not actually necessary, but untested."
        if isinstance(self.preprocessor, nn.Module):
            self.preprocessor.eval()

        patch_size = self.dinov2.patch_size # For some models you may have to statically define this.
        self.dinov2 = self.dinov2.to(self.dtype)
        self.dinov2.requires_grad_(not freeze_vit)
        if "dino" in self.version.lower():
            _model = self.dinov2.inner
        else:
            _model = self.dinov2.model

        if cfg.DINO_UNFREEZE_LAYERS is not None and len(cfg.DINO_UNFREEZE_LAYERS) > 0:
            for i, block in enumerate(_model.blocks):
                if i in self.unfreeze_layers:
                    print(f"Unfreezing layer {i}")
                    block.requires_grad_(True)
        
        if freeze_vit: self.dinov2.eval()
        self.output = output
        self.patch_size = patch_size
        self.cfg = cfg
        
        feat_dim = _model.embed_dim
        num_layers = len(_model.blocks)
        self.multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        print(f"DINOv2 has {num_layers} layers with 3D Layers: {self.multilayers}")

        # Define layer name (for logging).
        self.layer = "-".join(str(_x) for _x in self.multilayers)

        # Optional special 3D layers
        if cfg is not None and not cfg.DISABLE_VIT_PANET and cfg.MODEL.CROSS_VIEW_BACKBONE and (
            cfg.MODEL.DECODER_3D or cfg.PASS_2D_CROSS_VIEW
        ):
            # Starting index to apply the special 3D layers
            num_layers = len(_model.blocks)
            num_cross_attn_layers = len(cfg.MODEL.CROSS_VIEW_NUM_LAYERS)
            assert len(self.multilayers) == num_cross_attn_layers
            cross_attn_indices = self.multilayers
            conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
            cross_attn_indices = list(range(num_cross_attn_layers))

            if cfg.MODEL.CROSS_VIEW_CONTEXTUALIZE:
                self.cross_view_attn = nn.ModuleDict(
                    {
                        str(idx): CrossViewPAnet(
                            latent_dim=conv_dim,
                            nsample=cfg.MODEL.KNN,
                            dropout=cfg.MODEL.MASK_FORMER.DROPOUT,
                            num_layers=cfg.MODEL.CROSS_VIEW_NUM_LAYERS[i],
                            cfg=cfg,
                        )
                        for i, idx in enumerate(cross_attn_indices)
                    }
                )

                # Project the ViT features to the "conv_dim" dimension
                if cfg.PANET:
                    self.token_to_trans = nn.ModuleDict(
                        {
                            str(idx): nn.Sequential(
                                nn.Conv2d(feat_dim, conv_dim, kernel_size=1),
                                nn.GroupNorm(32, conv_dim),
                            )
                            for idx in cross_attn_indices
                        }
                    )
                    # xavier initilization
                    for layer in self.token_to_trans.values():
                        weight_init.c2_xavier_fill(layer[0])
                else:
                    self.token_to_trans = nn.ModuleDict(
                        {
                            str(idx): nn.Sequential(
                                nn.Linear(feat_dim, conv_dim),
                                nn.LayerNorm(conv_dim),
                            )
                            for idx in cross_attn_indices
                        }
                    )

                # Project conv_dim back to ViT feature dimension
                
                # Ayush: zero residual is suspicious, group norm might be better
                # group norm should be for both token to trans and trans to tokens
                # cross_attn_indices is suspicious.
                if cfg.PANET:
                    self.trans_to_token = nn.ModuleDict(
                        {
                            str(idx): nn.Sequential(
                                nn.Conv2d(conv_dim, feat_dim, kernel_size=1),
                                nn.GroupNorm(32, feat_dim),
                            )
                            for idx in cross_attn_indices
                        }
                    )
                    # xavier initilization
                    for layer in self.trans_to_token.values():
                        weight_init.c2_xavier_fill(layer[0])
                else:
                    self.trans_to_token = nn.ModuleDict(
                        {
                            str(idx): nn.Sequential(
                                nn.Conv2d(conv_dim, feat_dim * (1 if cfg.NON_ZERO_DINO_RESIDUAL else 2), kernel_size=1),
                                nn.GroupNorm(32, feat_dim * (1 if cfg.NON_ZERO_DINO_RESIDUAL else 2)),
                            ) if cfg.DINO_GROUP_NORM else nn.Sequential(
                                nn.SiLU(),
                                nn.LayerNorm(conv_dim),
                                nn.Linear(conv_dim, feat_dim * (1 if cfg.NON_ZERO_DINO_RESIDUAL else 2)),
                            )
                            for idx in cross_attn_indices
                        }
                    )

                    # if cfg.DINO_GROUP_NORM and not cfg.ENABLE_LEGACY_ENCODER_LR_BUG:
                    #     import fvcore.nn.weight_init as weight_init
                    #     weight_init.c2_xavier_fill(self.token_to_trans[0])
                    #     weight_init.c2_xavier_fill(self.trans_to_token[0])

        self._out_features = cfg.MODEL.SWIN.OUT_FEATURES
        self._out_feature_strides = {
            "res2": 1,
            "res3": 1,
            "res4": 1,
            "res5": 1,
        }
        
        # Ayush: weird -- potentially we need to have output dim as 256 i.e. conv_dim
        self._out_feature_channels = {
            "res2": feat_dim,
            "res3": feat_dim,
            "res4": feat_dim,
            "res5": feat_dim,
        }

        # zero residual initialization is weird
        if not self.cfg.DISABLE_VIT_PANET and not self.cfg.NON_ZERO_DINO_RESIDUAL and cfg.MODEL.CROSS_VIEW_CONTEXTUALIZE and (
            cfg.MODEL.DECODER_3D or cfg.PASS_2D_CROSS_VIEW
        ) and not cfg.PANET:
            print("Initializing DINOv2 residual to zero")
            assert not cfg.DINO_GROUP_NORM
            for block in self.trans_to_token.values():
                nn.init.constant_(block[-1].weight, 0)
                nn.init.constant_(block[-1].bias, 0)

    def forward(self, x, x_xyz=None, x_p2v=None, shape=None, decoder_3d=False):
        assert x.min() >= -1e-3 and x.max() <= 1 + 1e-3 and x.ndim == 4
        with torch.no_grad() if self.cfg.DINO_GENERIC_DISABLE_INFERENCE_MODE else torch.inference_mode():
            with torch.autocast(x.device.type, dtype=self.dtype):
                p_images = self.preprocessor(x)
                _fn = partial(
                    self.dinov2.forward_intermediates,
                    indices=self.multilayers,
                    return_prefix_tokens=False,
                    norm=True,
                    stop_early=True,
                    output_fmt='NCHW',
                    intermediates_only=True,
                    aggregation=self.intermediate_aggregation,
                    allow_grad_last_n_layers=len(self.cfg.DINO_UNFREEZE_LAYERS) if self.cfg.DINO_UNFREEZE_LAYERS is not None else None,
                )
                if not self.training and self.cfg.DINO_EVAL_BATCH:
                    features = auto_batch(self.cfg.DINO_EVAL_BATCH_SIZE, _fn, p_images)
                else:
                    features = _fn(p_images)

        features = [x.clone() for x in features]

        if self.cfg is not None and not self.cfg.DISABLE_VIT_PANET and (
            (decoder_3d or self.cfg.PASS_2D_CROSS_VIEW)
            and self.cfg.MODEL.CROSS_VIEW_CONTEXTUALIZE
            and self.cfg.MODEL.CROSS_VIEW_BACKBONE
        ):
            all_feat_3d = []
            num_skip = 0
            for i in range(len(self.multilayers)):
                # Project x
                if self.cfg.PANET:
                    x = features[i]
                else:
                    x = features[i].permute(0, 2, 3, 1)
                x2 = self.token_to_trans[str(i)](x)
                
                if not self.cfg.PANET:
                    x2 = x2.permute(0, 3, 1, 2)

                # Cross-view attention
                if self.cfg.FORCE_VIT_XYZ_SCALE: index = 1
                mv_data = {}
                mv_data["multi_scale_p2v"] = [x_p2v[index]]

                x2 = self.cross_view_attn[str(i)](
                    feature_list=[x2],
                    xyz_list=[x_xyz[index]],
                    shape=shape[:2],
                    multiview_data=mv_data,
                    voxelize=self.cfg.INPUT.VOXELIZE,
                )[0]

                x2 = E.rearrange(x2, "b c h w -> b (h w) c")

                if self.cfg.NON_ZERO_DINO_RESIDUAL and not self.cfg.PANET:
                    if self.cfg.DINO_GROUP_NORM: x2 = x2.permute(0, 2, 1)
                    x2 = self.trans_to_token[str(i)](x2)
                    if self.cfg.DINO_GROUP_NORM: x2 = x2.permute(0, 2, 1)
                    x = torch.cat([
                        x[:, :num_skip],
                        x[:, num_skip:] + x2,
                    ], dim=1)
                elif self.cfg.PANET:
                    x2 = x2.permute(0, 2, 1)
                    x2 = self.trans_to_token[str(i)](x2[..., None])[..., 0]
                    x2 = E.rearrange(x2, "b c (h w) -> b c h w", h=x.shape[2], w=x.shape[3])
                    x = x + x2
                else:
                    # Project back
                    if self.cfg.DINO_GROUP_NORM:
                        x2 = x2.permute(0, 2, 1)[..., None]
                        x2 = self.trans_to_token[str(i)](x2)
                        x2 = x2[..., 0].permute(0, 2, 1)
                        x2_scale, x2_shift = x2.chunk(2, dim=-1)
                    else:
                        x2_scale, x2_shift = self.trans_to_token[str(i)](x2).chunk(2, dim=-1)

                    x2_shift = E.rearrange(x2_shift, "b (h w) c -> b c h w", h=x.shape[1], w=x.shape[2])
                    x2_scale = E.rearrange(x2_scale, "b (h w) c -> b c h w", h=x.shape[1], w=x.shape[2])
                    x = E.rearrange(x, "b h w c -> b c h w")

                    # Skip connection
                    x = modulate(x, x2_shift, x2_scale)
                
                index += 1

                all_feat_3d.append(x)

            outputs = all_feat_3d
            if self.cfg.FIX_PANET_BUG:
                features = all_feat_3d
        else:
            outputs = features
    
        keys = list(self._out_feature_strides.keys())
        assert len(features) == len(keys)
        outputs = {}
        for i, key in enumerate(keys):
            outputs[key] = features[i]

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }