# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from detectron2.modeling import build_backbone
from detectron2.layers import ShapeSpec

from univlg.modeling.backbone.resnet import build_resnet_backbone_custom
from univlg.modeling.pixel_decoder.msdeformattn import build_pixel_decoder


import ipdb
st = ipdb.set_trace

class UniVLGVisualBackbone(nn.Module):
    def __init__(
        self,
        cfg,
        freeze_backbone=False
    ):
        super().__init__()
        
        # initialize main backbone
        if cfg.MODEL.BACKBONE.NAME == "build_resnet_backbone":
            backbone = build_resnet_backbone_custom(
                cfg, ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
            )
        else:
            backbone = build_backbone(cfg)
        
        if freeze_backbone:
            panet_resnet_layers = ["cross_view_attn", "res_to_trans", "trans_to_res"]
            panet_swin_layers = [
                "cross_view_attn",
                "cross_layer_norm",
                "res_to_trans",
                "trans_to_res",
            ]

            if cfg.MODEL.BACKBONE.NAME == "build_resnet_backbone":
                backbone_panet_layers = panet_resnet_layers
            elif cfg.MODEL.BACKBONE.NAME == "D2SwinTransformer":
                backbone_panet_layers = panet_swin_layers
            else:
                raise NotImplementedError

            for name, param in backbone.named_parameters():
                if any([layer in name for layer in backbone_panet_layers]):
                    print(f"Not freezing {name}")
                    continue
                else:
                    param.requires_grad = False
                    
        self.backbone = backbone
        
        # initialize UNet (Pixel Decoder)
        self.pixel_decoder = build_pixel_decoder(cfg, backbone.output_shape())
        
    
    def forward(
        self, images, multi_scale_xyz, multi_scale_p2v, shape, decoder_3d, actual_decoder_3d,
        mesh_pc=None, mesh_p2v=None):
        """Visual Backbone
        
        Keyword arguments:
            - images: (B*V, H, W, 3)
            - multi_scale_xyz: list of xyz pointmaps at various resolutions [(B, V, H1, W1, 3), .....]
            - multi_scale_p2v: list of "Points to Voxel" mapping for the xyzs [(B, N), ....] where N=V*H1*W1
            - shape: (4) -> B, V, H, W
            - decoder_3d: (bool) whether to use 3D layers or not
            - actual_decoder_3d: (bool) False when the input is actually a 2D image / video
            - mesh_pc: B, N, 3 (optional mesh points for benchmark evaluations that need predictions on mesh pc)
            - mesh_p2v: B, N (maps each mesh point to a voxel location)
            
        Return:
            - mask_features: (B*num_views, F_m, H_m, W_m),
                m is largest f_map (res2)
            - multi_scale_features: feats of small scales [res5, res4, res3]
        """
        features = self.backbone(
            images, x_xyz=multi_scale_xyz, x_p2v=multi_scale_p2v, shape=shape, decoder_3d=decoder_3d
        )
        
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(
            features,
            shape,
            multi_scale_xyz,
            multi_scale_p2v,
            scannet_pc=mesh_pc,
            scannet_p2v=mesh_p2v,
            decoder_3d=decoder_3d,
            actual_decoder_3d=actual_decoder_3d,
        )
        
        return mask_features, multi_scale_features
        
        
        
        
        
        