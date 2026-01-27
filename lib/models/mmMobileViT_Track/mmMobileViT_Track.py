"""
mmMobileViT_Track model. Developed on SMAT.
"""
import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from ..layers.neck import build_neck, build_feature_fusor
from ..layers.head import build_box_head
from .mobilevit_v2 import MobileViTv2_backbone
from ...utils.box_ops import box_xyxy_to_cxcywh
from ..modules.mobilevit_block import FusionTransformer


class mmMobileViT_Track(nn.Module):
    """ This is the base class for mmMobileViT_Track developed on SMAT (Gopal, Amer. WACV 2024 ) """

    def __init__(self, backbone, neck, neck_i, feature_fusor, feature_fusor_i, box_head, aux_loss=False,
                 head_type="CORNER"):
        """
        Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        if neck is not None:
            self.neck = neck
            self.feature_fusor = feature_fusor
            self.neck_i = neck_i
            self.feature_fusor_i = feature_fusor_i
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        fusion_opts = {
            "attn_unit_dim": 128,
            "ffn_multiplier": 2.0,
            "n_attn_blocks": 1,
            "attn_dropout": 0.1,
            "dropout": 0.1,
            "ffn_dropout": 0.1,
            "attn_norm_layer": "layer_norm_2d",
            "conv_layer_activation_name": "relu"
        }

        self.feature_fusion = FusionTransformer(
            opts=fusion_opts,
            in_channels=128,
            attn_unit_dim=fusion_opts["attn_unit_dim"],
            ffn_multiplier=fusion_opts["ffn_multiplier"],
            n_attn_blocks=fusion_opts["n_attn_blocks"],
            attn_dropout=fusion_opts["attn_dropout"],
            dropout=fusion_opts["dropout"],
            ffn_dropout=fusion_opts["ffn_dropout"],
            attn_norm_layer=fusion_opts["attn_norm_layer"]
        )

    def forward(self, template: torch.Tensor, search: torch.Tensor):
        x_v, z_v, x_i, z_i = self.backbone(x=search, z=template)

        # Forward neck
        x_v, z_v = self.neck(x_v, z_v)
        x_i, z_i = self.neck_i(x_i, z_i)

        # Forward feature fusor
        feat_fused_v = self.feature_fusor(z_v, x_v)
        feat_fused_i = self.feature_fusor_i(z_i, x_i)

        # Fuse RGB and IR features using cross-fusion transformer + weighted fusion
        fused_features = self.feature_fusion(feat_fused_v, feat_fused_i)

        # Forward head
        out = self.forward_head(fused_features, None)

        return out

    def forward_head(self, backbone_feature, gt_score_map=None):
        """
        backbone_feature: output embeddings of the backbone for search region. Block adapted from SMAT
        """
        opt_feat = backbone_feature.contiguous()
        bs, _, _, _ = opt_feat.size()

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif "CENTER" in self.head_type:
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_mmMobileViT_Track(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if "mobilevitv2" in cfg.MODEL.BACKBONE.TYPE:
        width_multiplier = float(cfg.MODEL.BACKBONE.TYPE.split('-')[-1])
        backbone = create_mobilevitv2_backbone(pretrained, width_multiplier,
                                               has_mixed_attn=cfg.MODEL.BACKBONE.MIXED_ATTN)
        if cfg.MODEL.BACKBONE.MIXED_ATTN is True:
            backbone.mixed_attn = True
        else:
            backbone.mixed_attn = False
        hidden_dim = backbone.model_conf_dict['layer4']['out']
    else:
        raise NotImplementedError

    # build neck module to fuse template and search region features
    if cfg.MODEL.NECK:
        neck = build_neck(cfg=cfg, hidden_dim=hidden_dim)
        neck_i = build_neck(cfg=cfg, hidden_dim=hidden_dim)
    else:
        neck = nn.Identity()

    if cfg.MODEL.NECK.TYPE == "BN_PWXCORR":
        feature_fusor = build_feature_fusor(cfg=cfg, in_features=backbone.model_conf_dict['layer4']['out'],
                                            xcorr_out_features=cfg.MODEL.NECK.NUM_CHANNS_POST_XCORR)
        feature_fusor_i = build_feature_fusor(cfg=cfg, in_features=backbone.model_conf_dict['layer4']['out'],
                                              xcorr_out_features=cfg.MODEL.NECK.NUM_CHANNS_POST_XCORR)
    elif cfg.MODEL.NECK.TYPE == "BN_SSAT" or cfg.MODEL.NECK.TYPE == "BN_HSSAT":
        feature_fusor = build_feature_fusor(cfg=cfg, in_features=backbone.model_conf_dict['layer4']['out'],
                                            xcorr_out_features=None)
        feature_fusor_i = build_feature_fusor(cfg=cfg, in_features=backbone.model_conf_dict['layer4']['out'],
                                              xcorr_out_features=None)
    else:
        raise NotImplementedError

    box_head = build_box_head(cfg, cfg.MODEL.HEAD.NUM_CHANNELS)

    model = mmMobileViT_Track(
        backbone=backbone,
        neck=neck,
        neck_i=neck_i,
        feature_fusor=feature_fusor,
        feature_fusor_i=feature_fusor_i,
        box_head=box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    return model


def create_mobilevitv2_backbone(pretrained, width_multiplier, has_mixed_attn):
    """
    function to create an instance of MobileViT backbone
    Args:
        pretrained:  str
        path to the pretrained image classification model to initialize the weights.
        If empty, the weights are randomly initialized
    Returns:
        model: nn.Module
        An object of Pytorch's nn.Module with MobileViT-v2 backbone (i.e., layer-1 to layer-4)
    """
    opts = {}
    opts['mode'] = width_multiplier
    opts['head_dim'] = None
    opts['number_heads'] = 4
    opts['conv_layer_normalization_name'] = 'batch_norm'
    opts['conv_layer_activation_name'] = 'relu'
    opts['mixed_attn'] = has_mixed_attn
    model = MobileViTv2_backbone(opts)

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

        # assert missing_keys == [], "The backbone layers do not exactly match with the checkpoint state dictionaries. " \
        #                            "Please have a look at what those missing keys are!"

        print('Load pretrained model from: ' + pretrained)

    return model
