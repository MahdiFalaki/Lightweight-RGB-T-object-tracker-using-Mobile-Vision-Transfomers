#
# For licensing see accompanying LICENSE file.
#
from torch import nn, Tensor
import torch
from typing import Optional, Dict, Tuple, Union, Sequence
from .mobilevit_block import MobileViTBlock, MobileViTBlockv2


class MobileViTv2_Track_Block(MobileViTBlockv2):
    """
    This class defines the Multimodal Mobile Vision Transformer Block deployed in the proposed model.
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        attn_unit_dim: int,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        n_attn_blocks: Optional[int] = 2,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        patch_h: Optional[int] = 8,
        patch_w: Optional[int] = 8,
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        attn_norm_layer: Optional[str] = "layer_norm_2d",
        *args,
        **kwargs):

        super().__init__(opts, in_channels, attn_unit_dim, ffn_multiplier, n_attn_blocks, attn_dropout, dropout,
                                 ffn_dropout, patch_h, patch_w, conv_ksize, dilation, attn_norm_layer, *args, **kwargs)
        self.mixed_attn = opts["mixed_attn"]
        self.attn_unit_dim = attn_unit_dim

    def forward_spatial(self, x_v: Tensor, z_v: Tensor, x_i: Tensor, z_i: Tensor) -> Tensor:

        fm_x_v = self.local_rep(x_v)
        fm_z_v = self.local_rep(z_v)
        fm_x_i = self.local_rep(x_i)
        fm_z_i = self.local_rep(z_i)

        # convert feature map to patches
        if self.enable_coreml_compatible_fn:
            patches_x_v, output_size_x = self.unfolding_coreml(fm_x_v)
            patches_z_v, output_size_z = self.unfolding_coreml(fm_z_v)
            patches_x_i, _ = self.unfolding_coreml(fm_x_i)
            patches_z_i, _ = self.unfolding_coreml(fm_z_i)
        else:
            patches_x_v, output_size_x = self.unfolding_pytorch(fm_x_v)
            patches_z_v, output_size_z = self.unfolding_pytorch(fm_z_v)
            patches_x_i, _ = self.unfolding_pytorch(fm_x_i)
            patches_z_i, _ = self.unfolding_pytorch(fm_z_i)

        if self.mixed_attn is True and self.attn_unit_dim == 192:

            # concatenate search (i.e., x) and template (i.e., z) feature patches
            concatenated_patches_v = torch.cat((patches_x_v, patches_z_v), 3)
            concatenated_patches_i = torch.cat((patches_x_i, patches_z_i), 3)

            concat_all = torch.cat((concatenated_patches_v, concatenated_patches_i), 3)

            # perform joint feature extraction and template-search region fusion using the separable transformer blocks
            for transformer_layer in self.global_rep:
                concat_all = transformer_layer(concat_all)

            concatenated_patches_v = concat_all[:, :, :, :(patches_x_v.size(3)+patches_z_v.size(3))]
            concatenated_patches_i = concat_all[:, :, :, (patches_x_v.size(3)+patches_z_v.size(3)):]

        elif self.mixed_attn is True and self.attn_unit_dim == 128:

            # concatenate search (i.e., x) and template (i.e., z) feature patches
            concatenated_patches_v = torch.cat((patches_x_v, patches_z_v), 3)
            concatenated_patches_i = torch.cat((patches_x_i, patches_z_i), 3)

            # perform joint feature extraction and template-search region fusion using the separable transformer blocks
            for transformer_layer in self.global_rep:
                concatenated_patches_v = transformer_layer(concatenated_patches_v)
                concatenated_patches_i = transformer_layer(concatenated_patches_i)

        else:
            # perform feature extraction only (i.e., no feature fusion) using the separable transformer blocks
            for transformer_layer in self.global_rep:
                patches_x = transformer_layer(patches_x)
                patches_z = transformer_layer(patches_z)

        # split search (i.e., x) and template (i.e., z) feature patches
        patches_x_v = concatenated_patches_v[:, :, :, 0:patches_x_v.size(3)]
        patches_z_v = concatenated_patches_v[:, :, :, patches_x_v.size(3):]
        patches_x_i = concatenated_patches_i[:, :, :, 0:patches_x_v.size(3)]
        patches_z_i = concatenated_patches_i[:, :, :, patches_x_v.size(3):]

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        if self.enable_coreml_compatible_fn:
            fm_x_v = self.folding_coreml(patches=patches_x_v, output_size=output_size_x)
            fm_z_v = self.folding_coreml(patches=patches_z_v, output_size=output_size_z)
            fm_x_i = self.folding_coreml(patches=patches_x_i, output_size=output_size_x)
            fm_z_i = self.folding_coreml(patches=patches_z_i, output_size=output_size_z)
        else:
            fm_x_v = self.folding_pytorch(patches=patches_x_v, output_size=output_size_x)
            fm_z_v = self.folding_pytorch(patches=patches_z_v, output_size=output_size_z)
            fm_x_i = self.folding_pytorch(patches=patches_x_i, output_size=output_size_x)
            fm_z_i = self.folding_pytorch(patches=patches_z_i, output_size=output_size_z)

        fm_x_v = self.conv_proj(fm_x_v)
        fm_z_v = self.conv_proj(fm_z_v)
        fm_x_i = self.conv_proj(fm_x_i)
        fm_z_i = self.conv_proj(fm_z_i)

        return fm_x_v, fm_z_v, fm_x_i, fm_z_i

    def forward(self, x_v: Tensor, z_v: Tensor, x_i: Tensor, z_i: Tensor):

        return self.forward_spatial(x_v, z_v, x_i, z_i)


class MobileViT_Track_Block(MobileViTBlock):

    def __init__(
        self,
        opts,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: Optional[int] = 2,
        head_dim: Optional[int] = 32,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[int] = 0.0,
        ffn_dropout: Optional[int] = 0.0,
        patch_h: Optional[int] = 8,
        patch_w: Optional[int] = 8,
        transformer_norm_layer: Optional[str] = "layer_norm",
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        no_fusion: Optional[bool] = False,
        *args,
        **kwargs):

        super().__init__(opts, in_channels, transformer_dim, ffn_dim, n_transformer_blocks, head_dim, attn_dropout,
                         dropout, ffn_dropout, patch_h, patch_w, transformer_norm_layer, conv_ksize, dilation,
                         no_fusion, *args, **kwargs)

    def forward(self, x: Tensor, z: Tensor):
        return self.forward_spatial(x, z)

    def forward_spatial(self, x: Tensor, z: Tensor) -> Tensor:
        res_x = x
        res_z = z

        fm_x = self.local_rep(x)
        fm_z = self.local_rep(z)

        # convert feature map to patches
        patches_x, info_dict_x = self.unfolding(fm_x)
        patches_z, info_dict_z = self.unfolding(fm_z)

        # concatenate search (i.e., x) and template (i.e., z) feature patches
        concatenated_patches = torch.cat((patches_x, patches_z), 1)

        # learn global representations
        for transformer_layer in self.global_rep:
            concatenated_patches = transformer_layer(concatenated_patches)

        # split search (i.e., x) and template (i.e., z) feature patches
        patches_x = concatenated_patches[:, 0:info_dict_x['total_patches'], :]
        patches_z = concatenated_patches[:, info_dict_x['total_patches']:, :]

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm_x = self.folding(patches=patches_x, info_dict=info_dict_x)
        fm_z = self.folding(patches=patches_z, info_dict=info_dict_z)

        fm_x = self.conv_proj(fm_x)
        fm_z = self.conv_proj(fm_z)

        if self.fusion is not None:
            fm_x = self.fusion(torch.cat((res_x, fm_x), dim=1))
            fm_z = self.fusion(torch.cat((res_z, fm_z), dim=1))

        return fm_x, fm_z
