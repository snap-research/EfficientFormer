"""
EfficientFormer
"""
import os
import copy
import torch
import torch.nn as nn

from typing import Dict
import itertools

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
import EfficientFormer

class SnapML_Attention(torch.nn.Module):
    """
    SnapML Compatible Multi-headed Attention with Positional Bias.
    """

    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4,
                 resolution=7):
        """
        Initializes Attention Layer.
        Args:
            dim: An `int` of the input/output dimension.
            key_dim: An `int` of key dimension.
            num_heads: An `int` of the number of heads.
            attn_ratio: An `int` for the attention ratio.
            resolution: An `int` for the resolution of the attention block.
        """
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.q = nn.Conv2d(dim, num_heads*key_dim, 1)
        self.k = nn.Conv2d(dim, num_heads*key_dim, 1)
        self.v = nn.Conv2d(dim, self.d*self.num_heads, 1)

        self.proj = nn.Conv2d(self.dh, dim, 1)

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, 1, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, :, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        N, C, H, W = x.shape

        q = self.q(x).reshape(N, 1, self.num_heads, self.key_dim)
        k = self.k(x).reshape(N, 1, self.num_heads, self.key_dim)
        v = self.v(x).reshape(N, 1, self.num_heads, self.d)

        q = q.permute(2,1,0,3)
        k = k.permute(2,1,3,0)
        v = v.permute(2,1,0,3)


        attn = (
                (q @ k) * self.scale
                +
                (self.attention_biases[:, :, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        A, B, C, D = attn.shape
        attn = attn.reshape(A*B*C, D)
        attn = attn.softmax(dim=1)
        attn = attn.reshape(A,B,C, D)
        
        x = (attn @ v)
        x = x.transpose(0,1).transpose(1, 2).reshape(N, self.dh, 1, 1)

        x = self.proj(x)
        return x


class SnapML_Flat(nn.Module):
    """
    Flatten tensor for ViT Blocks.
    """

    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        B, C, D, E = x.shape
        x = x.reshape(C, D*E, 1, 1).transpose(0,1)
        return x


class SnapML_Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    """

    def __init__(self, pool_size=3):
        """
        Initializes the pooling operation for PoolFormer.
        Args:
            pool_size: An `int` for the pool size.
        """
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) + -1.*x


class SnapML_LinearMlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        """
        Initializes the Linear MLP.
        Args:
            hidden_features: An `int` of the number of hidden features.
            out_features: An `int` of the output feature size.
            activation: A `torch.nn` class that defines the activation.
            drop: A `float` for the dropout rate.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SnapML_Meta3D(nn.Module):
    """
    SnapML Compatible Meta3D Block.
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        """
        Initializes the Meta3D block.
        Args:
            dim: An `int` of the input/output dimension.
            mlp_ratio: A `float` for the MLP ratio.
            activation: A `torch.nn` class that defines the activation.
            norm_layer: A `torch.nn` class that defines
                the normalization.
            drop_rate: A `float` for the dropout rate.
            drop_path_rate: A `float` for the drop path rate.
            use_layer_scale: A `bool` for using layer scale.
            layer_scale_init_value: A `float` for initializing layer scale.
        """

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = SnapML_Attention(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SnapML_LinearMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.mlp(self.norm2(x)))

        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SnapML_Meta4D(nn.Module):
    """
    SnapML Compatible Meta4D Block.
    """

    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.ReLU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        """
        Initializes the Meta4D block.
        Args:
            dim: An `int` of the input/output dimension.
            pool_size: An `int` for the pool size.
            mlp_ratio: A `float` for the MLP ratio.
            activation: A `torch.nn` class that defines the activation.
            norm_layer: A `torch.nn` class that defines
                the normalization.
            drop_rate: A `float` for the dropout rate.
            drop_path_rate: A `float` for the drop path rate.
            use_layer_scale: A `bool` for using layer scale.
            layer_scale_init_value: A `float` for initializing layer scale.
        """
        super().__init__()

        self.token_mixer = SnapML_Pooling(pool_size=pool_size)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = EfficientFormer.Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.token_mixer(x) * self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
            x = x + self.drop_path(
                self.mlp(x) * self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


def SnapML_meta_blocks(dim, index, layers,
                pool_size=3, mlp_ratio=4.,
                act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                drop_rate=.0, drop_path_rate=0.,
                use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1):
    blocks = []
    if index == 3 and vit_num == layers[index]:
        blocks.append(SnapML_Flat())
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        if index == 3 and layers[index] - block_idx <= vit_num:
            blocks.append(SnapML_Meta3D(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
        else:
            blocks.append(SnapML_Meta4D(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
            if index == 3 and layers[index] - block_idx - 1 == vit_num:
                blocks.append(SnapML_Flat())

    blocks = nn.Sequential(*blocks)
    return blocks


class SnapML_EfficientFormer(nn.Module):

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=4, downsamples=None,
                 pool_size=3,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU,
                 num_classes=1000,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 vit_num=0,
                 distillation=True,
                 **kwargs):
        """
        Initializes EfficientFormer model.
        Args:
            layers: A `list` of the number of blocks for the four stages.
            embed_dims: A `list` of the embedding dimensions for the
                four stages.
            mlp_ratios: A `list` of the mlp ratios for the four stages.
            downsamples: A `list` of the downsample flags for the four stages.
            pool_size: An `int` for the pool size.
            norm_layer: A `torch.nn` class that defines
                the normalization.
            activation: A `torch.nn` class that defines the activations.
            num_classes: An `int` for the number of output classes.
            down_patch_size: An `int` for the patch size of embedding layers.
            down_stride: An `int` for the stride of embedding layers.
            down_pad: An `int` for the padding size of embedding layers.
            drop_rate: A `float` for the dropout rate of the layers.
            drop_path: A `float` for the drop path rate of the layers.
            layer_scale: A `bool` for using layer scale.
            layer_scale_init_value: A `float` for initializing layer scale.
            fork_feat: A `bool` for using feature forking for downstream tasks.
            vit_num: An `int` for the Vision Transformer Model number.
            distillation: A `bool` indicating distillation mode.
            **kwargs: Additional keyword arguments to be passed.
        """
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = EfficientFormer.stem(3, embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = SnapML_meta_blocks(embed_dims[i], i, layers,
                                pool_size=pool_size, mlp_ratio=mlp_ratios,
                                act_layer=act_layer, norm_layer=norm_layer,
                                drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate,
                                use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                vit_num=vit_num)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    EfficientFormer.Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Conv2d(
                embed_dims[-1], num_classes, 1) if num_classes > 0 \
                else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Conv2d(
                    embed_dims[-1], num_classes, 1) if num_classes > 0 \
                    else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()
        self.training = False

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            # output features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean(0)), self.dist_head(x.mean(0))
        cls_out = (cls_out[0] + cls_out[1]) / 2
        cls_out = cls_out.unsqueeze(0).softmax(1)
        # for image classification
        return cls_out

@register_model
def SnapML_efficientformer_l0(pretrained=False, **kwargs):
    model = SnapML_EfficientFormer(
        layers=EfficientFormer.EfficientFormer_depth['l0'],
        embed_dims=EfficientFormer.EfficientFormer_width['l0'],
        downsamples=[True, True, True, True],
        vit_num=2,
        **kwargs)
    model.default_cfg = EfficientFormer._cfg(crop_pct=0.9)
    return model


@register_model
def SnapML_efficientformer_l1(pretrained=False, **kwargs):
    model = SnapML_EfficientFormer(
        layers=EfficientFormer.EfficientFormer_depth['l1'],
        embed_dims=EfficientFormer.EfficientFormer_width['l1'],
        downsamples=[True, True, True, True],
        vit_num=1,
        **kwargs)
    model.default_cfg = EfficientFormer._cfg(crop_pct=0.9)
    return model


@register_model
def SnapML_efficientformer_l3(pretrained=False, **kwargs):
    model = SnapML_EfficientFormer(
        layers=EfficientFormer.EfficientFormer_depth['l3'],
        embed_dims=EfficientFormer.EfficientFormer_width['l3'],
        downsamples=[True, True, True, True],
        vit_num=4,
        **kwargs)
    model.default_cfg = EfficientFormer._cfg(crop_pct=0.9)
    return model


@register_model
def SnapML_efficientformer_l7(pretrained=False, **kwargs):
    model = SnapML_EfficientFormer(
        layers=EfficientFormer.EfficientFormer_depth['l7'],
        embed_dims=EfficientFormer.EfficientFormer_width['l7'],
        downsamples=[True, True, True, True],
        vit_num=8,
        **kwargs)
    model.default_cfg = EfficientFormer._cfg(crop_pct=0.9)
    return model