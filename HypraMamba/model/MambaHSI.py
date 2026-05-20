import math
import torch
from torch import nn
from einops import rearrange
from mamba_ssm import Mamba


VALID_ABLATIONS = {
    'full',
    'wo_lpps',
    'wo_dgs',
    'wo_lsp',
    'wo_prca',
    'wo_competitive',
}
SPATIAL_BRANCH_DISABLED_ABLATIONS = {'wo_lpps'}
SPECTRAL_BRANCH_DISABLED_ABLATIONS = {'wo_dgs'}
SPATIAL_PRIOR_DISABLED_ABLATIONS = {'wo_lsp'}
SPATIAL_PRCA_DISABLED_ABLATIONS = {'wo_prca'}
COMPETITIVE_FUSION_DISABLED_ABLATIONS = {'wo_competitive'}
VALID_OUTER_RESIDUAL_MODES = {'standard', 'no_outer', 'scaled'}


def _validate_ablation(ablation):
    if ablation not in VALID_ABLATIONS:
        raise ValueError('Unsupported ablation: {}'.format(ablation))
    return ablation


def _validate_outer_residual_mode(mode):
    if mode not in VALID_OUTER_RESIDUAL_MODES:
        raise ValueError('Unsupported outer_residual_mode: {}'.format(mode))
    return mode


def _normalize_dilations(dilation):
    if isinstance(dilation, str):
        dilations = tuple(int(value.strip()) for value in dilation.split(',') if value.strip())
    elif isinstance(dilation, int):
        dilations = (dilation,)
    else:
        dilations = tuple(int(value) for value in dilation)

    if len(dilations) == 0:
        raise ValueError('dilation must contain at least one value.')
    if any(value < 1 for value in dilations):
        raise ValueError('all dilation values must be positive integers.')

    return dilations


class PyramidAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, dilation=2):
        super(PyramidAttention, self).__init__()
        # Number of attention heads
        self.num_heads = num_heads
        self.dilations = _normalize_dilations(dilation)
        # Temperature parameter for scaling the attention
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Query, Key, Value (QKV) convolution to generate attention-related features
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # Depthwise separable convolution for better performance
        self.qkv_dwconvs = nn.ModuleList([
            nn.Conv2d(
                dim * 3,
                dim * 3,
                kernel_size=3,
                stride=1,
                dilation=value,
                padding=value,
                groups=dim * 3,
                bias=bias
            )
            for value in self.dilations
        ])
        if len(self.dilations) > 1:
            dilation_logits = torch.zeros(len(self.dilations), dtype=torch.float32)
            preferred_index = self.dilations.index(3) if 3 in self.dilations else len(self.dilations) - 1
            dilation_logits[preferred_index] = 2.0
            self.dilation_logits = nn.Parameter(dilation_logits)
        else:
            self.register_parameter('dilation_logits', None)
        # Output projection layer after attention calculation
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # Compute QKV using the 1x1 convolution and depthwise convolution
        qkv_base = self.qkv(x)
        if self.dilation_logits is None:
            qkv = self.qkv_dwconvs[0](qkv_base)
        else:
            weights = torch.softmax(self.dilation_logits, dim=0)
            qkv = None
            for weight, qkv_dwconv in zip(weights, self.qkv_dwconvs):
                branch_qkv = weight * qkv_dwconv(qkv_base)
                qkv = branch_qkv if qkv is None else qkv + branch_qkv
        # Split QKV into query, key, and value
        q, k, v = qkv.chunk(3, dim=1)

        # Rearrange the tensor to shape suitable for attention computation
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # Normalize query and key for stable attention computation
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Compute attention scores (scaled dot-product)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # Apply softmax to normalize the attention scores
        attn = attn.softmax(dim=-1)

        # Compute the output by applying attention to the value tensor
        out = (attn @ v)
        # Rearrange the output to match the original spatial dimensions
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # Apply the final projection to get the output
        out = self.project_out(out)

        return out

class PyramidRefinedChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, num_scales=3, num_layers=2, dilation=2):
        super(PyramidRefinedChannelAttention, self).__init__()

        # Create PyramidAttention modules for different scales dynamically
        self.attention_modules = nn.ModuleList([
            PyramidAttention(dim, num_heads, bias, dilation=dilation) for _ in range(num_scales)
        ])

        # Create a set of layers for refining channel-wise attention for each scale
        self.attention_layers = nn.ModuleList([
            nn.ModuleList([PyramidAttention(dim, num_heads, bias, dilation=dilation) for _ in range(num_layers)])
            for _ in range(num_scales)
        ])

        # Final projection layer to map the concatenated feature maps to the output
        self.project_out = nn.Conv2d(dim * num_scales, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        outputs = []

        # Loop over each scale to process the input at multiple resolutions
        for i, attention_module in enumerate(self.attention_modules):
            # Downsample the input image for higher scales
            if i == 0:
                scaled_input = x  # No downsampling for the first scale
            else:
                scaled_input = nn.functional.avg_pool2d(x, kernel_size=2 ** i, stride=2 ** i)

            # Apply the first layer of pyramid attention to the scaled input
            output = attention_module(scaled_input)

            # Apply multiple layers of pyramid attention for channel refinement
            for layer in self.attention_layers[i]:
                output = layer(output)

            # Upsample the output to match the original resolution
            if i > 0:
                output = torch.nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)

            outputs.append(output)

        # Concatenate the outputs of all scales along the channel dimension
        out = torch.cat(outputs, dim=1)

        # Project the concatenated result to the final output space
        out = self.project_out(out)

        return out


class ImprovedSpeMamba(nn.Module):
    def __init__(self, channels, token_num=4, use_residual=True, group_num=4):
        super(ImprovedSpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual
        # Set group_channel_num based on token_num and channels
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num
        # Initialize Mamba module for feature learning
        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        # Projection layer to project the concatenated feature maps to the output
        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = x.new_zeros((B, pad_c, H, W))
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def spectral_difference_enhance(self, x):
        diff = x.new_zeros(x.shape)
        diff[:, :-1, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
        return x + diff

    def forward(self, x):
        # Inject first-order spectral variation before grouped tokenization.
        x_diff = self.spectral_difference_enhance(x)
        # Apply padding to the input if necessary
        x_re = self.padding_feature(x_diff)

        # Treat each spatial location as one spectral token sequence.
        B, C, H, W = x_re.shape
        origin_c = x.shape[1]
        x_re_flat = x_re.permute(0, 2, 3, 1).reshape(
            B * H * W,
            self.token_num,
            self.group_channel_num,
        )
        # Apply Mamba for feature learning
        x_out = self.mamba(x_re_flat)

        # Reshape back to original dimensions
        x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        # Apply the final projection to map the feature map to the output space
        x_out = self.proj(x_out)[:, :origin_c, :, :]
        # If residual connection is enabled, add the input to the output
        return x_out + x if self.use_residual else x_out


class LightSpatialPrior(nn.Module):
    def __init__(self, channels, group_num=4, reduction=4):
        super(LightSpatialPrior, self).__init__()
        mid = max(channels // reduction, 8)

        self.dw = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1, groups=channels
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(mid, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(group_num, channels)
        self.act = nn.SiLU()

    def forward(self, x):
        local_feat = self.dw(x)
        gate = self.spatial_gate(x)
        out = local_feat * gate
        out = self.pw(out)
        out = self.norm(out)
        out = self.act(out)
        return out + x


class ImprovedSpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4, token_num=4, num_scales=3, num_layers=2,
                 pyramid_dilation=2, ablation='full'):
        super(ImprovedSpaMamba, self).__init__()
        self.ablation = _validate_ablation(ablation)
        self.use_residual = use_residual
        self.token_num = token_num
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num
        self.use_prca = self.ablation not in SPATIAL_PRCA_DISABLED_ABLATIONS
        if self.use_prca:
            self.pyramid_refined_attention = PyramidRefinedChannelAttention(
                dim=self.channel_num,
                num_heads=4,
                bias=True,
                num_scales=num_scales,
                num_layers=num_layers,
                dilation=pyramid_dilation
            )
        else:
            self.pyramid_refined_attention = None

        self.mamba = Mamba(
            d_model=channels,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.use_spatial_prior = self.ablation not in SPATIAL_PRIOR_DISABLED_ABLATIONS
        if self.use_spatial_prior:
            self.spatial_prior = LightSpatialPrior(channels, group_num=group_num)
        else:
            self.spatial_prior = None

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, channels),
            nn.SiLU()
        )

    def forward(self, x):
        if self.spatial_prior is None:
            x_prior = x
        else:
            x_prior = self.spatial_prior(x)

        if self.pyramid_refined_attention is None:
            x_re = x_prior
        else:
            x_re = self.pyramid_refined_attention(x_prior)
        B, C, H, W = x_re.shape
        x_flat = x_re.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_flat = self.mamba(x_flat)

        x_out = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_out = self.proj(x_out)

        return x_out + x_prior if self.use_residual else x_out


class CompetitiveFusion(nn.Module):
    def __init__(self, channels):
        super(CompetitiveFusion, self).__init__()
        self.fc_spa = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, channels, bias=False),
        )
        self.fc_spe = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, channels, bias=False),
        )

    def forward(self, spa_feat, spe_feat):
        assert spa_feat.dim() == 4 and spe_feat.dim() == 4, \
            'CompetitiveFusion expects 4D inputs [B, C, H, W].'
        assert spa_feat.shape == spe_feat.shape, \
            'CompetitiveFusion requires spa_feat and spe_feat to have identical shapes.'

        spa_logit = self.fc_spa(spa_feat)
        spe_logit = self.fc_spe(spe_feat)
        weights = torch.softmax(torch.stack([spa_logit, spe_logit], dim=1), dim=1)
        w_spa = weights[:, 0, :].unsqueeze(-1).unsqueeze(-1)
        w_spe = weights[:, 1, :].unsqueeze(-1).unsqueeze(-1)

        return w_spa * spa_feat + w_spe * spe_feat


class ImprovedBothMamba(nn.Module):
    def __init__(self, channels, token_num, use_residual, group_num=4, pyramid_dilation=2,
                 ablation='full', outer_residual_mode='standard', outer_residual_alpha=1.0):
        super(ImprovedBothMamba, self).__init__()
        self.ablation = _validate_ablation(ablation)
        self.outer_residual_mode = _validate_outer_residual_mode(outer_residual_mode)
        self.outer_residual_alpha = float(outer_residual_alpha)
        self.use_residual = use_residual
        self.use_spatial_branch = self.ablation not in SPATIAL_BRANCH_DISABLED_ABLATIONS
        self.use_spectral_branch = self.ablation not in SPECTRAL_BRANCH_DISABLED_ABLATIONS

        if self.use_spatial_branch:
            self.spa_mamba = ImprovedSpaMamba(
                channels,
                use_residual=use_residual,
                group_num=group_num,
                pyramid_dilation=pyramid_dilation,
                ablation=ablation,
            )
        else:
            self.spa_mamba = None

        if self.use_spectral_branch:
            self.spe_mamba = ImprovedSpeMamba(
                channels,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
            )
        else:
            self.spe_mamba = None

        if (
                self.use_spatial_branch
                and self.use_spectral_branch
                and self.ablation not in COMPETITIVE_FUSION_DISABLED_ABLATIONS
        ):
            self.fusion = CompetitiveFusion(channels)
        else:
            self.fusion = None

    def _apply_outer_residual(self, x, block_x):
        if not self.use_residual or self.outer_residual_mode == 'no_outer':
            return block_x
        if self.outer_residual_mode == 'scaled':
            return x + self.outer_residual_alpha * block_x
        return block_x + x

    def forward(self, x):
        if self.spa_mamba is None:
            spe_x = self.spe_mamba(x)
            return self._apply_outer_residual(x, spe_x)

        if self.spe_mamba is None:
            spa_x = self.spa_mamba(x)
            return self._apply_outer_residual(x, spa_x)

        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)

        if self.fusion is None:
            fusion_x = 0.5 * (spa_x + spe_x)
        else:
            fusion_x = self.fusion(spa_x, spe_x)
        return self._apply_outer_residual(x, fusion_x)


class ImprovedMambaHSI(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, use_residual=True,
                 token_num=4, group_num=4, pyramid_dilation=(2, 3), ablation='full',
                 outer_residual_mode='standard', outer_residual_alpha=1.0):
        super(ImprovedMambaHSI, self).__init__()
        self.ablation = _validate_ablation(ablation)

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        self.mamba = nn.Sequential(
            ImprovedBothMamba(
                hidden_dim,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                pyramid_dilation=pyramid_dilation,
                ablation=ablation,
                outer_residual_mode=outer_residual_mode,
                outer_residual_alpha=outer_residual_alpha,
            ),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x_feat = self.mamba(x)
        return self.cls_head(x_feat)
