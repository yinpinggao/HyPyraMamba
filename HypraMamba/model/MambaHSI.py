import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from mamba_ssm import Mamba

class PyramidAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(PyramidAttention, self).__init__()
        # Number of attention heads
        self.num_heads = num_heads
        # Temperature parameter for scaling the attention
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Query, Key, Value (QKV) convolution to generate attention-related features
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # Depthwise separable convolution for better performance
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim * 3,
                                    bias=bias)
        # Output projection layer after attention calculation
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # Compute QKV using the 1x1 convolution and depthwise convolution
        qkv = self.qkv_dwconv(self.qkv(x))
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
    def __init__(self, dim, num_heads, bias, num_scales=3, num_layers=2):
        super(PyramidRefinedChannelAttention, self).__init__()

        # Create PyramidAttention modules for different scales dynamically
        self.attention_modules = nn.ModuleList([
            PyramidAttention(dim, num_heads, bias) for _ in range(num_scales)
        ])

        # Create a set of layers for refining channel-wise attention for each scale
        self.attention_layers = nn.ModuleList([
            nn.ModuleList([PyramidAttention(dim, num_heads, bias) for _ in range(num_layers)])
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
    def __init__(self, channels, token_num=4, use_residual=True, group_num=4, num_scales=3, num_layers=2):
        super(ImprovedSpeMamba, self).__init__()
        self.orig_channels = channels
        self.token_num = token_num
        self.use_residual = use_residual
        # Set group_channel_num based on token_num and channels
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num
        assert self.token_num * self.group_channel_num == self.channel_num
        # Initialize PyramidRefinedChannelAttention
        self.pyramid_refined_attention = PyramidRefinedChannelAttention(
            dim=self.channel_num,  # Apply attention on channel_num
            num_heads=4,  # You can adjust num_heads as per your requirements
            bias=True,
            num_scales=num_scales,
            num_layers=num_layers
        )
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
        assert C == self.orig_channels, \
            f'ImprovedSpeMamba expected {self.orig_channels} channels, got {C}.'
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = x.new_zeros((B, pad_c, H, W))
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        return x

    def forward(self, x):
        # Apply padding to the input if necessary
        x_pad = self.padding_feature(x)
        # Inject first-order spectral differences before spectral attention.
        x_diff = torch.diff(x_pad, n=1, dim=1)
        x_diff = F.pad(x_diff, (0, 0, 0, 0, 0, 1), value=0.0)
        x_pad = x_pad + x_diff
        # Apply PyramidRefinedChannelAttention directly to the input tensor
        x_re = self.pyramid_refined_attention(x_pad)

        B, C, H, W = x_re.shape
        assert C == self.channel_num, \
            f'Expected attention output with {self.channel_num} channels, got {C}.'
        assert self.token_num * self.group_channel_num == self.channel_num

        # Treat each spatial pixel as one spectral grouped-token sequence.
        x_re = x_re.permute(0, 2, 3, 1).contiguous()
        x_re_flat = x_re.view(B * H * W, self.token_num, self.group_channel_num)
        # Apply Mamba for feature learning
        x_recon = self.mamba(x_re_flat)

        # Reshape back to original dimensions
        x_recon = x_recon.view(B, H, W, self.channel_num)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        # Apply the final projection to map the feature map to the output space
        x_recon = self.proj(x_recon)
        if self.channel_num > self.orig_channels:
            x_recon = x_recon[:, :self.orig_channels, :, :]
        # If residual connection is enabled, add the input to the output
        return x_recon + x if self.use_residual else x_recon


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


def _consume_removed_spatial_init_rng(channels):
    """
    Preserve the pre-cleanup parameter-init RNG sequence.

    These layers were removed from the forward graph, but their original
    constructors consumed random numbers during weight initialization. This
    project trains from scratch, so deleting those constructors changed the
    initial weights of later live modules under the same seed and caused metric
    drift. We instantiate the removed layers locally, then discard them
    immediately, so:
    1. they do not become part of the model/state_dict,
    2. they do not run in forward,
    3. later live layers still see the same RNG state as before cleanup.
    """
    out_channels = channels
    reduction = 16
    window_size = 7
    group_kernel_sizes = [3, 5, 7, 9]
    group_chans = channels // 4

    multi_scale_conv = nn.Sequential(
        nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
        nn.Conv2d(channels, out_channels, kernel_size=5, padding=2),
        nn.Conv2d(channels, out_channels, kernel_size=7, padding=3),
        nn.Conv2d(channels, out_channels, kernel_size=1),
        nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
        nn.GroupNorm(1, out_channels),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=False),
        nn.Sigmoid(),
    )

    scsa = nn.ModuleList([
        nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0], padding=group_kernel_sizes[0] // 2, groups=group_chans),
        nn.ModuleList([
            nn.Conv1d(group_chans, group_chans, kernel_size=size, padding=size // 2, groups=group_chans)
            for size in group_kernel_sizes[1:]
        ]),
        nn.GroupNorm(4, channels),
        nn.GroupNorm(4, channels),
        nn.GroupNorm(1, channels),
        nn.Conv2d(channels, channels, kernel_size=1, bias=False, groups=channels),
        nn.Conv2d(channels, channels, kernel_size=1, bias=False, groups=channels),
        nn.Conv2d(channels, channels, kernel_size=1, bias=False, groups=channels),
        nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size),
    ])

    del multi_scale_conv, scsa


class ImprovedSpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4, token_num=4, num_scales=3, num_layers=2):
        super(ImprovedSpaMamba, self).__init__()
        self.use_residual = use_residual
        self.token_num = token_num
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num
        self.pyramid_refined_attention = PyramidRefinedChannelAttention(
            dim=self.channel_num,
            num_heads=4,
            bias=True,
            num_scales=num_scales,
            num_layers=num_layers
        )

        self.mamba = Mamba(
            d_model=channels,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # Keep init order numerically aligned with the pre-cleanup model.
        _consume_removed_spatial_init_rng(channels)
        self.spatial_prior = LightSpatialPrior(channels, group_num=group_num)

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, channels),
            nn.SiLU()
        )

    def forward(self, x):
        x_prior = self.spatial_prior(x)
        x_re = self.pyramid_refined_attention(x_prior)
        B, C, H, W = x_re.shape
        x_flat = x_re.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_recon = self.proj(x_recon)

        return x_recon + x_prior if self.use_residual else x_recon


class CrossBranchBridge(nn.Module):
    """
    Channel-wise gating bridge. No spatial attention, O(C) not O(N^2).
    """
    def __init__(self, channels, reduction=4):
        super(CrossBranchBridge, self).__init__()
        mid = max(channels // reduction, 4)
        # spa gated by spe channel descriptor
        self.gate_spa = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )
        # spe gated by spa channel descriptor
        self.gate_spe = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )
        self.gamma_spa = nn.Parameter(torch.zeros(1))
        self.gamma_spe = nn.Parameter(torch.zeros(1))

    def forward(self, spa_feat, spe_feat):
        # spe tells spa which channels are spectrally informative
        g_spa = self.gate_spa(spe_feat).unsqueeze(-1).unsqueeze(-1)
        spa_out = spa_feat + self.gamma_spa * (g_spa * spa_feat)

        # spa tells spe which channels are spatially informative
        g_spe = self.gate_spe(spa_feat).unsqueeze(-1).unsqueeze(-1)
        spe_out = spe_feat + self.gamma_spe * (g_spe * spe_feat)
        return spa_out, spe_out
class ConflictSuppressedCCAF(nn.Module):
    """
    在 CCAF 基础上加入冲突抑制，只在低冲突通道放大共识项。
    """
    def __init__(self, channels, reduction=8):
        super(ConflictSuppressedCCAF, self).__init__()
        hidden = max(channels // reduction, 8)
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
        self.consensus_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )
        self.conflict_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )
        # 0 初始化，保证训练初期尽量贴近原竞争融合。
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, spa_feat, spe_feat):
        assert spa_feat.dim() == 4 and spe_feat.dim() == 4, \
            'ConflictSuppressedCCAF expects 4D inputs [B, C, H, W].'
        assert spa_feat.shape == spe_feat.shape, \
            'ConflictSuppressedCCAF requires spa_feat and spe_feat to have identical shapes.'

        spa_logit = self.fc_spa(spa_feat)
        spe_logit = self.fc_spe(spe_feat)
        weights = torch.softmax(torch.stack([spa_logit, spe_logit], dim=1), dim=1)
        w_spa = weights[:, 0, :].unsqueeze(-1).unsqueeze(-1)
        w_spe = weights[:, 1, :].unsqueeze(-1).unsqueeze(-1)

        common_feat = spa_feat * spe_feat
        g_cons = self.consensus_gate(common_feat).unsqueeze(-1).unsqueeze(-1)

        diff_feat = torch.abs(spa_feat - spe_feat)
        g_conflict = self.conflict_gate(diff_feat).unsqueeze(-1).unsqueeze(-1)

        competitive = w_spa * spa_feat + w_spe * spe_feat
        consensus = g_cons * 0.5 * (spa_feat + spe_feat)
        return competitive + self.beta * consensus * (1 - g_conflict)

class ImprovedBothMamba(nn.Module):
    def __init__(self, channels, token_num, use_residual, group_num=4):
        super(ImprovedBothMamba, self).__init__()
        self.use_residual = use_residual
        self.spa_mamba = ImprovedSpaMamba(
            channels,
            use_residual=use_residual,
            group_num=group_num,
        )
        self.spe_mamba = ImprovedSpeMamba(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)
        self.cross_bridge = CrossBranchBridge(channels, reduction=4)
        self.fusion = ConflictSuppressedCCAF(channels, reduction=8)

    def forward(self, x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        spa_x, spe_x = self.cross_bridge(spa_x, spe_x)
        fusion_x = self.fusion(spa_x, spe_x)
        return fusion_x + x if self.use_residual else fusion_x

    def get_fusion_beta(self):
        if hasattr(self.fusion, 'beta'):
            return self.fusion.beta.detach().item()
        return None

class ImprovedMambaHSI(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, use_residual=True,
                 token_num=4, group_num=4, output_stride=1):
        super(ImprovedMambaHSI, self).__init__()
        if output_stride == 1:
            self.downsample = nn.Identity()
        elif output_stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError('output_stride must be 1 or 2.')

        mid_channels = max(hidden_dim // 2, group_num)
        if mid_channels % group_num != 0:
            mid_channels += group_num - mid_channels % group_num

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        self.backbone = ImprovedBothMamba(
            hidden_dim,
            token_num=token_num,
            use_residual=use_residual,
            group_num=group_num,
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.recon_head = nn.Sequential(
            nn.Conv2d(hidden_dim, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, return_aux=False):
        x = self.patch_embedding(x)
        x = self.backbone(x)
        x_feat = self.downsample(x)
        logits = self.cls_head(x_feat)

        if return_aux:
            recon = self.recon_head(x_feat)
            return logits, recon

        return logits

    def get_fusion_beta(self):
        if hasattr(self.backbone, 'get_fusion_beta'):
            return self.backbone.get_fusion_beta()
        return None
