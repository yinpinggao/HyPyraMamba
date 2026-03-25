import math
import typing as t

import torch
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba
from torch import nn

class SCSA(nn.Module):
    def __init__(self,
                 dim: int,
                 head_num: int,
                 window_size: int = 7,
                 group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
                 qkv_bias: bool = False,
                 fuse_bn: bool = False,
                 down_sample_mode: str = 'avg_pool',
                 attn_drop_ratio: float = 0.,
                 gate_layer: str = 'sigmoid',
                 ):
        super(SCSA, self).__init__()

        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = self.dim // 4

        # 局部和全局深度卷积层
        self.local_dwc = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=self.group_kernel_sizes[0],
                                   padding=self.group_kernel_sizes[0] // 2, groups=self.group_chans)
        self.global_dwcs = nn.ModuleList([
            nn.Conv1d(self.group_chans, self.group_chans, kernel_size=size, padding=size // 2, groups=self.group_chans)
            for size in self.group_kernel_sizes[1:]
        ])

        # 注意力门控层
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)
        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)

        # 查询、键、值卷积层
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        # 根据窗口大小和下采样模式选择下采样函数
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def conv_group(self, x, kernel_size: int) -> torch.Tensor:
        """
        简化卷积操作
        """
        return nn.Conv1d(x.size(1), x.size(1), kernel_size=kernel_size, padding=kernel_size // 2, groups=x.size(1))(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h_, w_ = x.size()

        # 计算水平和垂直方向的特征
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        # 计算水平注意力
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            *[gw(l_x_h) for gw in self.global_dwcs]
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        # 计算垂直注意力
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            *[gw(l_x_w) for gw in self.global_dwcs]
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        # 计算最终的加权结果
        x = x * x_h_attn * x_w_attn

        # 通道自注意力
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        # 计算查询、键、值
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        # 调整维度以进行注意力计算
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=self.head_num,
                      head_dim=self.head_dim)
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=self.head_num,
                      head_dim=self.head_dim)
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=self.head_num,
                      head_dim=self.head_dim)

        # 计算注意力
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))

        # 加权值
        attn = attn @ v
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=h_, w=w_)

        # 计算通道注意力
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)

        return attn * x

# Pyramid Attention module, which computes attention across spatial dimensions
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


# GCSA module with Pyramid-Refined Channel Attention (PRCA)
class PyramidRefinedChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, num_scales=3, num_layers=2):
        super(PyramidRefinedChannelAttention, self).__init__()

        # Store the number of heads and initialize the temperature for attention scaling
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

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


class LitePSA(nn.Module):
    def __init__(self, dim, num_heads, bias, num_scales=3, se_ratio=4):
        super(LitePSA, self).__init__()
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            dilation=2,
            padding=2,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim * num_scales, dim, kernel_size=1, bias=bias)

        se_hidden = max(dim // se_ratio, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, se_hidden, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(se_hidden, dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def _attention(self, x):
        _, _, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head d) h w -> b head d (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head d) h w -> b head d (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head d) h w -> b head d (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return rearrange(out, 'b head d (h w) -> b (head d) h w', h=h, w=w)

    def forward(self, x):
        _, _, h, w = x.shape
        outputs = []

        for scale_idx in range(self.num_scales):
            if scale_idx == 0:
                scaled_input = x
            else:
                stride = 2 ** scale_idx
                scaled_input = F.avg_pool2d(x, kernel_size=stride, stride=stride)

            scaled_out = self._attention(scaled_input)
            if scale_idx > 0:
                scaled_out = F.interpolate(
                    scaled_out,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False,
                )
            outputs.append(scaled_out)

        out = self.project_out(torch.cat(outputs, dim=1))
        return out * self.se(out)


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.ReLU()
        self.attention = ChannelAttention(out_channels)  # Add attention here

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = self.final_conv(out)
        out = self.norm(out)
        out = self.activation(out)
        return self.attention(out)


class DynamicConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, num_experts=4, reduction=4, dropout=0.1):
        super(DynamicConvBlock, self).__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.kernel_size = kernel_size

        # Generate dynamic weights using a lightweight attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, num_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Define a set of expert convolutions
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels)
            for _ in range(num_experts)
        ])

        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape

        # Get the attention weights for each expert
        attention_weights = self.attention(x).view(B, self.num_experts, 1, 1, 1)

        # Apply each expert convolution
        outputs = [conv(x).unsqueeze(1) for conv in self.convs]
        outputs = torch.cat(outputs, dim=1)  # Shape: (B, num_experts, C, H, W)

        # Weighted sum of the outputs from experts
        x = (outputs * attention_weights).sum(dim=1)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SharedDynamicConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, num_experts=2, reduction=4, dropout=0.1):
        super(SharedDynamicConvBlock, self).__init__()
        self.channels = channels
        self.num_experts = num_experts

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, num_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.base_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.residual_convs = nn.ModuleList([
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            )
            for _ in range(num_experts)
        ])

        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x):
        b, _, _, _ = x.shape
        attention_weights = self.attention(x).view(b, self.num_experts, 1, 1, 1)

        base = self.base_conv(x)
        outputs = [(base + conv(x)).unsqueeze(1) for conv in self.residual_convs]
        outputs = torch.cat(outputs, dim=1)

        x = (outputs * attention_weights).sum(dim=1)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        return x * self.sigmoid(y)


class PostMambaSE(nn.Module):
    def __init__(self, channels, reduction=4):
        super(PostMambaSE, self).__init__()
        hidden_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        return x * self.gate(scale)


class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, spa_feat, spe_feat):
        fused_input = torch.cat([spa_feat, spe_feat], dim=1)
        attention_map = self.attention(fused_input)
        spa_x_attended = spa_feat * attention_map
        spe_x_attended = spe_feat * (1 - attention_map)
        return spa_x_attended + spe_x_attended


class CrossGateFusion(nn.Module):
    def __init__(self, channels):
        super(CrossGateFusion, self).__init__()
        self.gate_spa = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gate_spe = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, spa_feat, spe_feat):
        g_spa = torch.sigmoid(self.gate_spa(spe_feat))
        g_spe = torch.sigmoid(self.gate_spe(spa_feat))
        return g_spa * spa_feat + g_spe * spe_feat


def build_attention_module(attention_mode, dim, num_heads=4, bias=True, num_scales=3, num_layers=2):
    if attention_mode == 'prca':
        return PyramidRefinedChannelAttention(
            dim=dim,
            num_heads=num_heads,
            bias=bias,
            num_scales=num_scales,
            num_layers=num_layers,
        )
    if attention_mode == 'lite_psa':
        return LitePSA(
            dim=dim,
            num_heads=num_heads,
            bias=bias,
            num_scales=num_scales,
        )
    if attention_mode == 'none':
        return nn.Identity()
    raise ValueError(f'Unsupported attention mode: {attention_mode}')


def build_dynamic_conv(dynamic_conv_mode, channels):
    if dynamic_conv_mode == 'dynamic':
        return DynamicConvBlock(channels=channels)
    if dynamic_conv_mode == 'shared':
        return SharedDynamicConvBlock(channels=channels)
    if dynamic_conv_mode == 'none':
        return nn.Identity()
    raise ValueError(f'Unsupported dynamic conv mode: {dynamic_conv_mode}')


def build_cls_head(hidden_dim, num_classes, cls_hidden_dim, group_num):
    if cls_hidden_dim and cls_hidden_dim > 0:
        return nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=cls_hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, cls_hidden_dim),
            nn.SiLU(),
            nn.Conv2d(in_channels=cls_hidden_dim, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )
    return nn.Conv2d(in_channels=hidden_dim, out_channels=num_classes, kernel_size=1, stride=1, padding=0)


class ImprovedSpeMamba(nn.Module):
    def __init__(
        self,
        channels,
        token_num=4,
        use_residual=True,
        group_num=4,
        num_scales=3,
        num_layers=2,
        attention_mode='prca',
        num_heads=4,
        post_mamba_se=False,
    ):
        super(ImprovedSpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num
        self.attention = build_attention_module(
            attention_mode=attention_mode,
            dim=self.channel_num,
            num_heads=num_heads,
            bias=True,
            num_scales=num_scales,
            num_layers=num_layers,
        )
        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )
        self.post_se = PostMambaSE(self.channel_num) if post_mamba_se else nn.Identity()

    def padding_feature(self, x):
        b, c, h, w = x.shape
        if c < self.channel_num:
            pad_c = self.channel_num - c
            pad_features = torch.zeros((b, pad_c, h, w), device=x.device, dtype=x.dtype)
            return torch.cat([x, pad_features], dim=1)
        return x

    def forward(self, x):
        x_pad = self.padding_feature(x)
        x_re = self.attention(x_pad)

        b, c, h, w = x_re.shape
        x_re_flat = x_re.view(b * h * w, self.token_num, self.group_channel_num)
        x_recon = self.mamba(x_re_flat)

        x_recon = x_recon.view(b, c, h, w)
        x_recon = self.post_se(x_recon)
        x_recon = self.proj(x_recon)
        x_recon = x_recon[:, :x.shape[1]]
        return x_recon + x if self.use_residual else x_recon


class ImprovedSpaMamba(nn.Module):
    def __init__(
        self,
        channels,
        use_residual=True,
        group_num=4,
        token_num=4,
        num_scales=3,
        num_layers=2,
        attention_mode='prca',
        spatial_mode='baseline',
        num_heads=4,
        post_mamba_se=False,
    ):
        super(ImprovedSpaMamba, self).__init__()
        self.use_residual = use_residual
        self.attention = build_attention_module(
            attention_mode=attention_mode,
            dim=channels,
            num_heads=num_heads,
            bias=True,
            num_scales=num_scales,
            num_layers=num_layers,
        )

        self.mamba = Mamba(
            d_model=channels,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        if spatial_mode == 'dwconv_mamba':
            self.spatial_prior = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )
        elif spatial_mode == 'baseline':
            self.spatial_prior = nn.Identity()
        else:
            raise ValueError(f'Unsupported spatial mode: {spatial_mode}')

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, channels),
            nn.SiLU()
        )
        self.post_se = PostMambaSE(channels) if post_mamba_se else nn.Identity()

    def forward(self, x):
        x_re = self.attention(x)
        x_re = self.spatial_prior(x_re)

        b, c, h, w = x_re.shape
        x_flat = x_re.view(b * h * w, 1, c)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(b, c, h, w)
        x_recon = self.post_se(x_recon)
        x_recon = self.proj(x_recon)

        return x_recon + x if self.use_residual else x_recon

class ImprovedBothMamba(nn.Module):
    def __init__(
        self,
        channels,
        token_num,
        use_residual,
        group_num=4,
        attention_mode='prca',
        spatial_mode='baseline',
        fusion_mode='attention',
        num_scales=3,
        num_layers=2,
        num_heads=4,
        post_mamba_se=False,
    ):
        super(ImprovedBothMamba, self).__init__()
        self.use_residual = use_residual

        self.spa_mamba = ImprovedSpaMamba(
            channels,
            use_residual=use_residual,
            group_num=group_num,
            token_num=token_num,
            num_scales=num_scales,
            num_layers=num_layers,
            attention_mode=attention_mode,
            spatial_mode=spatial_mode,
            num_heads=num_heads,
            post_mamba_se=post_mamba_se,
        )
        self.spe_mamba = ImprovedSpeMamba(
            channels,
            token_num=token_num,
            use_residual=use_residual,
            group_num=group_num,
            num_scales=num_scales,
            num_layers=num_layers,
            attention_mode=attention_mode,
            num_heads=num_heads,
            post_mamba_se=post_mamba_se,
        )

        if fusion_mode == 'attention':
            self.fusion = AttentionFusion(channels)
        elif fusion_mode == 'cross_gate':
            self.fusion = CrossGateFusion(channels)
        else:
            raise ValueError(f'Unsupported fusion mode: {fusion_mode}')

    def forward(self, x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        fusion_x = self.fusion(spa_x, spe_x)
        return fusion_x + x if self.use_residual else fusion_x

class ImprovedMambaHSI(nn.Module):
    def __init__(
        self,
        in_channels=128,
        hidden_dim=64,
        num_classes=10,
        use_residual=True,
        mamba_type='both',
        token_num=4,
        group_num=4,
        attention_mode='prca',
        shared_attention_mode='none',
        spatial_mode='baseline',
        fusion_mode='attention',
        dynamic_conv_mode='dynamic',
        cls_hidden_dim=128,
        num_scales=3,
        num_layers=2,
        num_heads=4,
        post_mamba_se=False,
    ):
        super(ImprovedMambaHSI, self).__init__()
        self.mamba_type = mamba_type

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )
        self.shared_attention = build_attention_module(
            attention_mode=shared_attention_mode,
            dim=hidden_dim,
            num_heads=num_heads,
            bias=True,
            num_scales=num_scales,
            num_layers=num_layers,
        )

        if mamba_type == 'spa':
            self.mamba = nn.Sequential(
                ImprovedSpaMamba(
                    hidden_dim,
                    use_residual=use_residual,
                    group_num=group_num,
                    token_num=token_num,
                    num_scales=num_scales,
                    num_layers=num_layers,
                    attention_mode=attention_mode,
                    spatial_mode=spatial_mode,
                    num_heads=num_heads,
                    post_mamba_se=post_mamba_se,
                ),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            )
        elif mamba_type == 'spe':
            self.mamba = nn.Sequential(
                ImprovedSpeMamba(
                    hidden_dim,
                    token_num=token_num,
                    use_residual=use_residual,
                    group_num=group_num,
                    num_scales=num_scales,
                    num_layers=num_layers,
                    attention_mode=attention_mode,
                    num_heads=num_heads,
                    post_mamba_se=post_mamba_se,
                ),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            )
        elif mamba_type == 'both':
            self.mamba = nn.Sequential(
                ImprovedBothMamba(
                    hidden_dim,
                    token_num=token_num,
                    use_residual=use_residual,
                    group_num=group_num,
                    attention_mode=attention_mode,
                    spatial_mode=spatial_mode,
                    fusion_mode=fusion_mode,
                    num_scales=num_scales,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    post_mamba_se=post_mamba_se,
                ),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            )
        else:
            raise ValueError(f'Unsupported mamba type: {mamba_type}')

        self.dynamic_conv = build_dynamic_conv(dynamic_conv_mode, hidden_dim)
        self.cls_head = build_cls_head(hidden_dim, num_classes, cls_hidden_dim, group_num)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.shared_attention(x)
        x = self.mamba(x)
        x = self.dynamic_conv(x)
        logits = self.cls_head(x)
        return logits


