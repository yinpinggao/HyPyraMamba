import math
import torch
from torch import nn
from mamba_ssm import Mamba
from torch.cuda.amp import autocast
import torch.nn.functional as F
from einops import rearrange
import typing as t
from torch import nn, einsum

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


class ImprovedSpeMamba(nn.Module):
    def __init__(self, channels, token_num=4, use_residual=True, group_num=4, num_scales=3, num_layers=2):
        super(ImprovedSpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual
        # Set group_channel_num based on token_num and channels
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num
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
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self, x):
        # Apply padding to the input if necessary
        x_pad = self.padding_feature(x)
        # Apply PyramidRefinedChannelAttention directly to the input tensor
        x_re = self.pyramid_refined_attention(x_pad)

        # Flatten the input for Mamba
        B, C, H, W = x_re.shape
        x_re_flat = x_re.view(B * H * W, self.token_num, self.group_channel_num)  # Flatten for Mamba
        # Add first-order spectral differences before Mamba without changing dimensions.
        x_diff = torch.diff(x_re_flat, n=1, dim=1)
        x_diff = F.pad(x_diff, (0, 0, 0, 1))
        x_re_flat = x_re_flat + x_diff
        # Apply Mamba for feature learning
        x_recon = self.mamba(x_re_flat)

        # Reshape back to original dimensions
        x_recon = x_recon.view(B, C, H, W)
        # Apply the final projection to map the feature map to the output space
        x_recon = self.proj(x_recon)
        # If residual connection is enabled, add the input to the output
        return x_recon + x if self.use_residual else x_recon


class ImprovedSpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4, token_num=4,  num_scales=3, num_layers=2,
                 spatial_mode='baseline'):
        super(ImprovedSpaMamba, self).__init__()
        self.use_residual = use_residual
        self.token_num = token_num
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num
        self.spatial_mode = spatial_mode
        # Initialize PyramidRefinedChannelAttention
        self.pyramid_refined_attention = PyramidRefinedChannelAttention(
            dim=self.channel_num,  # Apply attention on channel_num
            num_heads=4,  # You can adjust num_heads as per your requirements
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

        self.multi_scale_conv = MultiScaleConv(channels, channels)

        # 添加 SCSA 模块
        self.scsa = SCSA(dim=channels, head_num=4, window_size=7)

        if spatial_mode == 'dwconv_mamba':
            self.spatial_prior = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )
        else:
            self.spatial_prior = nn.Identity()

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, channels),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.spatial_prior(x)
        # 首先应用多尺度卷积
        # x_re = self.multi_scale_conv(x)
        x_re = self.pyramid_refined_attention(x)
        # 然后应用 SCSA 模块进行空间-通道自注意力
        # x_re = self.scsa(x)  # Apply SCSA (Spatial-Channel Self Attention)

        # 将数据展平并通过 Mamba
        B, C, H, W = x_re.shape
        x_flat = x_re.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_flat = self.mamba(x_flat)

        # 重塑回原始形状
        x_recon = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # 应用最后的投影层
        x_recon = self.proj(x_recon)

        return x_recon + x if self.use_residual else x_recon

class CollaborativeFusion(nn.Module):
    def __init__(self, channels, reduction=4, group_num=4):
        super(CollaborativeFusion, self).__init__()
        mid = max(channels // reduction, 8)

        self.spatial_mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 2, kernel_size=1)
        )

        self.channel_spa = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.channel_spe = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GroupNorm(group_num, channels),
            nn.SiLU()
        )

    def forward(self, spa_x, spe_x, identity):
        fusion = torch.cat([spa_x, spe_x], dim=1)

        spatial_weights = self.spatial_mix(fusion)
        w_spa, w_spe = torch.chunk(spatial_weights, 2, dim=1)
        w_spa = torch.sigmoid(w_spa)
        w_spe = torch.sigmoid(w_spe)

        spa_s = spa_x + w_spe * spe_x
        spe_s = spe_x + w_spa * spa_x

        spa_c = spa_s + self.channel_spe(spe_s) * spe_s
        spe_c = spe_s + self.channel_spa(spa_s) * spa_s

        out = self.out_proj(torch.cat([spa_c, spe_c], dim=1))
        return out + identity

class ImprovedBothMamba(nn.Module):
    def __init__(self, channels, token_num, use_residual, group_num=4, spatial_mode='baseline'):
        super(ImprovedBothMamba, self).__init__()
        self.use_residual = use_residual

        self.spa_mamba = ImprovedSpaMamba(
            channels,
            use_residual=use_residual,
            group_num=group_num,
            spatial_mode=spatial_mode
        )
        self.spe_mamba = ImprovedSpeMamba(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)
        self.fusion = CollaborativeFusion(channels, reduction=4, group_num=group_num)

    def forward(self, x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        fusion_x = self.fusion(spa_x, spe_x, x)
        return fusion_x

class ImprovedMambaHSI(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, use_residual=True, mamba_type='both',
                 token_num=4, group_num=4, spatial_mode='baseline'):
        super(ImprovedMambaHSI, self).__init__()
        self.mamba_type = mamba_type

        # Patch Embedding Layer
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        # Choose between different Mamba modules
        if mamba_type == 'spa':
            self.mamba = nn.Sequential(
                ImprovedSpaMamba(
                    hidden_dim,
                    use_residual=use_residual,
                    group_num=group_num,
                    spatial_mode=spatial_mode
                ),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            )
        elif mamba_type == 'spe':
            self.mamba = nn.Sequential(
                ImprovedSpeMamba(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            )
        elif mamba_type == 'both':
            self.mamba = nn.Sequential(
                ImprovedBothMamba(
                    hidden_dim,
                    token_num=token_num,
                    use_residual=use_residual,
                    group_num=group_num,
                    spatial_mode=spatial_mode
                ),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            )

        # Ablation: remove DynamicConvBlock while keeping the rest unchanged.
        self.dynamic_conv = nn.Identity()

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.mamba(x)    # 下采样到 H/2, W/2
        logits = self.cls_head(x) # [B, num_classes, H/2, W/2]
        # logits = self.upsample(logits) # [B, num_classes, H, W]
        return logits


