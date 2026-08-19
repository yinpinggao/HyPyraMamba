# `MambaHSI.py` 源码逐行解读：对照 PyS²CF-Mamba 论文模型

> 源码文件：`/data2/gyp/HyPyraMamba/HypraMamba/model/MambaHSI.py`<br>
> 当前源码长度：613行<br>
> 本文对照版本：`168cf0cd259b233e907b7c5fa009ee6277ffee9a`之后的当前工作区文件<br>
> 训练入口：`/data2/gyp/HyPyraMamba/HypraMamba/train.py`<br>
> 解读日期：2026-08-11

---

## 0. 这份逐行解读怎样使用

本文不是只解释“每个模块大概做什么”，而是按照当前`MambaHSI.py`的真实行号，逐段解释：

1. 这一行Python代码做了什么；
2. 创建了什么参数或中间张量；
3. 张量形状如何变化；
4. 对应论文中的哪个模块或公式；
5. 当前训练入口实际传入什么配置；
6. 有哪些容易误解、被老师追问或值得修正的地方。

“逐行”采用**逻辑行组**解释：连续多行只是同一个函数调用的换行书写时，会作为一个整体解释；所有实际执行语句、判断、参数和返回路径都会覆盖，空行和纯排版括号不单独占一项。

建议阅读顺序：

- 第1～3节：先理解完整模型和真实配置；
- 第4～12节：对照源码逐行阅读；
- 第13节：看一次完整前向传播；
- 第14节：看论文名词与代码类名映射；
- 第15节：准备老师从源码角度的追问。

---

## 1. 先把论文模块和代码类名对上

| 论文名称 | 代码类/成员 | 源码行号 |
|---|---|---:|
| PyS²CF-Mamba | `ImprovedMambaHSI` | 512–613 |
| 共享Embedding | `patch_embedding` | 543–547 |
| 双分支主块 | `ImprovedBothMamba` | 423–509 |
| LPPS-Mamba | `ImprovedSpaMamba` | 333–391 |
| LSP | `LightSpatialPrior` | 302–330 |
| PRCA | `PyramidRefinedChannelAttention` | 184–233 |
| 单尺度通道注意力 | `PyramidAttention` | 109–182 |
| Spatial Mamba | `ImprovedSpaMamba.mamba` | 356–361 |
| DGS-Mamba | `ImprovedSpeMamba` | 236–299 |
| Spectral Mamba | `ImprovedSpeMamba.mamba` | 250–255 |
| 通道级竞争融合 | `CompetitiveFusion` | 394–420 |
| 融合后外层残差 | `_apply_outer_residual` | 484–489 |
| 池化与预测头 | `pool`、`cls_head` | 571–591 |

代码中没有名为`PyS2CFMamba`、`LPPSMamba`或`DGSMamba`的类。论文命名是在现有`Improved*`类结构基础上抽象出来的。

---

## 2. 当前训练真正使用的模型，不是类默认模型

`ImprovedMambaHSI.__init__`第513～521行给出的类默认值包括：

- `in_channels=128`；
- `hidden_dim=64`；
- `num_classes=10`；
- `pyramid_dilation=(2,3)`；
- `spectral_diff_alpha=1.0`。

但`train.py`第218～300行和第444～468行会覆盖这些值。论文主线/QUH GROUP2当前实际配置是：

| 参数 | 训练实际值 | 对源码的影响 |
|---|---:|---|
| `in_channels` | 30 | PCA后输入30维 |
| `hidden_dim` | 128 | 共享特征、空间分支宽度 |
| `num_classes` | 数据集类别数 | QY 6、LK 9、TDW 18 |
| `token_num` | 4 | 光谱序列长度为4 |
| `group_num` | 4 | GroupNorm分4组 |
| `pyramid_dilation` | `"3"` | 实际只启用dilation 3 |
| `ablation` | `full` | 两分支和竞争融合都启用 |
| `use_residual` | `True` | 分支内残差和外残差启用 |
| `outer_residual_mode` | `standard` | 输出加共享特征$F_0$ |
| `spectral_diff_alpha` | 0.5 | 相邻潜在通道等权插值 |
| `spectral_fusion_scale` | 1.0 | 不向空间分支回缩 |
| `pool_size` | 2 | 模型输出半分辨率logits |
| `high_res_skip` | `none` | 不启用池化后skip |
| `cls_head_dim` | 128 | 分类头中间通道 |
| PRCA scales/layers/heads | 3/2/4 | 每尺度实际3个attention，共9个 |
| Spatial Mamba | 16/4/2 | state/conv/expand |
| Spectral Mamba | 16/4/2 | state/conv/expand |

答辩时如果只背类定义中的`hidden_dim=64`，会与真实训练模型不符。

---

## 3. 完整模块树与前向数据流

### 3.1 模块树

```text
ImprovedMambaHSI
├── patch_embedding
│   ├── Conv2d(30 → 128, 1×1)
│   ├── GroupNorm(4, 128)
│   └── SiLU
├── mamba_block: ImprovedBothMamba
│   ├── spa_mamba: ImprovedSpaMamba
│   │   ├── spatial_prior: LightSpatialPrior
│   │   ├── pyramid_refined_attention: PRCA
│   │   │   ├── 3个基础PyramidAttention
│   │   │   └── 每尺度2个refinement，共6个
│   │   ├── Mamba(d_model=128)
│   │   └── GroupNorm + SiLU
│   ├── spe_mamba: ImprovedSpeMamba
│   │   ├── 一阶潜在通道差分
│   │   ├── reshape为[BHW, 4, 32]
│   │   ├── Mamba(d_model=32)
│   │   └── GroupNorm + SiLU
│   └── fusion: CompetitiveFusion
├── AvgPool2d(2)
└── cls_head
    ├── Conv2d(128 → 128, 1×1)
    ├── GroupNorm
    ├── SiLU
    └── Conv2d(128 → K, 1×1)
```

### 3.2 以512×512 tile为例

$$
[1,30,512,512]
\xrightarrow{\text{Embedding}}
[1,128,512,512].
$$

空间分支：

$$
[1,128,512,512]
\rightarrow[1,262144,128]
\rightarrow[1,128,512,512].
$$

光谱分支：

$$
[1,128,512,512]
\rightarrow[262144,4,32]
\rightarrow[1,128,512,512].
$$

融合、池化和分类：

$$
[1,128,512,512]
\rightarrow[1,128,256,256]
\rightarrow[1,K,256,256].
$$

最终恢复到512×512不是在`MambaHSI.py`内部完成，而是在`train.py`或`utils/Loss.py`中插值。

---

## 4. 第1～24行：依赖导入和消融开关

### 4.1 第1～5行：导入依赖

```python
1  import math
2  import torch
3  from torch import nn
4  from einops import rearrange
5  from mamba_ssm import Mamba
```

| 行号 | 解读 |
|---:|---|
| 1 | 导入Python数学库。当前文件只在第247、341行使用`math.ceil`计算每个光谱组的通道数。 |
| 2 | 导入PyTorch主包，用于张量、Softmax、拼接、插值和函数式操作。 |
| 3 | 导入`torch.nn`并命名为`nn`，所有网络层和`nn.Module`都从这里创建。 |
| 4 | 导入`einops.rearrange`，在PRCA中把通道拆成多头并把空间维展平。它比手写`view/permute`更直观。 |
| 5 | 从外部`mamba_ssm`库导入Mamba。Selective Scan、状态空间参数和CUDA实现都不在本文件中；本文件只决定输入序列怎样构造。 |

第5行是理解源码的关键边界：

> 你实现的是“如何把高光谱空间和光谱特征组织成Mamba序列”，不是从零实现Mamba内部扫描算子。

### 4.2 第8～16行：合法消融名称

```python
VALID_ABLATIONS = {
    'full',
    'wo_lpps',
    'wo_dgs',
    'wo_lsp',
    'wo_prca',
    'wo_diff',
    'wo_competitive',
}
```

| 值 | 真实含义 |
|---|---|
| `full` | 空间分支、光谱分支、差分和竞争融合全部启用 |
| `wo_lpps` | 删除整个空间分支 |
| `wo_dgs` | 删除整个光谱分支 |
| `wo_lsp` | 空间分支保留PRCA和Spatial Mamba，只删LSP |
| `wo_prca` | 空间分支保留LSP和Spatial Mamba，只删PRCA |
| `wo_diff` | 光谱Mamba保留，只不做差分 |
| `wo_competitive` | 双分支都保留，用等权平均代替竞争融合 |

这里的集合用于白名单校验，防止命令行拼写错误悄悄生成错误模型。

### 4.3 第17～24行：把消融名称按功能分类

```python
17 SPATIAL_BRANCH_DISABLED_ABLATIONS = {'wo_lpps'}
18 SPECTRAL_BRANCH_DISABLED_ABLATIONS = {'wo_dgs'}
19 SPATIAL_PRIOR_DISABLED_ABLATIONS = {'wo_lsp'}
20 SPATIAL_PRCA_DISABLED_ABLATIONS = {'wo_prca'}
21 SPECTRAL_DIFF_DISABLED_ABLATIONS = {'wo_diff'}
22 COMPETITIVE_FUSION_DISABLED_ABLATIONS = {'wo_competitive'}
23 VALID_OUTER_RESIDUAL_MODES = {'standard', 'no_outer', 'scaled'}
24 VALID_HIGH_RES_SKIP_MODES = {'none', 'patch', 'pre_pool'}
```

这些集合让后面的模块通过统一判断决定是否创建某个分支。例如：

```python
self.use_spatial_branch = self.ablation not in SPATIAL_BRANCH_DISABLED_ABLATIONS
```

比到处写`ablation != 'wo_lpps'`更容易维护。

第23行三种外残差：

- `standard`：$x+\mathrm{block}(x)$；
- `no_outer`：只返回$\mathrm{block}(x)$；
- `scaled`：$x+\alpha_{\mathrm{outer}}\mathrm{block}(x)$。

第24行三种高分辨率skip：

- `none`：论文主线，不启用；
- `patch`：池化后加入Embedding特征；
- `pre_pool`：池化后加入双分支块输出。

---

## 5. 第27～107行：输入参数校验

### 5.1 第27～30行：消融名校验

```python
def _validate_ablation(ablation):
    if ablation not in VALID_ABLATIONS:
        raise ValueError('Unsupported ablation: {}'.format(ablation))
    return ablation
```

逐行：

- 第27行定义内部辅助函数，前导下划线表示仅供本模块内部使用。
- 第28行判断传入字符串是否在合法集合中。
- 第29行非法时立即抛出异常，而不是默认回退到`full`。
- 第30行返回经过验证的原值，便于写成`self.ablation = _validate_ablation(ablation)`。

### 5.2 第33～42行：残差模式和skip模式校验

`_validate_outer_residual_mode`和`_validate_high_res_skip_mode`逻辑相同：

1. 检查字符串；
2. 非法立即报错；
3. 返回合法值。

这样可以在模型构造阶段就发现配置问题，而不是训练数小时后才触发某个分支。

### 5.3 第45～58行：统一dilation输入格式

```python
def _normalize_dilations(dilation):
    if isinstance(dilation, str):
        dilations = tuple(int(value.strip()) for value in dilation.split(',') if value.strip())
    elif isinstance(dilation, int):
        dilations = (dilation,)
    else:
        dilations = tuple(int(value) for value in dilation)
```

第46～51行允许三种写法：

| 输入 | 输出 |
|---|---|
| `"3"` | `(3,)` |
| `"2,3"` | `(2,3)` |
| `3` | `(3,)` |
| `[2,3]`或`(2,3)` | `(2,3)` |

第47行细节：

- `split(',')`把字符串按逗号拆开；
- `strip()`删除空格；
- `if value.strip()`跳过空字段；
- `int(...)`转为整数；
- 最终用`tuple`固定下来。

第53～56行：

- 空tuple说明没有任何dilation，报错；
- 任一值小于1，报错；
- dilation必须是正整数。

第58行返回标准化后的tuple。

当前训练入口传入字符串`"3"`，最终得到：

```python
self.dilations = (3,)
```

因此不会进入多dilation融合分支。

### 5.4 第61～64行：正整数校验

```python
if int(value) != value or int(value) <= 0:
```

它同时排除：

- 0；
- 负数；
- 2.5之类非整数浮点数。

合法时转成Python `int`返回。

注意：布尔值在Python中是`int`的子类，`True`理论上可能通过并变为1，但训练入口不会把这些结构参数传成布尔值。

### 5.5 第67～82行：模型配置校验函数签名

这里列出所有必须是正整数、且影响形状的参数：

- 隐藏维度；
- 光谱token数；
- GroupNorm组数；
- PRCA头数、尺度数、层数；
- pool大小；
- 分类头宽度；
- LSP reduction；
- 两个Mamba的state、conv和expand。

它不返回配置字典，而是通过“非法就抛异常”完成校验。

### 5.6 第83～97行：逐个调用正整数校验

每一行都把对应参数送入`_validate_positive_int`。局部变量虽然被转换为`int`，但函数没有返回这些变量，因此外部参数本身不会被替换。

在当前训练入口中参数本来就是整数，所以没有影响。更严谨的写法可以返回标准化后的配置，但当前功能足够完成合法性检查。

### 5.7 第99～106行：整除约束

```python
if hidden_dim % group_num != 0:
    raise ValueError(...)
```

保证`GroupNorm(group_num, hidden_dim)`可以把通道均匀分组。

```python
if cls_head_dim % group_num != 0:
```

保证分类头的GroupNorm合法。

```python
if hidden_dim % token_num != 0:
```

保证128个潜在通道可以无补零地分为4组，每组32维。

```python
if hidden_dim % prca_num_heads != 0:
```

保证PRCA的128通道可以分为4个head，每头32通道。

当前配置：

$$
128\bmod4=0.
$$

四个约束全部满足。

一个重要源码细节：

> `ImprovedSpeMamba`虽然实现了padding，但通过顶层`ImprovedMambaHSI`构造时，`hidden_dim % token_num == 0`被强制成立，所以论文主配置实际上从不触发padding。

---

## 6. 第109～182行：`PyramidAttention`逐行解读

### 6.1 这个类究竟做什么

它不是空间自注意力，而是：

1. 先用1×1卷积生成Q、K、V；
2. 用膨胀depthwise 3×3卷积注入局部空间结构；
3. 把每个head的通道作为“被注意的对象”；
4. 在空间维上计算通道相关；
5. 得到每头$32\times32$通道注意力矩阵。

当前输入：

$$
x\in\mathbb R^{B\times128\times H\times W}.
$$

### 6.2 第109～116行：类定义、头数和温度

```python
class PyramidAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, dilation=2):
        super(PyramidAttention, self).__init__()
        self.num_heads = num_heads
        self.dilations = _normalize_dilations(dilation)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
```

逐行：

- 第109行声明PyTorch模块。
- 第110行接收通道宽度、头数、卷积bias和dilation。
- 第111行初始化父类，保证参数和子模块能被PyTorch注册。
- 第113行保存头数，后续`rearrange`依赖它拆分通道。
- 第114行把dilation标准化为tuple。
- 第116行创建每个head一个可学习温度，形状`[heads,1,1]`。

当前heads=4：

$$
\mathrm{temperature}\in\mathbb R^{4\times1\times1}.
$$

它会广播到注意力矩阵`[B,4,32,32]`，控制每个head Softmax前logit的锐利程度。

### 6.3 第118～120行：QKV的1×1投影

```python
self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
```

当前：

$$
[B,128,H,W]\rightarrow[B,384,H,W].
$$

1×1卷积只混合通道，不改变空间尺寸。输出384通道，稍后均分为Q、K、V各128通道。

### 6.4 第121～133行：每个dilation一个depthwise卷积分支

```python
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
```

逐项解释：

- `ModuleList`保证多个卷积分支都被注册为模型子模块；
- 输入输出都是384通道；
- kernel=3；
- stride=1，不下采样；
- dilation由当前分支决定；
- padding=dilation，使空间尺寸保持不变；
- groups=384，表示每个Q/K/V通道独立做depthwise卷积；
- 它不在此处做通道混合，通道混合已由前面的1×1卷积完成。

当前dilation=3时，有效卷积核感受野为：

$$
k_{\mathrm{eff}}=3+(3-1)(3-1)=7.
$$

即每个depthwise卷积具有7×7的稀疏有效感受野。

### 6.5 第134～140行：多dilation时的全局可学习融合

只有`len(self.dilations)>1`才执行：

```python
dilation_logits = torch.zeros(len(self.dilations))
preferred_index = self.dilations.index(3) if 3 in self.dilations else len(self.dilations) - 1
dilation_logits[preferred_index] = 2.0
self.dilation_logits = nn.Parameter(dilation_logits)
```

含义：

- 每个dilation有一个全局标量logit；
- 初始偏向dilation 3；
- forward中通过Softmax转成权重；
- 权重不是每张图动态生成，也不是逐通道或逐像素权重；
- 每个`PyramidAttention`实例拥有自己的dilation logits。

若只有一个dilation，第140行：

```python
self.register_parameter('dilation_logits', None)
```

显式注册一个值为None的参数名，使forward可以统一检查，同时不会加入可训练参数。

当前训练配置只有`(3,)`，所以多dilation逻辑完全不启用。

### 6.6 第142行：输出投影

```python
self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
```

注意力输出仍为128通道，1×1卷积进一步混合不同head恢复后的通道。

### 6.7 第144～150行：forward开始和QKV生成

```python
def forward(self, x):
    b, c, h, w = x.shape
    qkv_base = self.qkv(x)
    if self.dilation_logits is None:
        qkv = self.qkv_dwconvs[0](qkv_base)
```

- 第145行读取输入尺寸；`b`和`c`后面没有直接使用，`h,w`用于恢复形状。
- 第148行得到`qkv_base:[B,384,H,W]`。
- 当前单dilation路径直接使用唯一depthwise卷积。

### 6.8 第151～156行：多dilation加权路径

```python
weights = torch.softmax(self.dilation_logits, dim=0)
qkv = None
for weight, qkv_dwconv in zip(weights, self.qkv_dwconvs):
    branch_qkv = weight * qkv_dwconv(qkv_base)
    qkv = branch_qkv if qkv is None else qkv + branch_qkv
```

逻辑是：

$$
\mathrm{QKV}
=\sum_d\pi_d\,\mathrm{DWConv}_d(\mathrm{QKV}_{base}),
\qquad
\sum_d\pi_d=1.
$$

`qkv=None`只是为了在第一次循环时不需要预先创建全零张量。

### 6.9 第157～163行：拆QKV并拆多头

```python
q, k, v = qkv.chunk(3, dim=1)
```

把384通道均分：

$$
Q,K,V\in\mathbb R^{B\times128\times H\times W}.
$$

接着：

```python
q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
```

当前4头：

$$
[B,128,H,W]\rightarrow[B,4,32,HW].
$$

K、V完全相同。

这里的`c`是每头通道数32，不是输入总通道128。

### 6.10 第165～167行：Q、K沿空间维归一化

```python
q = torch.nn.functional.normalize(q, dim=-1)
k = torch.nn.functional.normalize(k, dim=-1)
```

最后一维是`HW`，因此每个head、每个子通道的整幅空间响应被L2归一化。

这意味着后面的点积更接近不同通道空间响应之间的余弦相似度。

### 6.11 第169～172行：生成通道注意力矩阵

```python
attn = (q @ k.transpose(-2, -1)) * self.temperature
attn = attn.softmax(dim=-1)
```

形状：

$$
[B,4,32,HW]\times[B,4,HW,32]
\rightarrow[B,4,32,32].
$$

所以注意力对象是每个head中的32个通道，而不是$HW$个像素。

Softmax沿最后一个32维执行，表示每个query通道对所有key通道的权重和为1。

复杂度相对空间注意力：

- 当前核心相关矩阵：$O(B\cdot heads\cdot32^2\cdot HW)$；
- 空间全注意力：$O(B\cdot(HW)^2\cdot C)$。

### 6.12 第174～180行：注意力作用于V并投影

```python
out = attn @ v
```

形状：

$$
[B,4,32,32]\times[B,4,32,HW]
\rightarrow[B,4,32,HW].
$$

第177行恢复：

$$
[B,4,32,HW]\rightarrow[B,128,H,W].
$$

第180行再经过1×1输出投影。

### 6.13 第182行：返回

```python
return out
```

该类内部没有：

- 残差连接；
- GroupNorm；
- 激活函数；
- Dropout。

它只是通道注意力变换。残差要到后面的Spatial Mamba输出处才加入。

---

## 7. 第184～233行：`PyramidRefinedChannelAttention`逐行解读

### 7.1 第184～200行：构造三尺度、九个Attention

构造函数参数：

```python
def __init__(self, dim, num_heads, bias, num_scales=3, num_layers=2, dilation=2):
```

当前实际：

- dim=128；
- heads=4；
- scales=3；
- layers=2；
- dilation=3。

第189～191行：

```python
self.attention_modules = nn.ModuleList([
    PyramidAttention(...) for _ in range(num_scales)
])
```

为每个尺度创建1个基础`PyramidAttention`，共3个。

第194～197行：

```python
self.attention_layers = nn.ModuleList([
    nn.ModuleList([PyramidAttention(...) for _ in range(num_layers)])
    for _ in range(num_scales)
])
```

为每个尺度再创建2个refinement block：

$$
3\text{ scales}\times2=6.
$$

总数：

$$
3+6=9\text{ 个独立PyramidAttention}.
$$

所有9个block参数独立，不共享。

第200行：

```python
self.project_out = nn.Conv2d(dim * num_scales, dim, kernel_size=1, bias=bias)
```

三尺度输出拼接后通道为384，再压回128。

### 7.2 第202～205行：forward初始化

```python
b, c, h, w = x.shape
outputs = []
```

- 保存原始空间尺寸；
- 创建列表收集三个尺度输出；
- `b,c`此后不直接使用。

### 7.3 第207～212行：循环和尺度构造

第207行遍历三个基础attention。

第209～210行：

- i=0时不降采样；
- 输入为$H\times W$。

第211～212行：

```python
scaled_input = avg_pool2d(x, kernel_size=2 ** i, stride=2 ** i)
```

当前三个尺度：

| i | kernel/stride | 输出尺寸 |
|---:|---:|---|
| 0 | 不池化 | $H\times W$ |
| 1 | 2 | $\lfloor H/2\rfloor\times\lfloor W/2\rfloor$ |
| 2 | 4 | $\lfloor H/4\rfloor\times\lfloor W/4\rfloor$ |

注意：每个低尺度都直接从原始输入$x$池化，不是上一个尺度继续池化。

### 7.4 第214～219行：每尺度连续执行3个attention

```python
output = attention_module(scaled_input)
for layer in self.attention_layers[i]:
    output = layer(output)
```

实际是：

$$
X_s
\rightarrow PA_{s,0}
\rightarrow PA_{s,1}
\rightarrow PA_{s,2}.
$$

三个block之间没有显式残差和归一化。`num_layers=2`表示“基础block之后再加2层”，不是每尺度总共2层。

### 7.5 第221～225行：上采样并收集

低尺度输出通过：

```python
interpolate(..., size=(h,w), mode='bilinear', align_corners=False)
```

恢复到原始分辨率。

第225行把每个尺度结果加入`outputs`。

### 7.6 第227～233行：拼接、压缩、返回

```python
out = torch.cat(outputs, dim=1)
```

形状：

$$
3\times[B,128,H,W]
\rightarrow[B,384,H,W].
$$

第231行1×1卷积压回：

$$
[B,384,H,W]\rightarrow[B,128,H,W].
$$

第233行返回。

### 7.7 PRCA的准确机制总结

PRCA不是“在三个尺度各做一次注意力”，而是：

> 三个尺度，每个尺度连续做三个独立通道注意力block，低尺度上采样后拼接，再用1×1卷积融合。

当前PRCA参数量为678,308，占18类完整模型约76.8%，是模型参数主体。

## 8. 第236～299行：`ImprovedSpeMamba`，即DGS-Mamba逐行解读

### 8.1 先说结论

这个类对每个空间像素独立执行：

$$
[C]
\xrightarrow{\text{潜在通道差分}}
[C]
\xrightarrow{\text{分成4组}}
[4,32]
\xrightarrow{\text{Mamba沿4步扫描}}
[4,32]
\xrightarrow{\text{恢复}}
[C].
$$

空间位置之间在这个类中不交互；空间建模由`ImprovedSpaMamba`负责。

### 8.2 第236～245行：构造函数和开关

```python
class ImprovedSpeMamba(nn.Module):
    def __init__(self, channels, token_num=4, use_residual=True, group_num=4,
                 ablation='full', spectral_diff_alpha=1.0, mamba_d_state=16,
                 mamba_d_conv=4, mamba_expand=2):
```

参数含义：

| 参数 | 含义 | 当前训练值 |
|---|---|---:|
| `channels` | 输入潜在通道数 | 128 |
| `token_num` | 光谱序列长度/分组数 | 4 |
| `use_residual` | 是否加光谱分支残差 | True |
| `group_num` | 输出GroupNorm组数 | 4 |
| `ablation` | 消融类型 | full |
| `spectral_diff_alpha` | 差分系数 | 0.5 |
| `mamba_d_state` | SSM状态维度 | 16 |
| `mamba_d_conv` | Mamba局部卷积宽度 | 4 |
| `mamba_expand` | Mamba内部扩展倍数 | 2 |

第240行初始化父类。

第241行：

```python
self.ablation = _validate_ablation(ablation)
```

保存经过白名单验证的消融名。

第242～243行保存token数和残差开关。

第244行：

```python
self.use_diff_enhance = self.ablation not in SPECTRAL_DIFF_DISABLED_ABLATIONS
```

只有`wo_diff`会关闭差分。`wo_dgs`时整个`ImprovedSpeMamba`都不会被创建，所以不会运行到这里。

第245行把差分系数转为float。

### 8.3 第246～248行：计算token维度和补齐通道数

```python
self.group_channel_num = math.ceil(channels / token_num)
self.channel_num = self.token_num * self.group_channel_num
```

一般情况：

$$
G=\left\lceil\frac{C}{T}\right\rceil,
\qquad
C_{\mathrm{pad}}=T\cdot G.
$$

当前：

$$
G=\lceil128/4\rceil=32,
\qquad
C_{\mathrm{pad}}=4\times32=128.
$$

所以：

- `group_channel_num=32`；
- `channel_num=128`；
- 无需补零。

这里变量命名容易混淆：

- `token_num`是序列长度4；
- `group_channel_num`是每个token的特征维32；
- `channel_num`是补齐后的总通道128。

### 8.4 第249～255行：构造Spectral Mamba

```python
self.mamba = Mamba(
    d_model=self.group_channel_num,
    d_state=mamba_d_state,
    d_conv=mamba_d_conv,
    expand=mamba_expand,
)
```

Mamba约定输入形状：

$$
[\text{batch},\text{sequence length},d_{\mathrm{model}}].
$$

因此这里的输入必须是：

$$
[BHW,4,32].
$$

不是：

- `[BHW,32,4]`；
- `[B,HW,128]`；
- 4个独立Mamba；
- 128步光谱扫描。

当前内部扩展维度：

$$
d_{\mathrm{inner}}=d_{\mathrm{model}}\times expand=32\times2=64.
$$

### 8.5 第256～260行：输出归一化和激活

```python
self.proj = nn.Sequential(
    nn.GroupNorm(group_num, self.channel_num),
    nn.SiLU()
)
```

这里`proj`这个名字容易让人以为有卷积投影，但实际只有：

- GroupNorm；
- SiLU。

没有Linear或Conv。

当前：

$$
\mathrm{GN}(4,128).
$$

如果独立实例化`ImprovedSpeMamba`并使补齐后的`channel_num`不能被`group_num`整除，这里会构造失败。顶层模型的整除校验避免了论文主配置中的问题。

### 8.6 第262～270行：`padding_feature`

```python
B, C, H, W = x.shape
if C < self.channel_num:
    pad_c = self.channel_num - C
    pad_features = x.new_zeros((B, pad_c, H, W))
    cat_features = torch.cat([x, pad_features], dim=1)
    return cat_features
else:
    return x
```

逐行：

- 第263行读取形状；
- 第264行判断实际通道是否小于补齐通道；
- 第265行计算需要补多少个通道；
- 第266行用`x.new_zeros`创建与$x$相同device和dtype的零张量；
- 第267行沿通道维拼接；
- 第268行返回补齐特征；
- 若无需补齐，第270行原样返回。

为什么使用`x.new_zeros`而不是`torch.zeros`：

> 它会自动继承CUDA设备、float16/float32类型，避免device或dtype不匹配。

当前顶层配置强制`128 % 4 == 0`，因此这段属于通用防御代码，主线不会触发。

### 8.7 第272～275行：一阶潜在通道差分

```python
diff = x.new_zeros(x.shape)
diff[:, :-1, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
return x + self.spectral_diff_alpha * diff
```

第273行创建全零差分张量。

第274行对前$C-1$个通道赋值：

$$
\Delta F_c=F_{c+1}-F_c.
$$

由于左侧是`:-1`，最后一个通道未被写入，仍为0。

第275行：

$$
F_{\mathrm{diff},c}
=F_c+\alpha(F_{c+1}-F_c).
$$

当前$\alpha=0.5$：

$$
F_{\mathrm{diff},c}
=0.5F_c+0.5F_{c+1}.
$$

最后一个通道：

$$
F_{\mathrm{diff},C}=F_C.
$$

重要解释边界：

> 这里的$c$是PCA和1×1 embedding后的潜在通道，不是原始物理相邻波长。

### 8.8 第277～281行：forward开始

```python
x_diff = self.spectral_difference_enhance(x) if self.use_diff_enhance else x
x_re = self.padding_feature(x_diff)
```

`full`时：

1. 先得到差分引导特征`x_diff`；
2. 再检查是否要补零；
3. 当前形状保持`[B,128,H,W]`。

`wo_diff`时`x_diff=x`，但后面的分组Mamba、GN、SiLU和残差仍保留。

### 8.9 第283～290行：每个像素变成长度4的序列

```python
B, C, H, W = x_re.shape
origin_c = x.shape[1]
```

- `C`是补齐后的通道数；
- `origin_c`记录原始通道数，后面用于裁掉padding。

核心重排：

```python
x_re_flat = x_re.permute(0, 2, 3, 1).reshape(
    B * H * W,
    self.token_num,
    self.group_channel_num,
)
```

第一步：

$$
[B,C,H,W]\rightarrow[B,H,W,C].
$$

第二步：

$$
[B,H,W,128]\rightarrow[BHW,4,32].
$$

`reshape`按连续通道顺序分组：

- token 0：通道0～31；
- token 1：通道32～63；
- token 2：通道64～95；
- token 3：通道96～127。

没有通道打乱、聚类或可学习分组。

### 8.10 第291～292行：执行Spectral Mamba

```python
x_out = self.mamba(x_re_flat)
```

输入输出形状相同：

$$
[BHW,4,32]\rightarrow[BHW,4,32].
$$

对每个像素，Mamba沿4个token的顺序传播状态。不同像素被放在batch维，彼此不发生状态交互。

### 8.11 第294～297行：恢复二维、归一化和裁剪

```python
x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
```

恢复：

$$
[BHW,4,32]
\rightarrow[B,H,W,128]
\rightarrow[B,128,H,W].
$$

`contiguous()`保证permute后的张量在内存中连续，便于后续算子。

第297行：

```python
x_out = self.proj(x_out)[:, :origin_c, :, :]
```

执行GN+SiLU，再裁回原始通道数。

若发生padding，零通道会参与GroupNorm统计后才被裁掉；这可能影响真实通道。主配置无padding，所以不存在这一影响。

### 8.12 第298～299行：光谱残差

```python
return x_out + x_diff if self.use_residual else x_out
```

默认返回：

$$
F_{\mathrm{spe}}
=\operatorname{SiLU}(\operatorname{GN}(\operatorname{Mamba}(T)))
+F_{\mathrm{diff}}.
$$

论文式(7)写的是加$F_0$，当前代码加的是$F_{\mathrm{diff}}$。这是论文与代码的实质差异。

关闭`use_residual`时只返回Mamba变换输出，但差分仍可能发生并作为Mamba输入。

---

## 9. 第302～330行：`LightSpatialPrior`，即LSP逐行解读

### 9.1 第302～305行：类定义和中间通道

```python
class LightSpatialPrior(nn.Module):
    def __init__(self, channels, group_num=4, reduction=4):
        super(LightSpatialPrior, self).__init__()
        mid = max(channels // reduction, 8)
```

当前：

$$
mid=\max(128/4,8)=32.
$$

`max(...,8)`防止通道很小时门控瓶颈过窄。

### 9.2 第307～310行：depthwise 3×3局部特征

```python
self.dw = nn.Conv2d(
    channels, channels,
    kernel_size=3, padding=1, groups=channels
)
```

当前是128组depthwise卷积：

$$
[B,128,H,W]\rightarrow[B,128,H,W].
$$

每个通道独立提取局部纹理和边缘，不进行通道混合。

参数量：

$$
128\times3\times3+128\text{ bias}=1280.
$$

### 9.3 第312～317行：单通道空间门控

```python
self.spatial_gate = nn.Sequential(
    nn.Conv2d(channels, mid, kernel_size=1),
    nn.SiLU(),
    nn.Conv2d(mid, 1, kernel_size=1),
    nn.Sigmoid()
)
```

形状：

$$
[B,128,H,W]
\rightarrow[B,32,H,W]
\rightarrow[B,1,H,W].
$$

最后Sigmoid将门控限制到$(0,1)$。

这是一张所有通道共享的空间mask，不是`[B,128,H,W]`逐通道门控。

### 9.4 第319～321行：通道混合、归一化和激活

```python
self.pw = nn.Conv2d(channels, channels, kernel_size=1)
self.norm = nn.GroupNorm(group_num, channels)
self.act = nn.SiLU()
```

- `pw`在depthwise卷积之后进行通道混合；
- GroupNorm适合当前batch通常为1的整图/tile训练；
- SiLU提供平滑非线性。

### 9.5 第323～330行：forward逐行

```python
local_feat = self.dw(x)
```

得到每通道局部响应。

```python
gate = self.spatial_gate(x)
```

门控由原始输入$x$生成，而不是由`local_feat`生成。

```python
out = local_feat * gate
```

`gate:[B,1,H,W]`沿通道广播：

$$
out_{b,c,h,w}
=local_{b,c,h,w}\cdot gate_{b,1,h,w}.
$$

第327～329行：

```python
out = self.pw(out)
out = self.norm(out)
out = self.act(out)
```

完成通道混合、归一化和激活。

第330行：

```python
return out + x
```

LSP内部固定带残差：

$$
X_{\mathrm{prior}}=x+\mathrm{LSPTransform}(x).
$$

重要细节：

> 即使顶层`use_residual=False`，LSP这一条`out+x`仍然存在，因为LSP没有接收`use_residual`参数。

---

## 10. 第333～391行：`ImprovedSpaMamba`，即LPPS-Mamba逐行解读

### 10.1 第333～342行：参数和通道计算

构造函数接收：

- channels=128；
- residual；
- GroupNorm组数；
- token_num；
- PRCA尺度、层数、heads和dilation；
- LSP reduction；
- Spatial Mamba参数；
- ablation。

第338行校验并保存消融。

第339行保存残差开关。

第340～342行：

```python
self.token_num = token_num
self.group_channel_num = math.ceil(channels / token_num)
self.channel_num = self.token_num * self.group_channel_num
```

当前仍得到4、32、128。

但在空间分支里，`token_num`不用于Spatial Mamba序列化；Spatial Mamba直接使用`d_model=channels`。这些变量主要被用来指定PRCA的`dim=self.channel_num`。

顶层强制channels能被token_num整除，所以`channel_num==channels`。

若独立实例化该类并传入不能整除的channels，PRCA会按补齐后的`channel_num`构造，但forward没有对$x$补通道，可能产生输入通道不匹配。论文主配置不会触发。

### 10.2 第343～354行：是否创建PRCA

```python
self.use_prca = self.ablation not in SPATIAL_PRCA_DISABLED_ABLATIONS
```

只有`wo_prca`关闭。

启用时第345～352行创建PRCA：

- dim=128；
- heads=4；
- bias=True；
- scales=3；
- layers=2；
- dilation=3。

注意`bias=True`在这里硬编码，训练入口没有关闭PRCA卷积bias的参数。

关闭时：

```python
self.pyramid_refined_attention = None
```

forward根据None跳过。

### 10.3 第356～361行：构造Spatial Mamba

```python
self.mamba = Mamba(
    d_model=channels,
    d_state=mamba_d_state,
    d_conv=mamba_d_conv,
    expand=mamba_expand,
)
```

当前：

- d_model=128；
- d_state=16；
- d_conv=4；
- expand=2；
- 内部扩展维度256。

它要求输入：

$$
[B,\text{空间序列长度},128].
$$

### 10.4 第363～367行：是否创建LSP

只有`wo_lsp`关闭LSP。

启用时：

```python
LightSpatialPrior(128, group_num=4, reduction=4)
```

关闭时成员设为None，forward直接使用原输入。

### 10.5 第369～372行：Spatial Mamba输出处理

```python
self.proj = nn.Sequential(
    nn.GroupNorm(group_num, channels),
    nn.SiLU()
)
```

与光谱分支一样，`proj`实际不包含卷积，只做GN和SiLU。

### 10.6 第374～378行：LSP路径

```python
if self.spatial_prior is None:
    x_prior = x
else:
    x_prior = self.spatial_prior(x)
```

`full`时：

$$
x_{\mathrm{prior}}=x+\mathrm{LSPTransform}(x).
$$

`wo_lsp`时：

$$
x_{\mathrm{prior}}=x.
$$

### 10.7 第380～383行：PRCA路径

`wo_prca`时直接：

$$
x_{\mathrm{re}}=x_{\mathrm{prior}}.
$$

默认时：

$$
x_{\mathrm{re}}=\mathrm{PRCA}(x_{\mathrm{prior}}).
$$

PRCA本身没有把输入残差加回来，因此后续Spatial Mamba的残差基底仍是`x_prior`，而不是PRCA输入与输出在PRCA内部相加。

### 10.8 第384～386行：展开成空间序列并执行Mamba

```python
B, C, H, W = x_re.shape
x_flat = x_re.permute(0, 2, 3, 1).reshape(B, H * W, C)
x_flat = self.mamba(x_flat)
```

形状：

$$
[B,128,H,W]
\rightarrow[B,H,W,128]
\rightarrow[B,HW,128].
$$

排列顺序是PyTorch行优先：

1. 第一行从左到右；
2. 第二行从左到右；
3. 依次向下。

它是单向标准1D Mamba，不是四方向扫描或2D selective scan。

对512 tile：

$$
[1,262144,128].
$$

### 10.9 第388～390行：恢复二维并归一化

```python
x_out = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
x_out = self.proj(x_out)
```

恢复到`[B,128,H,W]`，再执行GN+SiLU。

这里没有调用`contiguous()`，但GroupNorm能够处理该非连续view；如果后续加入要求连续内存的自定义算子，可以考虑显式调用。

### 10.10 第391行：Spatial Mamba残差

```python
return x_out + x_prior if self.use_residual else x_out
```

默认：

$$
F_{\mathrm{spa}}
=\operatorname{SiLU}(\operatorname{GN}(\operatorname{Mamba}(\mathrm{PRCA}(X_{\mathrm{prior}}))))
+X_{\mathrm{prior}}.
$$

残差加的是LSP输出`x_prior`，不是裸共享特征$x$。

由于LSP内部已经有一次$x$残差，空间分支实际有嵌套残差。

---

## 11. 第394～420行：`CompetitiveFusion`逐行解读

### 11.1 第394～406行：两套独立分支打分器

```python
self.fc_spa = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(1),
    nn.Linear(channels, channels, bias=False),
)
```

空间分支：

$$
[B,128,H,W]
\rightarrow[B,128,1,1]
\rightarrow[B,128]
\rightarrow[B,128].
$$

光谱分支`fc_spe`结构相同，但参数独立。

每个Linear有：

$$
128\times128=16384
$$

个参数，两支合计32768。

### 11.2 第408～412行：输入断言

```python
assert spa_feat.dim() == 4 and spe_feat.dim() == 4
assert spa_feat.shape == spe_feat.shape
```

保证：

- 都是`[B,C,H,W]`；
- 形状完全相同。

这样后续逐元素加权才合法。

注意Python使用`-O`优化运行时可能移除assert。若需要生产级强校验，可以改成显式`if ...: raise ValueError`。

### 11.3 第414～416行：生成竞争权重

```python
spa_logit = self.fc_spa(spa_feat)
spe_logit = self.fc_spe(spe_feat)
weights = torch.softmax(torch.stack([spa_logit, spe_logit], dim=1), dim=1)
```

堆叠前：

$$
z_{\mathrm{spa}},z_{\mathrm{spe}}\in\mathbb R^{B\times128}.
$$

堆叠后：

$$
z\in\mathbb R^{B\times2\times128}.
$$

Softmax沿dim=1，即两个分支维：

$$
w_{\mathrm{spa},c}+w_{\mathrm{spe},c}=1.
$$

不是沿128通道做Softmax，所以不同通道之间不竞争。

### 11.4 第417～418行：恢复广播维度

```python
w_spa = weights[:, 0, :].unsqueeze(-1).unsqueeze(-1)
```

形状：

$$
[B,128]\rightarrow[B,128,1,1].
$$

同一通道的权重会广播到全部空间位置。

因此是：

> 每个tile、每个通道一对权重。

不是：

> 每个像素一对权重。

### 11.5 第420行：加权融合

```python
return w_spa * spa_feat + w_spe * spe_feat
```

公式：

$$
F_{\mathrm{fuse}}
=W_{\mathrm{spa}}\odot F_{\mathrm{spa}}
+W_{\mathrm{spe}}\odot F_{\mathrm{spe}}.
$$

由于权重非负且和为1，它是通道级凸组合。

该类只返回融合特征，没有返回权重用于可视化。若论文要展示机制，可以让forward可选返回`weights`或注册hook。

## 12. 第423～509行：`ImprovedBothMamba`，双分支总控逐行解读

### 12.1 这个类的职责

`ImprovedBothMamba`不直接做底层特征计算，它负责：

1. 根据消融配置决定创建哪些分支；
2. 把同一个共享特征送入空间和光谱分支；
3. 选择竞争融合或等权平均；
4. 可选减弱光谱融合影响；
5. 应用最外层残差。

它对应论文总体图中的双分支编码与融合主干。

### 12.2 第423～430行：构造函数参数

参数可分为五组：

**基础结构：**

- channels；
- token_num；
- use_residual；
- group_num。

**PRCA/LSP：**

- pyramid_dilation；
- scales/layers/heads；
- lsp_reduction。

**消融和残差：**

- ablation；
- outer_residual_mode；
- outer_residual_alpha。

**光谱融合：**

- spectral_diff_alpha；
- spectral_fusion_scale。

**两个Mamba：**

- 各自d_state；
- d_conv；
- expand。

构造函数参数多的原因是这个类承担了所有子模块的配置转发。

### 12.3 第431～440行：保存配置并确定分支开关

第431行初始化父类。

第432行校验消融名称。

第433行校验外残差模式。

第434行把外残差系数转float。

第435～437行：

```python
self.spectral_fusion_scale = float(spectral_fusion_scale)
if not 0.0 < self.spectral_fusion_scale <= 1.0:
    raise ValueError(...)
```

限制：

$$
0<\beta\le1.
$$

不允许0，因此即使设置很小，也不会在接口语义上完全删除光谱影响；真正删除光谱分支应使用`wo_dgs`。

第438行保存残差总开关。

第439行：

```python
self.use_spatial_branch = self.ablation not in {'wo_lpps'}
```

第440行类似决定光谱分支。

### 12.4 第442～458行：创建或删除空间分支

默认创建：

```python
self.spa_mamba = ImprovedSpaMamba(...)
```

传入：

- 128通道；
- 残差开关；
- GroupNorm组数；
- dilation；
- PRCA三个结构参数；
- 消融名；
- LSP reduction；
- Spatial Mamba三个参数。

一个细节：

> 这里没有把`token_num`传给`ImprovedSpaMamba`，所以它使用自身默认`token_num=4`。

当前两者恰好一致，但若未来只修改顶层`token_num`，光谱分支会改变，而空间分支内部用于计算PRCA dim的token_num仍为4。由于当前强制hidden_dim可整除且PRCA dim最终仍等于128，通常不影响主配置，但从配置一致性上建议显式传入。

`wo_lpps`时第458行把`spa_mamba=None`，不会创建空间分支参数。

### 12.5 第460～473行：创建或删除光谱分支

默认创建`ImprovedSpeMamba`，并显式传入：

- channels=128；
- token_num=4；
- residual；
- group_num；
- ablation；
- $\alpha=0.5$；
- Mamba参数。

`wo_dgs`时设为None。

### 12.6 第475～482行：何时创建竞争融合

必须同时满足：

1. 空间分支存在；
2. 光谱分支存在；
3. 不是`wo_competitive`。

才创建：

```python
self.fusion = CompetitiveFusion(channels)
```

其他情况设为None。

因此`fusion=None`有三种可能：

- 只剩空间分支；
- 只剩光谱分支；
- 双分支存在但消融竞争融合。

forward会先区分单分支，再在双分支情况下把`fusion=None`解释为等权平均。

### 12.7 第484～489行：最外层残差函数

```python
def _apply_outer_residual(self, x, block_x):
    if not self.use_residual or self.outer_residual_mode == 'no_outer':
        return block_x
    if self.outer_residual_mode == 'scaled':
        return x + self.outer_residual_alpha * block_x
    return block_x + x
```

三条路径：

**无外残差：**

$$
F_{\mathrm{out}}=F_{\mathrm{block}}.
$$

触发条件：

- 全局`use_residual=False`；
- 或`outer_residual_mode=no_outer`。

**缩放外残差：**

$$
F_{\mathrm{out}}
=x+\alpha_{\mathrm{outer}}F_{\mathrm{block}}.
$$

注意这里缩放的是block输出，不是identity分支。

**标准外残差：**

$$
F_{\mathrm{out}}=F_{\mathrm{block}}+x.
$$

当前使用标准模式。

### 12.8 第491～494行：只有光谱分支

```python
if self.spa_mamba is None:
    spe_x = self.spe_mamba(x)
    return self._apply_outer_residual(x, spe_x)
```

对应`wo_lpps`：

$$
F_{\mathrm{out}}=x+F_{\mathrm{spe}}
$$

默认仍加外层$x$残差。

因为`F_spe`内部已经加了`x_diff`，所以即使删除空间分支仍有两层特征保留路径。

### 12.9 第496～498行：只有空间分支

对应`wo_dgs`：

$$
F_{\mathrm{out}}=x+F_{\mathrm{spa}}.
$$

`F_spa`内部又加了`x_prior`，`x_prior`内部又加了$x$，因此残差层级较深。

### 12.10 第500～501行：默认双分支前向

```python
spa_x = self.spa_mamba(x)
spe_x = self.spe_mamba(x)
```

两个分支都接收完全相同的共享特征$x=F_0$：

$$
F_{\mathrm{spa}},F_{\mathrm{spe}}
\in\mathbb R^{B\times128\times H\times W}.
$$

它们是顺序执行，不是Python线程意义上的并行；在计算图结构上是并列分支。

### 12.11 第503～506行：等权平均或竞争融合

```python
if self.fusion is None:
    fusion_x = 0.5 * (spa_x + spe_x)
else:
    fusion_x = self.fusion(spa_x, spe_x)
```

双分支且`fusion=None`只对应`wo_competitive`，真实行为是：

$$
F_{\mathrm{fuse}}
=\frac12(F_{\mathrm{spa}}+F_{\mathrm{spe}}).
$$

论文文字称direct addition，与代码不一致。

默认使用通道级竞争融合。

### 12.12 第507～508行：`spectral_fusion_scale`

```python
fusion_x = spa_x + self.spectral_fusion_scale * (fusion_x - spa_x)
```

设$\beta=\mathrm{spectral\_fusion\_scale}$：

$$
F'_{\mathrm{fuse}}
=F_{\mathrm{spa}}
+\beta(F_{\mathrm{fuse}}-F_{\mathrm{spa}})
=(1-\beta)F_{\mathrm{spa}}+\beta F_{\mathrm{fuse}}.
$$

这是把最终融合结果向空间分支线性回缩：

- $\beta=1$：不改变；
- $\beta=0.5$：空间分支与竞争融合结果各一半；
- $\beta\rightarrow0$：接近空间分支。

当前$\beta=1$，所以第507条件为False，不执行。

这个参数是后续工程调节，不是论文核心公式。

### 12.13 第509行：应用外残差

默认：

$$
F_{\mathrm{block}}
=F_0+F_{\mathrm{fuse}}.
$$

论文Fig. 1画了这条残差，正文公式没有完整写出。

### 12.14 默认完整主块公式

令共享输入为$F_0$：

$$
X_{\mathrm{prior}}
=F_0+\mathrm{LSPTransform}(F_0),
$$

$$
F_{\mathrm{spa}}
=X_{\mathrm{prior}}
+\mathrm{SpaMambaTransform}(\mathrm{PRCA}(X_{\mathrm{prior}})),
$$

$$
F_{\mathrm{diff}}
=F_0+\alpha\Delta(F_0),
$$

$$
F_{\mathrm{spe}}
=F_{\mathrm{diff}}
+\mathrm{SpeMambaTransform}(\mathrm{Group}(F_{\mathrm{diff}})),
$$

$$
F_{\mathrm{fuse}}
=\mathrm{CompetitiveFusion}(F_{\mathrm{spa}},F_{\mathrm{spe}}),
$$

$$
F_{\mathrm{block}}
=F_0+F_{\mathrm{fuse}}.
$$

---

## 13. 第512～613行：`ImprovedMambaHSI`完整模型逐行解读

### 13.1 第512～521行：顶层类和参数

`ImprovedMambaHSI`是`train.py`实际导入的完整网络类。

类默认值不等于训练实际值：

| 参数 | 类默认 | 当前训练 |
|---|---:|---:|
| in_channels | 128 | 30 |
| hidden_dim | 64 | 128 |
| num_classes | 10 | 数据集决定 |
| dilation | (2,3) | 3 |
| diff alpha | 1.0 | 0.5 |
| high_res_skip | none | none |

第521行结束参数列表。

### 13.2 第522～524行：父类、消融和skip校验

```python
super(ImprovedMambaHSI, self).__init__()
self.ablation = _validate_ablation(ablation)
self.high_res_skip = _validate_high_res_skip_mode(high_res_skip)
```

在任何层创建前先确认字符串合法。

### 13.3 第525～541行：整体形状约束

调用`_validate_model_config`验证：

- 所有结构参数是正整数；
- hidden_dim能被GN组数、token数和attention头数整除；
- 分类头宽度能被GN组数整除。

这个函数只检查，不保存返回值。

当前：

$$
128/4=32
$$

同时满足：

- 每个GN组32通道；
- 每个光谱token 32维；
- 每个attention head 32通道。

三种“4”恰好共享同一个32，但语义不同。

### 13.4 第543～547行：共享Embedding

```python
self.patch_embedding = nn.Sequential(
    nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
    nn.GroupNorm(group_num, hidden_dim),
    nn.SiLU()
)
```

当前：

$$
[B,30,H,W]
\rightarrow[B,128,H,W].
$$

逐层：

1. 1×1 Conv：每个像素的30维PCA向量映射到128维；
2. GroupNorm(4,128)：不依赖batch统计；
3. SiLU：平滑非线性。

它不切patch、不下采样，名称`patch_embedding`是工程遗留。

参数量：

$$
30\times128+128+2\times128=4224.
$$

其中GroupNorm有128个scale和128个bias。

### 13.5 第549～570行：创建双分支主块

```python
self.mamba_block = ImprovedBothMamba(...)
```

这里把训练配置继续向下转发。

关键转发链：

```text
train.py args
→ model_kwargs
→ ImprovedMambaHSI
→ ImprovedBothMamba
→ ImprovedSpaMamba / ImprovedSpeMamba
→ PyramidAttention / Mamba
```

如果某个参数没有在链条中显式传递，就可能悄悄使用子类默认值。前面指出的空间分支`token_num`就是一个例子。

### 13.6 第571～575行：池化

```python
self.pool = nn.Identity() if pool_size == 1 else nn.AvgPool2d(
    kernel_size=pool_size,
    stride=pool_size,
    padding=0
)
```

两种行为：

- pool_size=1：Identity，分辨率不变；
- 当前pool_size=2：2×2平均池化，stride 2。

形状：

$$
[B,128,H,W]
\rightarrow
[B,128,\lfloor H/2\rfloor,\lfloor W/2\rfloor].
$$

若H或W是奇数，padding=0会舍弃最末不能组成完整2×2窗口的边缘。

### 13.7 第577～584行：可选高分辨率skip

默认`high_res_skip=none`：

```python
self.skip_proj = None
```

所以论文主线完全跳过这段。

如果使用`patch`或`pre_pool`，创建：

```text
1×1 Conv(128→128)
→ GroupNorm(4,128)
→ SiLU
```

用于把skip特征变换后加到池化特征。

它是后续工程扩展，不应拿来解释论文已有结果。

### 13.8 第586～591行：分类头

```python
self.cls_head = nn.Sequential(
    nn.Conv2d(hidden_dim, cls_head_dim, kernel_size=1),
    nn.GroupNorm(group_num, cls_head_dim),
    nn.SiLU(),
    nn.Conv2d(cls_head_dim, num_classes, kernel_size=1)
)
```

当前：

$$
[B,128,H/2,W/2]
\rightarrow[B,128,H/2,W/2]
\rightarrow[B,K,H/2,W/2].
$$

第一层1×1卷积用于分类前特征变换，第二层产生每类logit。

最后没有Softmax，因为`CrossEntropyLoss`内部会执行LogSoftmax；推理时直接对logits argmax。

18类分类头参数量：

$$
(128\times128+128)
+(2\times128)
+(128\times18+18)
=19090.
$$

### 13.9 第593～596行：主forward前三步

```python
x_embed = self.patch_embedding(x)
x_pre_pool = self.mamba_block(x_embed)
x_feat = self.pool(x_pre_pool)
```

逐步：

$$
x:[B,30,H,W],
$$

$$
x_{\mathrm{embed}}:[B,128,H,W],
$$

$$
x_{\mathrm{prepool}}:[B,128,H,W],
$$

$$
x_{\mathrm{feat}}:[B,128,H/2,W/2].
$$

变量名很有用：

- `x_embed`：共享$F_0$；
- `x_pre_pool`：双分支融合和外残差之后；
- `x_feat`：池化后分类特征。

### 13.10 第598～603行：选择skip来源

只有`skip_proj is not None`才执行。

`high_res_skip=patch`：

$$
skip=x_{\mathrm{embed}}.
$$

`high_res_skip=pre_pool`：

$$
skip=x_{\mathrm{prepool}}.
$$

两者区别：

- patch skip更浅，保留Embedding特征；
- pre_pool skip更深，已经包含双分支信息。

当前`none`，整段跳过。

### 13.11 第604～610行：把skip缩放到池化尺寸

若skip空间尺寸与`x_feat`不同，使用双线性插值：

```python
interpolate(
    skip_feat,
    size=x_feat.shape[-2:],
    mode='bilinear',
    align_corners=False
)
```

当前pool=2时，skip通常从$H\times W$缩小到$H/2\times W/2$。

这里用插值而不是AvgPool，因此skip路径与主路径下采样方式不同。

### 13.12 第611行：skip相加

```python
x_feat = x_feat + self.skip_proj(skip_feat)
```

这是池化后的额外残差。默认关闭，所以默认模型只有前面四类显式残差。

### 13.13 第613行：分类输出

```python
return self.cls_head(x_feat)
```

模型直接返回半分辨率logits：

$$
[B,K,\lfloor H/2\rfloor,\lfloor W/2\rfloor].
$$

它不返回：

- Softmax概率；
- 最终类别图；
- 竞争融合权重；
- 中间空间/光谱特征；
- 原分辨率logits。

训练时`head_loss`插值到标签尺寸，验证/测试由`train.py`插值。

## 14. 用真实forward hook验证整条数据流

在当前`gyp_hsi_env`和CUDA环境中，对18类模型输入：

$$
x\in\mathbb R^{1\times30\times32\times40}
$$

实际记录到：

| 模块 | 实际输入 | 实际输出 |
|---|---|---|
| patch_embedding | `[1,30,32,40]` | `[1,128,32,40]` |
| LSP | `[1,128,32,40]` | `[1,128,32,40]` |
| PRCA | `[1,128,32,40]` | `[1,128,32,40]` |
| Spatial Mamba | `[1,1280,128]` | `[1,1280,128]` |
| Spatial branch | `[1,128,32,40]` | `[1,128,32,40]` |
| Spectral Mamba | `[1280,4,32]` | `[1280,4,32]` |
| Spectral branch | `[1,128,32,40]` | `[1,128,32,40]` |
| Competitive Fusion | 两个`[1,128,32,40]` | `[1,128,32,40]` |
| Both block | `[1,128,32,40]` | `[1,128,32,40]` |
| AvgPool2d(2) | `[1,128,32,40]` | `[1,128,16,20]` |
| cls_head | `[1,128,16,20]` | `[1,18,16,20]` |

这里：

$$
32\times40=1280,
$$

所以空间序列长度和光谱分支的像素batch都是1280。

### 14.1 当前环境的运行边界

当前安装的`mamba_ssm/causal_conv1d`实现要求输入位于CUDA。直接在CPU执行会报：

```text
RuntimeError: Expected x.is_cuda() to be true
```

这不是`MambaHSI.py`形状逻辑错误，而是当前外部Mamba算子构建只支持CUDA前向。部署到无GPU环境前，需要：

- 使用支持CPU的替代实现；
- 或提供fallback；
- 或明确声明CUDA运行要求。

---

## 15. 参数量从源码怎样对应

以Tangdaowan 18类模型为例，实测总参数：

$$
883511.
$$

| 顶层模块 | 参数量 | 源码原因 |
|---|---:|---|
| patch_embedding | 4,224 | 30→128卷积、GN |
| 空间分支 | 817,253 | LSP、9个PRCA block、Spatial Mamba、GN |
| 光谱分支 | 10,176 | Spectral Mamba、GN |
| 竞争融合 | 32,768 | 两个128×128无bias Linear |
| 分类头 | 19,090 | 128→128→18与GN |
| 合计 | 883,511 | 与论文0.8835M一致 |

空间分支内部：

| 子模块 | 参数量 |
|---|---:|
| LSP | 22,209 |
| PRCA | 678,308 |
| Spatial Mamba | 116,480 |
| 输出GN | 256 |

光谱分支：

| 子模块 | 参数量 |
|---|---:|
| Spectral Mamba | 9,920 |
| 输出GN | 256 |

代码层面的结论：

> 模型虽然以Mamba命名，但参数主体是PRCA；DGS光谱分支只占约1.15%参数。

---

## 16. 源码和论文结构逐项对照

### 16.1 完全对应的部分

| 论文描述 | 源码证据 |
|---|---|
| 共享特征$F_0$ | 第543～547、594行 |
| LSP局部深度卷积 | 第307～310、324行 |
| LSP空间门控 | 第312～317、325～326行 |
| 三尺度PRCA | 第189～231行 |
| 通道注意力 | 第161～175行生成`[B,head,c,c]`矩阵 |
| Spatial Mamba | 第384～388行`[B,HW,C]` |
| 差分分组光谱Mamba | 第272～297行 |
| 通道级竞争融合 | 第394～420行 |
| 融合后$F_0$残差 | 第484～509行 |

### 16.2 论文需要按源码修订的部分

#### DGS残差

论文：

$$
F_{\mathrm{spe}}=\mathrm{MambaOutput}+F_0.
$$

源码第299行：

$$
F_{\mathrm{spe}}=\mathrm{MambaOutput}+F_{\mathrm{diff}}.
$$

#### DGS形状符号

源码唯一确定形状：

$$
[BHW,4,32].
$$

论文正文和Fig. 3的$T,G$定义需要统一。

#### Patch Embedding

源码只是1×1逐像素投影，没有切patch。

#### 竞争融合消融

源码是等权平均：

$$
0.5(F_{\mathrm{spa}}+F_{\mathrm{spe}}),
$$

不是论文文字中的direct addition。

#### 输出分辨率

源码返回半分辨率logits，原尺寸恢复发生在模型外。

#### whole-image

模型源码支持任意$H,W$的稠密输入，但当前训练入口对大图采用512重叠tile。因此更准确叫“稠密场景/tiled设计”。

---

## 17. 从代码审查角度需要知道的细节

### 17.1 `ImprovedSpaMamba`没有收到顶层token_num

第443行创建`ImprovedSpaMamba`时没有写：

```python
token_num=token_num
```

所以空间分支使用自身默认4。当前配置同样为4，无实际差异；未来若改变顶层token数，建议显式传递，避免配置漂移。

### 17.2 空间分支的通道补齐逻辑不完整

空间类计算了`channel_num=token_num*ceil(channels/token_num)`，并用它构造PRCA，但forward没有像光谱类那样padding。

顶层校验强制`hidden_dim % token_num == 0`，所以当前安全。独立使用`ImprovedSpaMamba`时可能出错。

### 17.3 光谱padding在顶层主线是不可达分支

顶层已经要求hidden_dim能被token_num整除，所以`padding_feature`不会执行。它只对单独实例化`ImprovedSpeMamba`或未来放宽顶层校验有意义。

### 17.4 PRCA连续三层没有残差和归一化

每尺度三个PyramidAttention直接串联。优点是结构简单，风险是深层通道变换可能优化不稳定。当前最终Spatial Mamba残差能保留`x_prior`，但PRCA内部没有短残差。

### 17.5 dilation融合是静态全局权重

多dilation时`dilation_logits`是模型参数，不依赖输入。不能称为“每幅图自适应选择dilation”。

### 17.6 LSP门控也是通道共享

输出只有1通道，所有语义通道共享空间权重。轻量，但表达能力受限。

### 17.7 竞争融合不是空间自适应

GAP抹去了空间位置，权重在$H,W$广播。不能说不同像素动态选择不同分支。

### 17.8 四类残差叠加

默认有：

1. LSP：`out+x`；
2. Spatial：`x_out+x_prior`；
3. Spectral：`x_out+x_diff`；
4. Both：`fusion+x_embed`。

多层残差有助优化，但也使“每个模块的净贡献”更难单独解释。

### 17.9 `use_residual=False`不代表完全无残差

它关闭空间、光谱和外层残差，但LSP内部第330行仍保留`out+x`。

### 17.10 assert不是最强运行时校验

`CompetitiveFusion`使用assert。Python优化模式可能移除assert，更稳妥是显式异常。

### 17.11 模型内部不做最终插值

第613行直接返回半分辨率结果。训练使用`align_corners=False`，验证/测试使用`True`，不一致发生在外部脚本。

### 17.12 当前模型没有正则化层

`MambaHSI.py`中没有：

- Dropout；
- DropPath；
- BatchNorm；
- stochastic depth。

主要依赖GroupNorm、残差、训练端label smoothing和少量参数控制过拟合。

### 17.13 输入尺寸必须满足PRCA最低尺度

三尺度中$i=2$使用kernel 4平均池化。输入H、W至少需要允许4×4池化；正常HSI tile远大于该尺寸。

### 17.14 奇数尺寸经过pool会向下取整

`AvgPool2d(2,padding=0)`对奇数边长舍弃最后边缘窗口，再由外部插值恢复。边缘像素可能因此受影响。

---

## 18. 老师从源码角度可能怎样追问

### 1. 为什么`MambaHSI.py`中没有Mamba内部状态方程

因为第5行直接导入外部`mamba_ssm.Mamba`。本文件负责构造空间/光谱序列和外围模块，Selective Scan实现在外部库。

### 2. 你的Spatial Mamba输入到底是什么

第385行明确是`[B,H*W,C]`，当前512 tile为`[1,262144,128]`。

### 3. 它是不是四方向扫描

不是。第385行按`B,H,W,C`重排并reshape，是单向行优先序列。

### 4. PRCA的注意力矩阵为什么不是$HW\times HW$

第161～163行得到`[B,head,c,HW]`，第170行Q乘K转置得到`[B,head,c,c]`。当前每头$c=32$。

### 5. `num_layers=2`为什么有三层

第189行先创建一个基础attention，第194～197行再为每尺度创建两个refinement attention，所以每尺度1+2=3。

### 6. PRCA一共有多少个attention

三尺度各三个，共9个。

### 7. dilation 2和3是否同时使用

类默认支持，但训练传入`"3"`，第45～58行转成`(3,)`，只使用3。

### 8. LSP门控为什么只有一个通道

第315行明确`Conv2d(mid,1)`，用于生成轻量全通道共享空间mask。

### 9. 光谱分组的顺序是什么

第286～290行直接reshape连续潜在通道，按0～31、32～63、64～95、96～127分组。

### 10. 为什么Spectral Mamba的d_model是32

Mamba输入为`[BHW,4,32]`，Mamba约定最后一维是d_model，所以第251行设置32。

### 11. token_num是不是GroupNorm组数

不是。`token_num=4`控制光谱序列长度；`group_num=4`控制GroupNorm。当前数值碰巧相同。

### 12. 差分最后一个通道怎样处理

第273行先全零，第274行只写`:-1`，所以最后差分为0，增强后保留原通道。

### 13. $\alpha=0.5$时代码实际做了什么

前127个通道变为$0.5F_c+0.5F_{c+1}$，更接近相邻潜在通道平均。

### 14. 差分是否发生在原始波段

不是。输入先在训练脚本PCA到30维，再由第543～547行映射到128维，差分在这128维上发生。

### 15. 为什么`proj`里没有projection

空间和光谱类的`proj`实际只有GroupNorm和SiLU，是命名遗留；真正通道投影在PRCA和分类头中。

### 16. 竞争融合权重在哪个维度Softmax

第416行`dim=1`，即两个分支维，不是通道维。

### 17. 权重是逐像素的吗

不是。第398、403行先GAP到1×1，第417～418行恢复为`[B,C,1,1]`广播。

### 18. `wo_competitive`为什么不是直接相加

第504行明确乘0.5，是等权平均。

### 19. 外层残差在哪里

第484～489行定义，第509行调用，默认返回`fusion_x+x`。

### 20. 为什么输出尺寸减半

第571～575行`AvgPool2d(pool_size=2)`，第613行没有上采样。

### 21. Softmax在哪里

类别Softmax不在模型中；CrossEntropy内部处理。模型内唯一显式Softmax是PRCA attention、dilation权重和双分支竞争权重。

### 22. 模型是否支持非512输入

支持。空间形状在forward动态读取；512只是训练脚本的tile设置。输入需要足够大以通过PRCA最低尺度池化。

### 23. 为什么当前CPU跑不了

当前环境的外部`causal_conv1d`算子要求CUDA，不是顶层PyTorch形状代码限制。

### 24. 哪段代码占参数最多

第189～200行创建的9个PyramidAttention和融合投影，PRCA共678,308参数。

### 25. 如果删除DGS会发生什么

第460～473行不创建光谱分支；第496～498行只运行空间分支并应用外残差。

### 26. 如果删除LPPS会发生什么

第442～458行不创建空间分支；第492～494行只运行光谱分支并应用外残差。

### 27. 如果只删除LSP呢

空间分支仍运行PRCA和Spatial Mamba，第375～378行令`x_prior=x`。

### 28. 如果只删除PRCA呢

第380～383行直接令`x_re=x_prior`，仍运行LSP和Spatial Mamba。

### 29. `high_res_skip`属于论文模型吗

当前论文主配置为none。patch/pre_pool是后续可选工程路径。

### 30. 你最应该修改哪几处源码

优先：

1. 显式向`ImprovedSpaMamba`传`token_num`；
2. 统一DGS残差与论文；
3. 统一训练/验证插值；
4. 让融合可选返回权重用于可视化；
5. 明确移除不可达padding或放宽校验并完整支持；
6. 将assert改为显式异常。

---

## 19. 一页源码速记

```text
1–24    导入、消融名、残差/skip模式
27–107  参数合法性与整除校验
109–182 PyramidAttention
         QKV 1×1 → dilation DWConv
         [B,128,H,W] → [B,4,32,HW]
         attention [B,4,32,32]
184–233 PRCA
         3尺度 × (1基础+2 refinement) = 9 blocks
236–299 DGS/Spectral
         latent diff → [BHW,4,32] → Mamba → +Fdiff
302–330 LSP
         DWConv + [B,1,H,W] gate + PWConv/GN/SiLU + x
333–391 LPPS/Spatial
         LSP → PRCA → [B,HW,128] Mamba → +x_prior
394–420 Competitive Fusion
         GAP + two FC → Softmax over branch → channel weights
423–509 Both block
         branch ablations → fusion/average → outer residual
512–613 Full model
         1×1 embed → both block → AvgPool2 → head
```

默认完整前向：

$$
[B,30,H,W]
\rightarrow[B,128,H,W]
\rightarrow
\begin{cases}
[B,HW,128] & \text{空间}\\
[BHW,4,32] & \text{光谱}
\end{cases}
\rightarrow[B,128,H,W]
\rightarrow[B,K,H/2,W/2].
$$

---

## 20. 与总答辩文档的关系

本文专门用于逐行阅读`MambaHSI.py`。

论文背景、实验、结果谱系、PPT和完整高频问答请同时阅读：

```text
/data2/gyp/HyPyraMamba/HypraMamba/docs/
PyS2CF-Mamba_预推免模型与论文答辩详解.md
```

复习建议：

1. 打开`model/MambaHSI.py`和本文并排阅读；
2. 每看到一个`permute/rearrange/reshape`，自己写一次前后形状；
3. 不看文档画出四条残差；
4. 能解释为什么PRCA是通道注意力；
5. 能解释DGS为什么是`[BHW,4,32]`；
6. 能主动指出论文和代码的DGS残差差异。

如果这六点能够独立讲清楚，老师从源码任何一层追问时，你基本都能沿着真实数据流回答，而不会只停留在论文模块名称。
