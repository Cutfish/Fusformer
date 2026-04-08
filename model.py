import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# --------------------------Main------------------------------- #

class MainNet(nn.Module):

    def __init__(self):
        super(MainNet, self).__init__()
        num_channel = 31
        num_feature = 48
        ####################
        self.T_E = Transformer_E(num_feature)
        self.T_D = Transformer_D(num_feature)
        self.Embedding = nn.Sequential(
            nn.Linear(num_channel+3,num_feature),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(num_feature,num_feature,3,1,1),
            nn.LeakyReLU(),
            nn.Conv2d(num_feature, num_channel, 3, 1, 1)
        )

    def forward(self, HSI, MSI):
        # 输入数据形状说明：
        # HSI: 低分辨率高光谱图像 LRHSI, shape: [b, 31, 16, 16]
        # MSI: 高分辨率多光谱图像 HRMSI, shape: [b, 3, 64, 64]
        
        ################LR-HSI###################
        # 上采样低分辨率高光谱图像: [b, 31, 16, 16] -> [b, 31, 64, 64]
        # 第1行：使用双三次插值将低分辨率HSI上采样4倍，空间尺寸从16x16变为64x64
        # 输入: HSI [b, 31, 16, 16] -> 输出: UP_LRHSI [b, 31, 64, 64]
        UP_LRHSI = F.interpolate(HSI,scale_factor=4, mode='bicubic') ### (b N h w)
        # 第2行：将像素值裁剪到[0,1]范围，防止插值后出现超出范围的值
        UP_LRHSI = UP_LRHSI.clamp_(0,1)
        # 第3行：获取特征图的高度（或宽度，因为H=W=64），用于后续reshape操作
        sz= UP_LRHSI.size(2)  # sz = 64
        
        # 第4行：在通道维度(dim=1)拼接上采样的HSI和MSI
        # UP_LRHSI [b, 31, 64, 64] + MSI [b, 3, 64, 64] -> Data [b, 34, 64, 64]
        Data = torch.cat((UP_LRHSI,MSI),1)
        # 第5-6行：使用einops的rearrange将图像格式转换为序列格式
        # 将空间维度H和W合并为一个序列长度：64*64=4096
        # Data [b, 34, 64, 64] -> E [b, 4096, 34]
        # 其中每个序列位置包含一个像素点的所有通道信息
        E = rearrange(Data, 'B c H W -> B (H W) c', H = sz)
        
        # 特征嵌入：将通道数从34映射到48维特征 [b, 4096, 34] -> [b, 4096, 48]
        E = self.Embedding(E)
        
        # Transformer编码器：提取特征，保持形状 [b, 4096, 48]
        Code = self.T_E(E)
        
        # Transformer解码器：进一步处理特征，保持形状 [b, 4096, 48]
        Highpass = self.T_D(Code)
        
        # 重排回图像格式：[b, 4096, 48] -> [b, 48, 64, 64]
        Highpass = rearrange(Highpass,'B (H W) C -> B C H W', H = sz)
        
        # 精炼网络：将特征通道数从48调整为31 [b, 48, 64, 64] -> [b, 31, 64, 64]
        Highpass = self.refine(Highpass)
        
        # 残差连接：高频信息 + 上采样的低分辨率图像 [b, 31, 64, 64]
        output = Highpass + UP_LRHSI
        output = output.clamp_(0,1)

        # 输出说明：
        # output: 最终融合的高分辨率高光谱图像, shape: [b, 31, 64, 64]
        # UP_LRHSI: 上采样的低分辨率高光谱图像, shape: [b, 31, 64, 64]
        # Highpass: 学习到的高频细节信息, shape: [b, 31, 64, 64]
        return output,UP_LRHSI,Highpass


# -----------------Transformer-----------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim) # 48
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    # norm 对于一个batch的每个样本进行归一化，即对同一个位置的所有通道进行归一化
    # 3 * 1024 * 48，就是 3 个样本，每个样本 1024 个位置，每个位置 48 个通道，每个位置的均值等于0，标准差是1

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# dropout = 0.意味着放弃丢弃，也就是相当于没有这一层
class Attention(nn.Module):
    def __init__(self, dim=48, heads=3, dim_head=16, dropout=0.):  # 默认值来自 Transformer_E 的调用参数
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5 # 0.25

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False) # 48 -> 48 * 3

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), # 48 -> 48
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() # nn.Identity()相当于什么都不做

    def forward(self, x, mask=None):
        # 3 * 4096 * 48 * 3
        b, n, _, h = *x.shape, self.heads 
        # 3*4096*48 -> 3*4096*(48*3) -> (3*4096*48)* 3
        qkv = self.to_qkv(x).chunk(3, dim=-1) 
        # 将 Q/K/V 的最后一维(48) 拆分为多头结构: [b, n, 48] -> [b, h=3, n=4096, d=16]
        # h: 注意力头数, d: 每个头的维度(dim_head), 48 = 3*16
        # q、k、v 的形状都是 [b=3, h=3, n=4096, d=16]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # 计算注意力分数矩阵[3, 3, 4096, 4096] （d没有了，所以是d被（点积）求和消掉）
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # 创建掩码填充值：取浮点类型的最小值（即负无穷大）
        # 这样经过softmax后，这些位置的权重会趋近于0，达到"屏蔽"效果
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            # 展平mask的第一维（序列长度维），并在前面填充一个True（用于CLS token）
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            # 将mask扩展为注意力矩阵的形状：[b, i] -> [b, (), i, ()]
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            # 用负无穷大填充需要mask的位置
            dots.masked_fill_(~mask, mask_value)
            del mask

        # Softmax归一化：将注意力分数转换为概率分布（各行和为1）（仍然是[3, 3, 4096, 4096]）
        # shape: [b, h, n, n]，表示每个头对每个位置查询对所有键的注意力权重
        attn = dots.softmax(dim=-1)

        # 加权聚合Value向量：用注意力权重对V进行加权求和
        # [3, 3, 4096, 4096] * [b=3, h=3, n=4096, d=16] = [b=3, h=3, n=4096, d=16]
        # shape: [b, h, n, n] x [b, h, n, d] -> [b, h, n, d]
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # 多头拼接：将多个头的输出合并回原始维度
        # shape: [b, h, n, d] -> [b, n, h*d]，即 [b=3, h=3, n=4096, d=16] -> [b, 4096, 48]
        out = rearrange(out, 'b h n d -> b n (h d)')

        # 输出线性投影 + Dropout：将拼接后的结果映射回原始维度
        out = self.to_out(out)

        return out

# Transformer编码器：提取特征，保持形状 [b, 4096, 48], dim = 48
class Transformer_E(nn.Module):
    def __init__(self, dim, depth=2, heads=3, dim_head=16, mlp_dim=48, sp_sz=64*64, num_channels = 48,dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_embedding = nn.Parameter(torch.randn(1, sp_sz, num_channels))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim,Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim,FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, mask=None):
        # pos = self.pos_embedding
        # x += pos
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Transformer_D(nn.Module):
    def __init__(self, dim, depth=2, heads=3, dim_head=16, mlp_dim=48 , sp_sz=64*64, num_channels = 48, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_embedding = nn.Parameter(torch.randn(1, sp_sz, num_channels))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim,Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim,Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim,FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        # pos = self.pos_embedding
        # x += pos
        for attn1,attn2, ff in self.layers:
            x = attn1(x, mask=mask)
            x = attn2(x, mask=mask)
            x = ff(x)
        return x