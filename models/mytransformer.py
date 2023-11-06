import torch
import torch.nn as nn
import numpy as np
from torch import  einsum
import torch.nn.functional as F
from einops import rearrange
from torchvision import models
from logging import Logger
from typing import Optional, Any
from torchinfo import summary
class _Residual_Block(nn.Module):
    def __init__(self, channels):
        super(_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity_data = x
        output = self.prelu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = torch.add(output, identity_data)
        output = self.prelu(output)
        return output

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Depth_separable_downBlock(nn.Module):
    def __init__(self,in_c,out_c,kernel_size,stride,padding=0):
        super(Depth_separable_downBlock, self).__init__()
        self.point_wise = nn.Sequential(
            nn.Conv2d(in_channels=in_c,
                      out_channels=out_c,
                      kernel_size=1,
                      stride=1,
                      padding=0)
        )
        self.depth_wise = nn.Sequential(
            nn.Conv2d(in_channels=in_c,
                      out_channels=in_c,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups =in_c),
            nn.PReLU())



    def forward(self, x):
        output = self.point_wise(self.depth_wise(x))
        return output
class Depth_separable_upBlock(nn.Module):
    def __init__(self,in_c,out_c,kernel_size,stride):
        super(Depth_separable_upBlock, self).__init__()
        self.point_wise = nn.Sequential(
            nn.Conv2d(in_channels=in_c,
                      out_channels=out_c,
                      kernel_size=1,
                      stride=1,
                      padding=0)
        )
        self.depth_wise= nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_c,
                               out_channels=out_c,
                               kernel_size=kernel_size,
                               stride=stride),
            nn.PReLU())
    def forward(self, x):
        output = self.depth_wise(self.point_wise(x))
        return output
class DeepFeatureExtractBlock(nn.Module):
    def __init__(self,in_ch,out_ch,ksize):
        super(DeepFeatureExtractBlock, self).__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.initial_kernel_size = ksize
        self.sub_channels = out_ch//4
        self.DeepFeature_Layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch,kernel_size=(ksize, ksize), stride=(1, 1), padding=ksize//2),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.DeepFeature_Layer2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4,kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.DeepFeature_Layer3 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4,kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.DeepFeature_Layer4 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.DeepFeature_Layer5 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4, kernel_size=(7, 7), stride=(1, 1), padding=3),
            nn.LeakyReLU(0.2,inplace=True)
        )


    def forward(self,x):
        x_deepfeature = self.DeepFeature_Layer1(x)
        f1,f3,f5,f7 = x_deepfeature[:,0:self.sub_channels,:,:],x_deepfeature[:,self.sub_channels:self.sub_channels*2,:,:]\
        ,x_deepfeature[:,self.sub_channels*2:self.sub_channels*3,:,:],x_deepfeature[:,self.sub_channels*3:self.sub_channels*4,:,:]
        x_deepfeature1 = self.DeepFeature_Layer2(f1)
        x_deepfeature3 = self.DeepFeature_Layer3(f3)
        x_deepfeature5 = self.DeepFeature_Layer4(f5)
        x_deepfeature7 = self.DeepFeature_Layer5(f7)
        x_deepfeature_cat = torch.cat([x_deepfeature1,x_deepfeature3,x_deepfeature5,x_deepfeature7],dim=1)
        y = x_deepfeature + x_deepfeature_cat
        return y
class MultiAttentionResBlock(nn.Module):
    def __init__(self,in_ch,out_ch,ksize):
        super(MultiAttentionResBlock, self).__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.initial_kernel_size = ksize
        self.sub_channels = out_ch//4
        self.se = se_block(self.out_channels, 8)
        self.DeepFeature_Layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch,kernel_size=(ksize, ksize), stride=(1, 1), padding=ksize//2),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.DeepFeature_Layer2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4,kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.DeepFeature_Layer3 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4,kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.DeepFeature_Layer4 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.DeepFeature_Layer5 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4, kernel_size=(7, 7), stride=(1, 1), padding=3),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.lrelu = nn.LeakyReLU()


    def forward(self,x):
        x_deepfeature = self.DeepFeature_Layer1(x)
        f1,f3,f5,f7 = x_deepfeature[:,0:self.sub_channels,:,:],x_deepfeature[:,self.sub_channels:self.sub_channels*2,:,:]\
        ,x_deepfeature[:,self.sub_channels*2:self.sub_channels*3,:,:],x_deepfeature[:,self.sub_channels*3:self.sub_channels*4,:,:]
        x_deepfeature1 = self.DeepFeature_Layer2(f1)
        x_deepfeature3 = self.DeepFeature_Layer3(f3)
        x_deepfeature5 = self.DeepFeature_Layer4(f5)
        x_deepfeature7 = self.DeepFeature_Layer5(f7)
        x_deepfeature_cat = torch.cat([x_deepfeature1,x_deepfeature3,x_deepfeature5,x_deepfeature7],dim=1)
        #x_deepfeature_cat = self.se(x_deepfeature_cat)
        y = x_deepfeature + x_deepfeature_cat

        
        y = self.lrelu(y)

        return y
class MultiResBlock_noConv(nn.Module):
    def __init__(self,in_ch,out_ch,ksize):
        super(MultiResBlock_noConv, self).__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.initial_kernel_size = ksize
        self.sub_channels = out_ch//4
        self.se = se_block(self.out_channels, 8)
        self.DeepFeature_Layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch,kernel_size=(ksize, ksize), stride=(1, 1), padding=ksize//2),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.DeepFeature_Layer2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4,kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.DeepFeature_Layer3 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4,kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.DeepFeature_Layer4 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.DeepFeature_Layer5 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch // 4, out_channels=out_ch // 4, kernel_size=(7, 7), stride=(1, 1), padding=3),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.lrelu = nn.LeakyReLU()


    def forward(self,x):
        x_deepfeature = x
        f1,f3,f5,f7 = x_deepfeature[:,0:self.sub_channels,:,:],x_deepfeature[:,self.sub_channels:self.sub_channels*2,:,:]\
        ,x_deepfeature[:,self.sub_channels*2:self.sub_channels*3,:,:],x_deepfeature[:,self.sub_channels*3:self.sub_channels*4,:,:]
        x_deepfeature1 = self.DeepFeature_Layer2(f1)
        x_deepfeature3 = self.DeepFeature_Layer3(f3)
        x_deepfeature5 = self.DeepFeature_Layer4(f5)
        x_deepfeature7 = self.DeepFeature_Layer5(f7)
        x_deepfeature_cat = torch.cat([x_deepfeature1,x_deepfeature3,x_deepfeature5,x_deepfeature7],dim=1)
        x_deepfeature_cat = self.se(x_deepfeature_cat)
        y = x_deepfeature + x_deepfeature_cat

        
        y = self.lrelu(y)

        return y



class ChannelAttentionBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelAttentionBlock, self).__init__()
        self.reduction = reduction
        self.dct_layer = nn.AdaptiveAvgPool2d(1)# DCTLayer(channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        n,c,h,w = x.size()
        y = self.dct_layer(x).squeeze(-1).squeeze(-1)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


#   Spatial Attention Block
class SpatialAttentionBlock(nn.Module):
    def __init__(self,channel):
        super(SpatialAttentionBlock, self).__init__()
         # Maximum pooling
        self.featureMap_max = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(1, 1),padding=0)
        )
        # Average pooling
        self.featureMap_avg = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.AvgPool2d(kernel_size=(5, 5), stride=(1,1), padding=0)
        )

        # Deviation pooling
        # var = \sqrt(featureMap - featureMap_avg)^2

        # Dimensionality Reduction
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(in_channels=channel * 4, out_channels=channel, kernel_size=(3,3), stride=(1, 1), padding=1,bias=False),
            nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=(1,1),stride=(1,1),bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        x_max = self.featureMap_max(x)
        x_avg = self.featureMap_avg(x)
        x_var = torch.sqrt(torch.pow(x - x_avg,2) + 1e-7)

        y = torch.cat([x_max,x_avg,x_var,x],dim=1)
        z = self.reduce_dim(y)
        return x * z

def conv1x1(in_channels, out_channels, stride=1, padding=0, *args, **kwargs):
    # type: (int, int, int, int, Any, Any) -> nn.Conv2d
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                     stride=stride, padding=padding, *args, **kwargs)


def conv3x3(in_channels, out_channels, stride=1, padding=1, *args, **kwargs):
    # type: (int, int, int, int, Any, Any) -> nn.Conv2d
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                     stride=stride, padding=padding, *args, **kwargs)


def build_norm_layer(logger, n_feats, norm_type='BN', *args, **kwargs):
    r""" build a normalization layer in [BatchNorm, InstanceNorm]
    Args:
        logger (Logger): logger
        n_feats (int): output channel
        norm_type (str): 'BN' or 'IN'
    Returns:
        nn.Module: expected normalization layer
    """
    if norm_type == 'BN':
        return nn.BatchNorm2d(num_features=n_feats, *args, **kwargs)
    elif norm_type == 'IN':
        return nn.InstanceNorm2d(num_features=n_feats, *args, **kwargs)
    else:
        logger.error(f'no such type of norm_layer:{norm_type}')
        raise SystemExit(f'no such type of norm_layer:{norm_type}')


class ResBlock(nn.Module):
    def __init__(self, logger, n_feats, norm_type='BN'):
        # type: (Logger, int, Optional[str]) -> None
        super(ResBlock, self).__init__()
        self.basic = []
        self.basic.append(conv3x3(n_feats, n_feats))
        if norm_type is not None:
            self.basic.append(build_norm_layer(logger, n_feats, norm_type))
        self.basic.append(nn.ReLU(True))
        self.basic.append(conv3x3(n_feats, n_feats))
        if norm_type is not None:
            self.basic.append(build_norm_layer(logger, n_feats, norm_type))
        self.basic = nn.Sequential(*self.basic)

    def forward(self, x):
        return self.basic(x) + x



class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False



class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding, cross_attn):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.cross_attn = cross_attn

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        if not self.cross_attn:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
            self.to_q = nn.Linear(dim, inner_dim, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, y=None):
        if self.shifted:
            x = self.cyclic_shift(x)
            if self.cross_attn:
                y = self.cyclic_shift(y)

        b, n_h, n_w, _, h = *x.shape, self.heads
        # print('forward-x: ', x.shape)   # [N, H//downscaling_factor, W//downscaling_factor, hidden_dim]
        if not self.cross_attn:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            # [N, H//downscaling_factor, W//downscaling_factor, head_dim * head] * 3
        else:
            kv = self.to_kv(x).chunk(2, dim=-1)
            qkv = (self.to_q(y), kv[0], kv[1])

        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)
        # print('forward-q: ', q.shape)   # [N, num_heads, num_win, win_area, hidden_dim/num_heads]
        # print('forward-k: ', k.shape)
        # print('forward-v: ', v.shape)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale  # q * k / sqrt(d)

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        # [N, H//downscaling_factor, W//downscaling_factor, head_dim * head]
        out = self.to_out(out)
        # [N, H//downscaling_factor, W//downscaling_factor, dim]
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding, cross_attn):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding,
                                                                     cross_attn=cross_attn)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x, y=None):
        x = self.attention_block(x, y=y)
        x = self.mlp_block(x)
        return x
class SwinBlock_change(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding, cross_attn):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding,
                                                                     cross_attn=cross_attn)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x, y=None):
        x = self.attention_block(x, y=y)
        #x = self.mlp_block(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x  # [N, H//downscaling_factor, W//downscaling_factor, out_channels]
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        #x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x


class SwinModule_change(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding, cross_attn,shifted=False):
        r"""
        Args:
            in_channels(int): 输入通道数
            hidden_dimension(int): 隐藏层维数，patch_partition提取patch时有个Linear学习的维数
            layers(int): swin block数，必须为2的倍数，连续的，regular block和shift block
            downscaling_factor: H,W上下采样倍数
            num_heads: multi-attn 的 attn 头的个数
            head_dim:   每个attn 头的维数
            window_size:    窗口大小，窗口内进行attn运算
        """
        super().__init__()
        #assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        if layers==1:
            self.layers=nn.Sequential(
                    SwinBlock_change(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            cross_attn=cross_attn)
                
            )
        if layers==2:#两个模块都不带MLP
           self.layers=nn.Sequential(
                SwinBlock_change(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            cross_attn=cross_attn),
                SwinBlock_change(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn),
            )
        if layers==3:#第一个模块带MLP 第二个模块不带
            self.layers=nn.Sequential(
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            cross_attn=cross_attn),
                SwinBlock_change(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn),
            )
        if layers==4:#两个模块都用multiblock替代MLP
            self.layers=nn.Sequential(
                SwinBlock_change(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            cross_attn=cross_attn),
                MultiResBlock_noConv(hidden_dimension, hidden_dimension, 3),
                SwinBlock_change(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn),
                MultiResBlock_noConv(hidden_dimension, hidden_dimension, 3),
            )
            



    def forward(self, x, y=None):
        if y is None:
            #original_x=x
            x = self.patch_partition(x)  # [N, H//downscaling_factor, W//downscaling_factor, hidden_dim]
            x = self.layers(x)
            return x.permute(0, 3, 1, 2)
            # [N, hidden_dim,  H//downscaling_factor, W//downscaling_factor]
class SwinModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding, cross_attn):
        r"""
        Args:
            in_channels(int): 输入通道数
            hidden_dimension(int): 隐藏层维数，patch_partition提取patch时有个Linear学习的维数
            layers(int): swin block数，必须为2的倍数，连续的，regular block和shift block
            downscaling_factor: H,W上下采样倍数
            num_heads: multi-attn 的 attn 头的个数
            head_dim:   每个attn 头的维数
            window_size:    窗口大小，窗口内进行attn运算
        """
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn),
            ]))

    def forward(self, x, y=None):
        if y is None:
            x = self.patch_partition(x)  # [N, H//downscaling_factor, W//downscaling_factor, hidden_dim]
            for regular_block, shifted_block in self.layers:
                x = regular_block(x)
                x = shifted_block(x)
            return x.permute(0, 3, 1, 2)
            # [N, hidden_dim,  H//downscaling_factor, W//downscaling_factor]
        else:
            x = self.patch_partition(x)
            y = self.patch_partition(y)
            for regular_block, shifted_block in self.layers:
                x = regular_block(x, y)
                x = shifted_block(x, y)
            return x.permute(0, 3, 1, 2)
class my_model_3_31_2(nn.Module):
    #proposed
    def __init__(self,in_channel):
        super(my_model_3_31_2, self).__init__()
        self.encoder1_pan=nn.Sequential(
             nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
           
            )

        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
           
            
        )
        self.encoder1_lr=nn.Sequential(
             nn.Conv2d(in_channels=in_channel,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
           
                       )
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
 
            SwinModule_change(in_channels=128, hidden_dimension=128, layers=3,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False),
                       
            MultiResBlock_noConv(128, 128, 5),
            
            
        )
        self.fusion2=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
        )
        self.restore1=nn.Sequential(
            
            SwinModule_change(in_channels=256, hidden_dimension=256, layers=3,
                       downscaling_factor=1, num_heads=8, head_dim=4,
                       window_size=4, relative_pos_embedding=True, cross_attn=False,shifted=True),
                       
            MultiResBlock_noConv(256, 256, 3),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=2,
                               stride=2),
            nn.PReLU())
        self.restore2=nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=2,
                               stride=2),
            nn.PReLU()
        )
        self.restore3=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            
            nn.Conv2d(in_channels=64,
                      out_channels=in_channel,
                      kernel_size=3,
                      stride=1,
                      padding=1),
        )

    def forward(self, x_pan, x_lr):
        encoder1_pan = self.encoder1_pan(x_pan)
        encoder1_lr = self.encoder1_lr(x_lr)

        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)

        fusion1 = self.fusion1(torch.cat((encoder2_pan, encoder2_lr), dim=1))
        fusion2 = self.fusion2(fusion1)

        restore1 = self.restore1(fusion2)
        restore2 = self.restore2(torch.cat((restore1, fusion1),dim=1))
        restore3 = self.restore3(torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1))

        return  restore3+x_lr

