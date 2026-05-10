import os
import sys
# import re
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
import numbers
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x, ref):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        ref_qkv = self.qkv_dwconv(self.qkv(ref))

        q_shared, k, v = qkv.chunk(3, dim=1)
        q, k_shared, v_shared = ref_qkv.chunk(3, dim=1)

        q_shared = rearrange(q_shared, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_shared = rearrange(k_shared, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_shared = rearrange(v_shared, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        _, _, C, _ = q_shared.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v_shared)
        out2 = (attn2 @ v_shared)
        out3 = (attn3 @ v_shared)
        out4 = (attn4 @ v_shared)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##  Mixed-Scale Feed-forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x
##  Sparse Transformer Block (STB)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, ref):
        x = x + self.attn(self.norm1(x), self.norm1(ref))
        x = x + self.ffn(self.norm2(x))

        return x

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))

        return out + x


class SAM(nn.Module):
    def __init__(self, nf, use_residual=True, learnable=True):
        super(SAM, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv_shared = nn.Sequential(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True),
                                             nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            style = self.conv_shared(torch.cat([lr, ref], dim=1))
            gamma = self.conv_gamma(style)
            beta = self.conv_beta(style)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Encoder(nn.Module):
    def __init__(self, in_chl, nf, n_blks=[1, 1, 1], act='relu'):
        super(Encoder, self).__init__()

        block1 = functools.partial(ResidualBlock, nf=nf)
        block2 = functools.partial(ResidualBlock, nf=nf * 2)
        block3 = functools.partial(ResidualBlock, nf=nf * 2 * 2)

        self.conv_L1 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block1, n_layers=n_blks[0])

        self.conv_L2 = Downsample(nf)
        self.blk_L2 = make_layer(block2, n_layers=n_blks[1])

        self.conv_L3 = Downsample(nf * 2)
        self.blk_L3 = make_layer(block3, n_layers=n_blks[2])

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))
        fea_L2 = self.blk_L2(self.act(self.conv_L2(fea_L1)))
        fea_L3 = self.blk_L3(self.act(self.conv_L3(fea_L2)))

        return [fea_L1, fea_L2, fea_L3]

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Decoder(nn.Module):
    def __init__(self, nf, out_chl, n_blks=[1, 1, 1, 1, 1, 1]):
        super(Decoder, self).__init__()

        block1 = functools.partial(ResidualBlock, nf=nf * 4 * 2)
        block2 = functools.partial(ResidualBlock, nf=nf * 4)
        block3 = functools.partial(ResidualBlock, nf=nf * 2)
        block4 = functools.partial(ResidualBlock, nf=nf)

        self.conv_L3 = Downsample(nf * 4)  # (1, 512, 32, 32)
        self.blk_L3 = make_layer(block1, n_layers=n_blks[0])

        self.conv_L2 = nn.Conv2d(nf * 4 * 2, nf * 4 * 2, 3, 1, 1, bias=True)
        self.blk_L2 = make_layer(block1, n_layers=n_blks[1])
        self.up1 = Upsample(nf * 2 ** 3)  # (1, 256, 64, 64)

        self.conv_L1 = nn.Conv2d(nf * 4 * 2, nf * 4, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block2, n_layers=n_blks[2])
        self.up2 = Upsample(nf * 2 ** 2)

        self.merge_warp_x1 = nn.Conv2d(nf * 4 * 2, nf * 4, 3, 1, 1, bias=True)
        self.blk_x1 = make_layer(block2, n_blks[3])

        self.merge_warp_x2 = nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=True)
        self.blk_x2 = make_layer(block3, n_blks[4])
        self.up3 = Upsample(nf * 2 ** 1)

        self.merge_warp_x4 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.blk_x4 = make_layer(block4, n_blks[5])

        self.conv_out = nn.Conv2d(64, out_chl, 3, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

        self.pAda1 = SAM(nf * 4, use_residual=True, learnable=True)
        self.pAda2 = SAM(nf * 2, use_residual=True, learnable=True)
        self.pAda3 = SAM(nf, use_residual=True, learnable=True)

    def forward(self, lr_l, warp_ref_l):
        fea_L3 = self.act(self.conv_L3(lr_l[2]))
        fea_L3 = self.blk_L3(fea_L3)

        fea_L2 = self.act(self.conv_L2(fea_L3))
        fea_L2 = self.blk_L2(fea_L2)
        fea_L2_up = self.up1(fea_L2)

        fea_L1 = self.act(self.conv_L1(torch.cat([fea_L2_up, lr_l[2]], dim=1)))
        fea_L1 = self.blk_L1(fea_L1)

        warp_ref_x1 = self.pAda1(fea_L1, warp_ref_l[2]) #(1, 256, 64, 64)
        fea_x1 = self.act(self.merge_warp_x1(torch.cat([warp_ref_x1, fea_L1], dim=1)))
        fea_x1 = self.blk_x1(fea_x1)
        fea_x1_up = self.up2(fea_x1) #(1, 128, 128, 128)

        warp_ref_x2 = self.pAda2(fea_x1_up, warp_ref_l[1])
        fea_x2 = self.act(self.merge_warp_x2(torch.cat([warp_ref_x2, fea_x1_up], dim=1)))
        fea_x2 = self.blk_x2(fea_x2)  #(1, 128, 128, 128)
        fea_x2_up = self.up3(fea_x2)  #(1, 64, 256, 256)

        warp_ref_x4 = self.pAda3(fea_x2_up, warp_ref_l[0])
        fea_x4 = self.act(self.merge_warp_x4(torch.cat([warp_ref_x4, fea_x2_up], dim=1)))
        fea_x4 = self.blk_x4(fea_x4)
        out = self.conv_out(fea_x4)

        return out


class SCANSR(nn.Module):
    def __init__(self, upscale):
        super().__init__()
        input_size = 256
        in_chl = 1
        nf = 64
        n_blks = [4, 4, 4]
        n_blks_dec = [2, 2, 2, 12, 8, 4]
        self.scale = upscale
        depths = [1, 1, 1]

        self.enc = Encoder(in_chl=in_chl, nf=nf, n_blks=n_blks)
        self.decoder = Decoder(nf, in_chl, n_blks=n_blks_dec)

        self.trans_lv1 = nn.ModuleList(
            [TransformerBlock(dim=int(nf), num_heads=1, ffn_expansion_factor=2.26, bias=False, LayerNorm_type='WithBias')
             for i in range(depths[0])])
        self.trans_lv2 = nn.ModuleList(
            [TransformerBlock(dim=int(nf * 2 ** 1), num_heads=2, ffn_expansion_factor=2.26, bias=False, LayerNorm_type='WithBias')
            for i in range(depths[1])])
        self.trans_lv3 = nn.ModuleList(
            [TransformerBlock(dim=int(nf * 2 ** 2), num_heads=4, ffn_expansion_factor=2.26, bias=False, LayerNorm_type='WithBias')
             for i in range(depths[2])])

    def forward(self, lr, ref):

        lrsr = F.interpolate(lr, scale_factor=self.scale, mode='bilinear', align_corners=True)

        fea_lrsr = self.enc(lrsr)
        fea_ref_l = self.enc(ref)

        warp_ref_patches_x4 = fea_lrsr[0]  # 1,64,256,256
        warp_ref_patches_x2 = fea_lrsr[1]  # 1,128,128,128
        warp_ref_patches_x1 = fea_lrsr[2]  # 1,256,64,64
        for transformer in self.trans_lv1:
            warp_ref_patches_x4 = transformer(warp_ref_patches_x4, fea_ref_l[0])
            fea_ref_l[0] = warp_ref_patches_x4

        for transformer in self.trans_lv2:
            warp_ref_patches_x2 = transformer(warp_ref_patches_x2, fea_ref_l[1])
            fea_ref_l[1] = warp_ref_patches_x2

        for transformer in self.trans_lv3:
            warp_ref_patches_x1 = transformer(warp_ref_patches_x1, fea_ref_l[2])
            fea_ref_l[2] = warp_ref_patches_x1

        warp_ref_l = [warp_ref_patches_x4, warp_ref_patches_x2, warp_ref_patches_x1]
        out = self.decoder(fea_lrsr, warp_ref_l)
        out = out + lrsr

        return out

if __name__=='__main__':
    net = SCANSR(upscale=4)
    # net.test()
    input1_tensor = torch.rand(1, 1, 64, 64)
    input2_tensor = torch.rand(1, 1, 256, 256)
    out = net(input1_tensor, input2_tensor)
    # print(out.shape)

    # 轻量级参数统计
    from thop import profile
    from thop import clever_format
    # macs, params = profile(network, inputs=(test_img,))
    macs1, params1 = profile(net, inputs=(input1_tensor, input2_tensor,))
    macs1, params1 = clever_format([macs1, params1], "%.3f")
    print("MGDUN x4:")
    print("Model FLOPs: ", macs1)
    print("Model Params:", params1)

    # net = Decoder(nf=64, out_chl=1)
    # w4 = torch.rand(1, 64, 256, 256)
    # w2 = torch.rand(1, 128, 128, 128)
    # w1 = torch.rand(1, 256, 64, 64)
    #
    # lr = [w4, w2, w1]
    # w = [w4, w2, w1]
    # out = net(lr, w)
    # print(' ')
