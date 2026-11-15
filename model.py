# -*- coding: utf-8 -*-
"""实验三 模型：黄斑区域检测 + 中心凹热图回归 + 可见性判别（共享编码器三头）

设计思路
--------
1. 任务要求“先定位黄斑区域，再在其基础上输出中心凹坐标”，因此网络同时输出：
   - mask 头：黄斑区域概率图，作为 ROI 先验；
   - heat 头：中心凹位置热图，坐标由 soft-argmax 得到（相对原图归一化坐标）；
   - vis 头：中心凹是否可见的二分类，若判为不可见则坐标直接置 (0,0)。
2. 三者共享同一个轻量 U-Net 编码器，检测误差可直接约束回归分支的输入范围。
3. 黄斑区域缺少逐像素标注时，用中心凹坐标生成高斯盘作为弱监督伪标签，
   仅用于学习“黄斑大概在哪”，不作为最终精度指标。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MaculaFoveaNet(nn.Module):
    """编码-解码主干 + 三个输出头"""

    def __init__(self, base_ch=32):
        super().__init__()
        c = base_ch
        self.enc1 = ConvBlock(3, c)            # 1/1
        self.enc2 = ConvBlock(c, c * 2)        # 1/2
        self.enc3 = ConvBlock(c * 2, c * 4)    # 1/4
        self.enc4 = ConvBlock(c * 4, c * 8)    # 1/8
        self.bottleneck = ConvBlock(c * 8, c * 8)   # 1/16
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c * 8, c * 8, 2, stride=2)
        self.dec4 = ConvBlock(c * 8 + c * 8, c * 4)
        self.up3 = nn.ConvTranspose2d(c * 4, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 4 + c * 4, c * 2)
        self.up2 = nn.ConvTranspose2d(c * 2, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 2 + c * 2, c)

        self.mask_head = nn.Conv2d(c, 1, 1)     # 黄斑区域 logits（全分辨率）
        self.heat_head = nn.Conv2d(c, 1, 1)     # 中心凹热图 logits
        self.vis_head = nn.Sequential(          # 可见性：编码器全局特征
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 8, 2),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))

        mask = self.mask_head(d2)
        heat = self.heat_head(d2)
        vis = self.vis_head(b)
        return mask, heat, vis


def make_gaussian_heatmap(size, cx, cy, sigma):
    """生成单峰高斯热图，size 为输出边长，(cx, cy) 为归一化坐标 [0,1]"""
    ys = torch.arange(size, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(size, dtype=torch.float32).view(1, -1)
    px, py = cx * size, cy * size
    return torch.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma ** 2))


def soft_argmax(heat):
    """对热图做空间期望得到归一化坐标，可导，无需额外回归分支"""
    b, _, h, w = heat.shape
    prob = torch.softmax(heat.view(b, -1), dim=1).view(b, 1, h, w)
    xs = torch.linspace(0, 1, w, device=heat.device).view(1, 1, 1, w)
    ys = torch.linspace(0, 1, h, device=heat.device).view(1, 1, h, 1)
    cx = (prob * xs).sum(dim=(2, 3))
    cy = (prob * ys).sum(dim=(2, 3))
    return torch.cat([cx, cy], dim=1)


if __name__ == '__main__':
    net = MaculaFoveaNet()
    m, h, v = net(torch.randn(2, 3, 512, 512))
    print('mask:', tuple(m.shape), 'heat:', tuple(h.shape), 'vis:', tuple(v.shape))
    print('coord:', soft_argmax(h).tolist())
