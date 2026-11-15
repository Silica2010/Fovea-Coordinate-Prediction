# -*- coding: utf-8 -*-
"""实验三 数据集：眼底图 + 中心凹坐标 (x, y)

csv 约定：image_path,x,y
- x, y 为原图像素坐标；x = y = 0 表示中心凹不可见，监督目标为 (0, 0)。
- 训练时坐标归一化到 [0,1]，避免不同分辨率带来的尺度漂移。
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from model import make_gaussian_heatmap


class FoveaDataset(Dataset):
    def __init__(self, csv_path, size=512, train=False,
                 sigma_heat=0.02, sigma_mask=0.06):
        self.size = size
        self.train = train
        self.sigma_heat = sigma_heat
        self.sigma_mask = sigma_mask
        self.samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            if header[:3] != ['image_path', 'x', 'y']:
                raise ValueError('csv 列名应为 image_path,x,y，实际为 %s' % header)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, x, y = line.split(',')[:3]
                self.samples.append((path, float(x), float(y)))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _preprocess(img):
        """眼底图常见黑边与圆形视野：先裁掉全黑边框，再做 CLAHE 增强血管与黄斑对比"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows = np.where(gray.max(axis=1) > 15)[0]
        cols = np.where(gray.max(axis=0) > 15)[0]
        if len(rows) > 0 and len(cols) > 0:
            img = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def __getitem__(self, idx):
        path, x, y = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError('无法读取图像: %s' % path)
        h0, w0 = img.shape[:2]
        img = self._preprocess(img)
        h1, w1 = img.shape[:2]
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
              np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = torch.from_numpy(img.transpose(2, 0, 1))

        visible = (x > 0) or (y > 0)
        if visible:
            nx = min(max(x / w0, 0.0), 1.0)
            ny = min(max(y / h0, 0.0), 1.0)
        else:
            nx, ny = 0.0, 0.0

        heat = make_gaussian_heatmap(self.size, nx, ny, self.sigma_heat * self.size)
        mask = make_gaussian_heatmap(self.size, nx, ny, self.sigma_mask * self.size)
        if not visible:                       # 不可见样本的监督目标全零
            heat = torch.zeros_like(heat)
            mask = torch.zeros_like(mask)

        coord = torch.tensor([nx, ny], dtype=torch.float32)
        return {
            'image': tensor,
            'mask': mask.unsqueeze(0),
            'heat': heat.unsqueeze(0),
            'coord': coord,
            'visible': torch.tensor(1 if visible else 0, dtype=torch.long),
            'orig_size': torch.tensor([w0, h0], dtype=torch.float32),
            'path': path,
        }


def collate_list(batch):
    """path 是字符串，默认 collate 会把它拼成 list，这里保持原样"""
    out = {}
    for key in batch[0].keys():
        if key == 'path':
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        ds = FoveaDataset(sys.argv[1])
        item = ds[0]
        for k, v in item.items():
            print(k, getattr(v, 'shape', v))
