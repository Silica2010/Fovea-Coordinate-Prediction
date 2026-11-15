# -*- coding: utf-8 -*-
"""实验三 单图推理：黄斑区域定位 → 中心凹坐标输出

输出规则与任务定义一致：
- 判为不可见，或热图响应过弱、黄斑区域面积为 0 → 坐标为 (0, 0)；
- 否则输出原图像素坐标。
"""

import argparse

import cv2
import numpy as np
import torch

from dataset import FoveaDataset
from model import MaculaFoveaNet, soft_argmax


def preprocess(path, size):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError('无法读取图像: %s' % path)
    h0, w0 = img.shape[:2]
    img = FoveaDataset._preprocess(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
          np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0), (w0, h0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--weights', type=str, default='best_fovea_model.pth')
    ap.add_argument('--mask_thr', type=float, default=0.5)
    ap.add_argument('--min_area_ratio', type=float, default=0.01,
                    help='黄斑区域最小面积占比，低于该值视为未检出')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.weights, map_location=device)
    size = ckpt.get('size', 512)
    model = MaculaFoveaNet().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    x, (w0, h0) = preprocess(args.image, size)
    with torch.no_grad():
        x = x.to(device)
        mask_logits, heat_logits, vis_logits = model(x)
        prob_mask = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
        prob_heat = torch.sigmoid(heat_logits)[0, 0].cpu().numpy()
        vis_prob = torch.softmax(vis_logits, dim=1)[0].cpu().numpy()
        coord = soft_argmax(heat_logits).cpu().numpy()[0]

    # 第一段：黄斑区域检测，给出区域面积与外接框
    binary = (prob_mask > args.mask_thr).astype(np.uint8)
    area_ratio = float(binary.sum()) / binary.size
    if binary.sum() > 0:
        ys, xs = np.where(binary > 0)
        box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    else:
        box = None
    print('黄斑区域面积占比：%.4f，外接框（512 坐标系）：%s' % (area_ratio, box))

    # 第二段：中心凹坐标输出
    visible = vis_prob[1] >= 0.5 and area_ratio >= args.min_area_ratio
    peak = float(prob_heat.max())
    print('可见性概率：%.3f（阈值 0.5），热图峰值：%.3f' % (vis_prob[1], peak))

    if not visible or peak < 0.2:
        print('判定为不可见，坐标：(0, 0)')
        return

    px = coord[0] * w0
    py = coord[1] * h0
    print('中心凹坐标（原图像素）：(%.1f, %.1f)，图像尺寸 %dx%d' % (px, py, w0, h0))
    print('归一化坐标：(%.4f, %.4f)' % (coord[0], coord[1]))

    # 若黄斑框存在，检查坐标是否落在框内，作为一致性提示
    if box is not None:
        nx, ny = coord[0] * size, coord[1] * size
        inside = box[0] <= nx <= box[2] and box[1] <= ny <= box[3]
        print('坐标是否位于黄斑框内：%s' % ('是' if inside else '否（建议复核）'))

    # 可视化
    vis_img = (prob_heat * 255).astype(np.uint8)
    cv2.imwrite('fovea_heat.png', vis_img)
    print('热图已保存：fovea_heat.png')


if __name__ == '__main__':
    main()
