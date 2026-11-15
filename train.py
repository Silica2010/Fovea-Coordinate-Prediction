# -*- coding: utf-8 -*-
"""实验三 训练脚本：黄斑区域检测 + 中心凹坐标回归 + 可见性判别"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FoveaDataset, collate_list
from model import MaculaFoveaNet, soft_argmax


def build_loss(mask_logits, heat_logits, vis_logits, pred_coord, batch, cfg):
    bce_mask = F.binary_cross_entropy_with_logits(mask_logits, batch['mask'])
    # 热图正样本极少（中心只有一个峰），用 pos_weight 抬高正类权重
    pos_weight = torch.tensor([cfg['heat_pos_weight']], device=heat_logits.device)
    bce_heat = F.binary_cross_entropy_with_logits(heat_logits, batch['heat'],
                                                  pos_weight=pos_weight)
    ce_vis = F.cross_entropy(vis_logits, batch['visible'])
    visible_mask = batch['visible'].float().unsqueeze(1)
    l1_coord = (torch.abs(pred_coord - batch['coord']) * visible_mask).sum() / \
               (visible_mask.sum() + 1e-6)
    total = cfg['w_mask'] * bce_mask + cfg['w_heat'] * bce_heat + \
        cfg['w_vis'] * ce_vis + cfg['w_coord'] * l1_coord
    return total, {'mask': bce_mask.item(), 'heat': bce_heat.item(),
                   'vis': ce_vis.item(), 'coord': l1_coord.item()}


def accuracy_of(pred, true):
    return float((pred == true).mean()) if len(true) else 0.0


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    dists, vis_pred, vis_true = [], [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        mask_logits, heat_logits, vis_logits = model(batch['image'])
        pred = soft_argmax(heat_logits).cpu()
        pred_vis = vis_logits.argmax(1).cpu()
        vis_pred += pred_vis.tolist()
        vis_true += batch['visible'].cpu().tolist()
        coords = batch['coord'].cpu()
        size = batch['orig_size'].cpu()          # (W, H)
        for i in range(coords.size(0)):
            if batch['visible'][i].item() == 0:
                continue
            dx = (pred[i, 0] - coords[i, 0]) * size[i, 0]
            dy = (pred[i, 1] - coords[i, 1]) * size[i, 1]
            dists.append(float(torch.sqrt(dx * dx + dy * dy)))

    dists = np.array(dists) if dists else np.array([np.inf])
    vis_pred = np.array(vis_pred)
    vis_true = np.array(vis_true)
    acc = accuracy_of(vis_pred, vis_true)
    tp = int(((vis_pred == 1) & (vis_true == 1)).sum())
    fp = int(((vis_pred == 1) & (vis_true == 0)).sum())
    fn = int(((vis_pred == 0) & (vis_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        'mean_dist_px': float(dists.mean()),
        'hit@5': float((dists <= 5).mean()),
        'hit@10': float((dists <= 10).mean()),
        'hit@20': float((dists <= 20).mean()),
        'vis_acc': acc,
        'vis_precision': precision,
        'vis_recall': recall,
        'score': float((dists <= 10).mean() * 0.6 + acc * 0.4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', type=str, required=True)
    ap.add_argument('--val_csv', type=str, required=True)
    ap.add_argument('--config', type=str, default='configs/train_config.json')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--patience', type=int, default=12)
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = FoveaDataset(args.train_csv, args.size, train=True,
                            sigma_heat=cfg['sigma_heat'], sigma_mask=cfg['sigma_mask'])
    val_ds = FoveaDataset(args.val_csv, args.size, train=False,
                          sigma_heat=cfg['sigma_heat'], sigma_mask=cfg['sigma_mask'])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=collate_list)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            collate_fn=collate_list)

    model = MaculaFoveaNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                          eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best = {'score': -1.0}
    bad = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        pbar = tqdm(train_loader, desc='ep%03d' % ep, ncols=110)
        for batch in pbar:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                mask_logits, heat_logits, vis_logits = model(batch['image'])
                pred_coord = soft_argmax(heat_logits.float())
                loss, parts = build_loss(mask_logits.float(), heat_logits.float(),
                                         vis_logits.float(), pred_coord, batch, cfg)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * batch['image'].size(0)
            n += batch['image'].size(0)
            pbar.set_postfix(loss='%.4f' % loss.item(),
                             coord='%.3f' % parts['coord'])
        scheduler.step()

        m = evaluate(model, val_loader, device, cfg)
        print('[ep%03d] train_loss=%.4f mae_px=%.2f hit@5=%.3f hit@10=%.3f hit@20=%.3f '
              'vis_acc=%.3f vis_recall=%.3f score=%.4f'
              % (ep, running / n, m['mean_dist_px'], m['hit@5'], m['hit@10'],
                 m['hit@20'], m['vis_acc'], m['vis_recall'], m['score']))

        if m['score'] > best['score']:
            best = dict(m)
            best['epoch'] = ep
            bad = 0
            torch.save({'model': model.state_dict(), 'cfg': cfg, 'size': args.size,
                        'metrics': {k: v for k, v in m.items()}},
                       'best_fovea_model.pth')
            print('  -> 保存 best_fovea_model.pth (score=%.4f)' % m['score'])
        else:
            bad += 1
            if bad >= args.patience:
                print('验证指标 %d 轮未提升，提前停止' % args.patience)
                break

    print('训练结束，最优：MAE %.2f px，hit@10 %.3f，可见性准确率 %.3f（epoch %d）'
          % (best['mean_dist_px'], best['hit@10'], best['vis_acc'], best['epoch']))


if __name__ == '__main__':
    main()
