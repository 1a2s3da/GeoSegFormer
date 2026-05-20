"""mIoU / mAcc 等语义分割指标 (基于混淆矩阵)."""
import numpy as np
import torch


class SegMetrics:
    def __init__(self, num_classes=115, ignore_index=255):
        self.n = num_classes
        self.ignore = ignore_index
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.n, self.n), dtype=np.int64)

    @torch.no_grad()
    def update(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        valid = (target != self.ignore)
        p = pred[valid].astype(np.int64)
        t = target[valid].astype(np.int64)
        idx = self.n * t + p
        bc = np.bincount(idx, minlength=self.n * self.n)
        self.cm += bc.reshape(self.n, self.n)

    def compute(self):
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        gt_sum = cm.sum(axis=1)
        pred_sum = cm.sum(axis=0)
        union = gt_sum + pred_sum - tp

        valid_cls = gt_sum > 0
        per_iou = np.zeros(self.n)
        per_iou[union > 0] = tp[union > 0] / union[union > 0]
        miou = per_iou[valid_cls].mean() if valid_cls.any() else 0.0

        per_acc = np.zeros(self.n)
        per_acc[gt_sum > 0] = tp[gt_sum > 0] / gt_sum[gt_sum > 0]
        macc = per_acc[valid_cls].mean() if valid_cls.any() else 0.0

        overall_acc = tp.sum() / (cm.sum() + 1e-9)
        return {
            'mIoU': float(miou),
            'mAcc': float(macc),
            'aAcc': float(overall_acc),
            'per_iou': per_iou,
            'per_acc': per_acc,
            'n_valid_classes': int(valid_cls.sum()),
        }


if __name__ == '__main__':
    m = SegMetrics(num_classes=5)
    m.update(np.array([[0, 1, 2], [3, 4, 0]]), np.array([[0, 1, 1], [3, 4, 4]]))
    res = m.compute()
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            print(f'{k}: {v}')
        else:
            print(f'{k}: {v}')
