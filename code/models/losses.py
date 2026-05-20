"""损失函数集合.

按方法论 PDF:
    L_total = L_region + lambda1 * L_boundary + lambda2 * L_cons + lambda3 * L_proto

L_region   = w_ce*CE + w_dice*Dice + w_ft*FocalTversky
L_boundary = w_l1*SmoothL1(distance_field) + w_dice*BandDice
L_cons     = |Sobel(P_region) - Sobel(P_boundary)|
L_proto    = NT-Xent over (病斑内部 / 边界 / 背景) prototypes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionLoss(nn.Module):
    """多类区域损失: CE + Dice + Focal Tversky."""
    def __init__(self, num_classes, ignore_index=255,
                 w_ce=1.0, w_dice=1.0, w_ft=0.5,
                 ft_alpha=0.7, ft_beta=0.3, ft_gamma=0.75,
                 class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore = ignore_index
        self.w_ce = w_ce
        self.w_dice = w_dice
        self.w_ft = w_ft
        self.ft_alpha = ft_alpha
        self.ft_beta = ft_beta
        self.ft_gamma = ft_gamma
        self.cw = class_weights
        self.ls = label_smoothing

    def _to_onehot(self, target):
        valid = (target != self.ignore)
        t = target.clone()
        t[~valid] = 0
        oh = F.one_hot(t, self.num_classes).permute(0, 3, 1, 2).float()
        oh = oh * valid.unsqueeze(1).float()
        return oh, valid

    def dice_loss(self, prob, oh, valid):
        """Multi-class Dice (mean over classes, ignoring background optional)."""
        v = valid.unsqueeze(1).float()
        inter = (prob * oh * v).sum(dim=(0, 2, 3))
        denom = (prob * v).sum(dim=(0, 2, 3)) + (oh * v).sum(dim=(0, 2, 3))
        dice = (2 * inter + 1e-6) / (denom + 1e-6)
        return 1 - dice.mean()

    def focal_tversky(self, prob, oh, valid):
        v = valid.unsqueeze(1).float()
        TP = (prob * oh * v).sum(dim=(0, 2, 3))
        FP = (prob * (1 - oh) * v).sum(dim=(0, 2, 3))
        FN = ((1 - prob) * oh * v).sum(dim=(0, 2, 3))
        tv = (TP + 1e-6) / (TP + self.ft_alpha * FN + self.ft_beta * FP + 1e-6)
        loss = (1 - tv) ** self.ft_gamma
        return loss.mean()

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore,
                             weight=self.cw, label_smoothing=self.ls)
        prob = F.softmax(logits, dim=1)
        oh, valid = self._to_onehot(target)
        dice = self.dice_loss(prob, oh, valid)
        ft = self.focal_tversky(prob, oh, valid)
        return self.w_ce * ce + self.w_dice * dice + self.w_ft * ft


class BoundaryLoss(nn.Module):
    """边界距离场损失: SmoothL1 + Band Dice."""
    def __init__(self, w_l1=1.0, w_dice=0.5, band_width=3):
        super().__init__()
        self.w_l1 = w_l1
        self.w_dice = w_dice
        self.band = band_width

    @staticmethod
    def compute_distance_field(mask_binary, normalize=True):
        """Approximate signed distance field (CPU friendly).
        mask_binary: (B, H, W) {0, 1}
        Returns: (B, H, W) in [-1, 1]
        """
        from scipy.ndimage import distance_transform_edt
        b, h, w = mask_binary.shape
        out = torch.zeros_like(mask_binary, dtype=torch.float32)
        m = mask_binary.cpu().numpy()
        for i in range(b):
            mi = m[i].astype(bool)
            if mi.any() and (~mi).any():
                d_in = distance_transform_edt(mi)
                d_out = distance_transform_edt(~mi)
                d = d_in - d_out
                d = d / (max(abs(d.min()), abs(d.max())) + 1e-6)
                out[i] = torch.from_numpy(d).float()
        return out

    def forward(self, pred_boundary, target_distance_field, target_mask):
        """
        pred_boundary: (B, 1, H, W) in [-1, 1]
        target_distance_field: (B, H, W) in [-1, 1]
        target_mask: (B, H, W) {0, 1, 255-ignore}
        """
        valid = (target_mask != 255).unsqueeze(1).float()
        pb = pred_boundary
        tb = target_distance_field.unsqueeze(1)
        l1 = (F.smooth_l1_loss(pb, tb, reduction='none') * valid).sum() / (valid.sum() + 1e-6)

        in_band = (tb.abs() < (self.band / 50.0)).float() * valid
        pred_in = (pb > 0).float()
        targ_in = (tb > 0).float()
        inter = (pred_in * targ_in * in_band).sum()
        denom = (pred_in * in_band).sum() + (targ_in * in_band).sum()
        dice = 1 - (2 * inter + 1e-6) / (denom + 1e-6)

        return self.w_l1 * l1 + self.w_dice * dice


class ConsistencyLoss(nn.Module):
    """Sobel 梯度一致性: 区域预测的边缘 vs 边界预测的梯度."""
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sy = sx.t()
        self.register_buffer('sx', sx.view(1, 1, 3, 3))
        self.register_buffer('sy', sy.view(1, 1, 3, 3))

    def sobel(self, x):
        gx = F.conv2d(x, self.sx, padding=1)
        gy = F.conv2d(x, self.sy, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, logits, pred_boundary, target_mask):
        prob = F.softmax(logits, dim=1)
        fg_prob = 1 - prob[:, 0:1, :, :]
        valid = (target_mask != 255).unsqueeze(1).float()

        gr = self.sobel(fg_prob)
        gb = self.sobel(pred_boundary)
        diff = (gr - gb).abs() * valid
        return diff.sum() / (valid.sum() + 1e-6)


class PrototypeLoss(nn.Module):
    """三原型对比 (病斑内部 / 边界 / 背景).

    feat: (B, D, H, W)  L2-normalized
    target_mask: (B, H, W) integer labels
    target_boundary: (B, H, W) {0, 1} 边界 mask
    """
    def __init__(self, temperature=0.1, max_pixels_per_class=512, ignore_index=255):
        super().__init__()
        self.tau = temperature
        self.max_per = max_pixels_per_class
        self.ignore = ignore_index

    def forward(self, feat, target_mask, target_boundary):
        B, D, H, W = feat.shape
        device = feat.device
        flat = feat.permute(0, 2, 3, 1).reshape(-1, D)
        m_flat = target_mask.reshape(-1)
        b_flat = target_boundary.reshape(-1)

        valid = (m_flat != self.ignore)
        is_bg = (m_flat == 0) & valid & (b_flat == 0)
        is_bd = (b_flat == 1) & valid
        is_in = (m_flat > 0) & valid & (b_flat == 0)

        protos = []
        for mask in (is_in, is_bd, is_bg):
            if mask.any():
                idx = mask.nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() > self.max_per:
                    perm = torch.randperm(idx.numel(), device=device)[:self.max_per]
                    idx = idx[perm]
                p = flat[idx].mean(dim=0)
                protos.append(F.normalize(p, dim=0))
            else:
                protos.append(torch.zeros(D, device=device))
        P = torch.stack(protos, dim=0)

        labels = []
        idx_pool = []
        for cls_id, mask in enumerate((is_in, is_bd, is_bg)):
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > self.max_per:
                perm = torch.randperm(idx.numel(), device=device)[:self.max_per]
                idx = idx[perm]
            idx_pool.append(idx)
            labels.append(torch.full((idx.numel(),), cls_id, dtype=torch.long, device=device))
        if not idx_pool:
            return feat.new_zeros(())

        sample_idx = torch.cat(idx_pool, dim=0)
        sample_labels = torch.cat(labels, dim=0)
        sample_feats = flat[sample_idx]
        logits = (sample_feats @ P.t()) / self.tau
        return F.cross_entropy(logits, sample_labels)


if __name__ == '__main__':
    B, C, H, W = 2, 115, 64, 64
    logits = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))
    target[0, 0, 0] = 255
    boundary_pred = torch.randn(B, 1, H, W).tanh()
    boundary_gt = torch.randn(B, H, W).tanh()
    feat = F.normalize(torch.randn(B, 32, H, W), dim=1)

    rl = RegionLoss(num_classes=C)
    bl = BoundaryLoss()
    cl = ConsistencyLoss()
    pl = PrototypeLoss()

    print('region:', rl(logits, target).item())
    print('boundary:', bl(boundary_pred, boundary_gt, target).item())
    print('cons:', cl(logits, boundary_pred, target).item())
    boundary_gt_mask = (boundary_gt.abs() < 0.05).long()
    print('proto:', pl(feat, target, boundary_gt_mask).item())
