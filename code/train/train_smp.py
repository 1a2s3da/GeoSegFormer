"""Train smp-based model (UperNet / SegFormer / etc.) on PlantSeg v3.

Standard recipe for transformer backbones:
    AdamW (lr=6e-5, wd=0.01, backbone lr x0.1)
  + linear warmup (1500 steps) + cosine decay
  + OHEM CE (top 25% hard pixels) + Dice
  + strong aug (RandomResizedCrop, hflip, vflip, colorjitter)
  + 448x448 resolution (2x the 320 that previous runs used)
  + AMP (float16)
"""
import os, sys, json, time, math, argparse, random
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CODE)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

os.environ.setdefault('HF_HUB_CACHE', '/root/autodl-tmp/hf_cache')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

from models.smp_models import SMPWrapper
from utils.metrics import SegMetrics
from data.plantseg import list_split


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PlantSegAug(Dataset):
    """Plant seg dataset with RandomResizedCrop + flips + colorjitter."""
    def __init__(self, root, split, img_size, num_classes=115, augment=True,
                 rrc_scale=(0.5, 1.5)):
        self.pairs = list_split(root, split)
        self.split = split
        self.size = img_size
        self.num_classes = num_classes
        self.augment = augment and (split == 'train')
        self.rrc_scale = rrc_scale

    def __len__(self):
        return len(self.pairs)

    def _load(self, img_path, mask_path):
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        return img, mask

    def _rrc(self, img, mask):
        W, H = img.size
        s = random.uniform(*self.rrc_scale)
        new_h, new_w = int(self.size * s), int(self.size * s)
        img = img.resize((max(new_w, self.size), max(new_h, self.size)), Image.BILINEAR)
        mask = mask.resize((max(new_w, self.size), max(new_h, self.size)), Image.NEAREST)
        W2, H2 = img.size
        x0 = random.randint(0, W2 - self.size)
        y0 = random.randint(0, H2 - self.size)
        img = img.crop((x0, y0, x0 + self.size, y0 + self.size))
        mask = mask.crop((x0, y0, x0 + self.size, y0 + self.size))
        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img, mask = self._load(img_path, mask_path)

        if self.augment:
            img, mask = self._rrc(img, mask)
            if random.random() < 0.5:
                img = TF.hflip(img); mask = TF.hflip(mask)
            if random.random() < 0.5:
                img = TF.vflip(img); mask = TF.vflip(mask)
            if random.random() < 0.7:
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
                img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
                img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
                img = TF.adjust_hue(img, random.uniform(-0.05, 0.05))
        else:
            img = img.resize((self.size, self.size), Image.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.NEAREST)

        img = TF.to_tensor(img)
        img = TF.normalize(img, MEAN, STD)
        m = np.array(mask, dtype=np.int64)
        m[m >= self.num_classes] = 0
        return {'image': img, 'mask': torch.from_numpy(m)}


class OHEMCEDiceLoss(nn.Module):
    """OHEM Cross-Entropy (top-k hard pixels) + multi-class Dice."""
    def __init__(self, num_classes=115, ignore_index=255,
                 ohem_keep_ratio=0.25, ohem_min_kept=100000,
                 w_ce=1.0, w_dice=1.0, label_smoothing=0.05,
                 class_weights=None):
        super().__init__()
        self.n = num_classes
        self.ig = ignore_index
        self.k = ohem_keep_ratio
        self.min_kept = ohem_min_kept
        self.w_ce = w_ce
        self.w_dice = w_dice
        self.ls = label_smoothing
        self.cw = class_weights

    def forward(self, logits, target):
        B, C, H, W = logits.shape
        ce_map = F.cross_entropy(logits, target, ignore_index=self.ig,
                                 weight=self.cw, label_smoothing=self.ls,
                                 reduction='none')
        valid = (target != self.ig)
        ce_flat = ce_map[valid]
        if ce_flat.numel() > self.min_kept:
            k = max(self.min_kept, int(ce_flat.numel() * self.k))
            k = min(k, ce_flat.numel())
            topk, _ = ce_flat.topk(k)
            ce_loss = topk.mean()
        else:
            ce_loss = ce_flat.mean() if ce_flat.numel() > 0 else logits.sum() * 0

        prob = F.softmax(logits, dim=1)
        t = target.clone()
        t[~valid] = 0
        oh = F.one_hot(t, self.n).permute(0, 3, 1, 2).float()
        v = valid.unsqueeze(1).float()
        inter = (prob * oh * v).sum(dim=(0, 2, 3))
        denom = (prob * v).sum(dim=(0, 2, 3)) + (oh * v).sum(dim=(0, 2, 3))
        dice = (2 * inter + 1e-6) / (denom + 1e-6)
        dice_loss = 1 - dice.mean()

        return self.w_ce * ce_loss + self.w_dice * dice_loss


def build_lr_schedule(opt, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, amp=True):
    model.eval()
    metric = SegMetrics(num_classes)
    for batch in loader:
        x = batch['image'].to(device, non_blocking=True)
        y = batch['mask']
        if amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x)['logits']
        else:
            logits = model(x)['logits']
        pred = logits.argmax(dim=1).cpu()
        metric.update(pred.numpy(), y.numpy())
    return metric.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', type=str, default='segformer',
                    choices=['segformer', 'upernet', 'deeplabv3p', 'unet', 'manet'])
    ap.add_argument('--encoder', type=str, default='mit_b3')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=6e-5)
    ap.add_argument('--backbone_lr_mult', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--img_size', type=int, default=448)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--num_classes', type=int, default=115)
    ap.add_argument('--warmup_steps', type=int, default=1500)
    ap.add_argument('--ohem_ratio', type=float, default=0.25)
    ap.add_argument('--amp', action='store_true', default=True)
    ap.add_argument('--out_dir', type=str, default=None)
    ap.add_argument('--data_root', type=str,
                    default=os.path.join(_ROOT, 'data', 'plantsegv3'))
    ap.add_argument('--resume', type=str, default=None)
    args = ap.parse_args()

    tag = f'{args.arch}_{args.encoder.replace("/", "_")}'
    if args.out_dir is None:
        args.out_dir = os.path.join(_ROOT, 'runs', tag)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}, out: {args.out_dir}')

    tr_ds = PlantSegAug(args.data_root, 'train', args.img_size,
                        args.num_classes, augment=True)
    va_ds = PlantSegAug(args.data_root, 'val', args.img_size,
                        args.num_classes, augment=False)
    tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True,
                           drop_last=True, persistent_workers=True)
    va_loader = DataLoader(va_ds, batch_size=args.bs, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True,
                           persistent_workers=True)
    print(f'train: {len(tr_ds)}, val: {len(va_ds)}, img_size={args.img_size}, bs={args.bs}')

    model = SMPWrapper(arch=args.arch, encoder=args.encoder,
                       num_classes=args.num_classes,
                       encoder_weights='imagenet').to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'model: {args.arch}/{args.encoder}, params: {n_params:.2f}M')

    backbone_params = list(model.net.encoder.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith('net.encoder.')]
    opt = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * args.backbone_lr_mult},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=args.wd, betas=(0.9, 0.999))

    n_steps = len(tr_loader) * args.epochs
    sched = build_lr_schedule(opt, n_steps, args.warmup_steps)
    loss_fn = OHEMCEDiceLoss(num_classes=args.num_classes,
                             ohem_keep_ratio=args.ohem_ratio,
                             label_smoothing=0.05).to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    start_ep = 0
    best_miou = -1
    best_path = os.path.join(args.out_dir, 'best.pt')
    last_path = os.path.join(args.out_dir, 'last.pt')
    history = []
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        start_ep = ck['epoch']
        best_miou = ck.get('best_miou', -1)
        history = ck.get('history', [])
        print(f'resumed from {args.resume}, ep={start_ep}, best={best_miou:.4f}')

    step = start_ep * len(tr_loader)
    for ep in range(start_ep, args.epochs):
        model.train()
        t0 = time.time()
        tl, n = 0.0, 0
        for batch in tr_loader:
            x = batch['image'].to(device, non_blocking=True)
            y = batch['mask'].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=args.amp):
                logits = model(x)['logits']
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            tl += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = tl / max(n, 1)
        res = evaluate(model, va_loader, device, args.num_classes, amp=args.amp)
        elapsed = time.time() - t0
        cur_lr = opt.param_groups[1]['lr']
        print(f'Ep {ep+1:3d}/{args.epochs} | loss={train_loss:.4f} lr={cur_lr:.2e} '
              f'| val mIoU={res["mIoU"]:.4f} mAcc={res["mAcc"]:.4f} aAcc={res["aAcc"]:.4f} '
              f'| {elapsed:.1f}s', flush=True)
        history.append(dict(epoch=ep + 1, train_loss=train_loss,
                            mIoU=res['mIoU'], mAcc=res['mAcc'], aAcc=res['aAcc']))
        if res['mIoU'] > best_miou:
            best_miou = res['mIoU']
            torch.save({'model': model.state_dict(), 'epoch': ep + 1,
                        'mIoU': res['mIoU'], 'mAcc': res['mAcc'],
                        'arch': args.arch, 'encoder': args.encoder}, best_path)
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': ep + 1, 'best_miou': best_miou,
                    'arch': args.arch, 'encoder': args.encoder,
                    'history': history}, last_path)
        with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print(f'\nbest val mIoU = {best_miou:.4f}')


if __name__ == '__main__':
    main()
