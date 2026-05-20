"""生成训练曲线 + 对比 baseline 图表."""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CODE)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = os.path.join(_ROOT, 'runs', 'figures')
os.makedirs(FIG, exist_ok=True)


def load(p):
    if not os.path.exists(p): return None
    with open(p) as f:
        return json.load(f)


def training_curves():
    runs = {
        'Baseline R50': load(os.path.join(_ROOT, 'runs/baseline/history.json')),
        'Baseline Swin-T': load(os.path.join(_ROOT, 'runs/baseline_swin/history.json')),
        'RBP R50': load(os.path.join(_ROOT, 'runs/rbp/history.json')),
        'RBP Swin-T': load(os.path.join(_ROOT, 'runs/rbp_swin/history.json')),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, h in runs.items():
        if not h: continue
        e = [r['epoch'] for r in h]
        axes[0].plot(e, [r['mIoU']*100 for r in h], label=name, lw=1.8)
        axes[1].plot(e, [r['mAcc']*100 for r in h], label=name, lw=1.8)
    axes[0].axhline(25.08, color='gray', ls='--', alpha=0.5, label='DeepLabV3+ R50 (25.08)')
    axes[0].axhline(27.18, color='gray', ls=':', alpha=0.5, label='DeepLabV3+ R101 (27.18)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('val mIoU (%)')
    axes[0].set_title('Validation mIoU')
    axes[0].grid(alpha=0.3); axes[0].legend(loc='lower right', fontsize=9)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('val mAcc (%)')
    axes[1].set_title('Validation mAcc')
    axes[1].grid(alpha=0.3); axes[1].legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIG, 'training_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {out}')


def comparison_bars():
    methods = [
        ('DeepLabV3 R50', 17.24, 37.95, 'gray'),
        ('DeepLabV3 R101', 20.72, 40.63, 'gray'),
        ('DeepLabV3+ R50', 25.08, 40.66, 'lightblue'),
        ('DeepLabV3+ R101', 27.18, 42.29, 'lightblue'),
        ('Ours: Swin-T', 31.89, 40.94, 'darkorange'),
        ('SAN ViT-B/16', 34.79, 50.19, 'lightgreen'),
        ('SAN ViT-L/14', 36.91, 52.81, 'lightgreen'),
        ('SegNext MSCAN-L', 44.52, 59.95, 'gold'),
    ]
    labels = [m[0] for m in methods]
    miou = [m[1] for m in methods]
    macc = [m[2] for m in methods]
    colors = [m[3] for m in methods]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - w/2, miou, w, color=colors, label='mIoU', edgecolor='black', linewidth=0.6)
    b2 = ax.bar(x + w/2, macc, w, color=colors, alpha=0.6, label='mAcc',
                edgecolor='black', linewidth=0.6, hatch='//')
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.4, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Score (%)')
    ax.set_title('PlantSeg v3: Methods Comparison (Test Set)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(FIG, 'comparison_bars.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {out}')


def per_class_iou():
    rt = load(os.path.join(_ROOT, 'runs/baseline_swin/test_results_tta_3scale_flip.json'))
    if not rt: return
    iou = np.array(rt['per_iou']) * 100
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(np.arange(len(iou)), iou, color='steelblue', alpha=0.85)
    ax.set_xlabel('Class ID')
    ax.set_ylabel('IoU (%)')
    ax.set_title(f'Per-class IoU on Test (Swin-T baseline + TTA, mean = {iou.mean():.2f}%)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG, 'per_class_iou.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {out}')


if __name__ == '__main__':
    training_curves()
    comparison_bars()
    per_class_iou()
