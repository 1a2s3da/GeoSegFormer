"""汇总所有实验结果生成最终报告."""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CODE)

OUT = os.path.join(_ROOT, 'runs', 'final_report.md')


def load(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def best_from(history, key='mIoU'):
    if not history:
        return None
    return max(history, key=lambda x: x.get(key, 0))


def main():
    bh_r50 = load(os.path.join(_ROOT, 'runs/baseline/history.json'), [])
    bh_swin = load(os.path.join(_ROOT, 'runs/baseline_swin/history.json'), [])
    bh_swinb = load(os.path.join(_ROOT, 'runs/baseline_swin_b/history.json'), [])
    rh_r50 = load(os.path.join(_ROOT, 'runs/rbp/history.json'), [])
    rh_swin = load(os.path.join(_ROOT, 'runs/rbp_swin/history.json'), [])

    bt_r50 = load(os.path.join(_ROOT, 'runs/baseline/test_results.json'))
    bt_swin_n = load(os.path.join(_ROOT, 'runs/baseline_swin/test_results_notta.json'))
    bt_swin_t = load(os.path.join(_ROOT, 'runs/baseline_swin/test_results_tta_3scale_flip.json'))
    bt_swinb_n = load(os.path.join(_ROOT, 'runs/baseline_swin_b/test_results_notta.json'))
    bt_swinb_t = load(os.path.join(_ROOT, 'runs/baseline_swin_b/test_results_tta_3scale_flip.json'))
    rt_r50 = load(os.path.join(_ROOT, 'runs/rbp/test_results.json'))
    rt_swin_n = load(os.path.join(_ROOT, 'runs/rbp_swin/test_results_notta.json'))
    rt_swin_t = load(os.path.join(_ROOT, 'runs/rbp_swin/test_results_tta_3scale_flip.json'))

    L = []
    L.append('# PlantSeg v3 植物叶片病斑分割 - 实验结果\n\n')
    L.append('数据集: PlantSeg v3 (115 类: 1 背景 + 114 病害, 7916 train / 1247 val / 2295 test)\n\n')

    L.append('## 1. 测试集主结果\n\n')
    L.append('| 模型 | Encoder | Params | TTA | mIoU | mAcc | aAcc |\n')
    L.append('|---|---|---|---|---|---|---|\n')
    rows = [
        ('Baseline UNet', 'ResNet50', '43.87M', '-', bt_r50),
        ('RBP-UNet', 'ResNet50', '44.07M', '-', rt_r50),
        ('Baseline UNet', 'Swin-T', '34.54M', '-', bt_swin_n),
        ('Baseline UNet', 'Swin-T', '34.54M', 'MS+Flip', bt_swin_t),
        ('RBP-UNet', 'Swin-T', '34.65M', '-', rt_swin_n),
        ('RBP-UNet', 'Swin-T', '34.65M', 'MS+Flip', rt_swin_t),
        ('Baseline UNet', 'Swin-B', '95.22M', '-', bt_swinb_n),
        ('Baseline UNet', 'Swin-B', '95.22M', 'MS+Flip', bt_swinb_t),
    ]
    best_idx = -1
    best_miou = -1
    for i, (n, e, p, tta, r) in enumerate(rows):
        if r and r['mIoU'] > best_miou:
            best_miou = r['mIoU']
            best_idx = i
    for i, (n, e, p, tta, r) in enumerate(rows):
        if not r: continue
        line = f'| {n} | {e} | {p} | {tta} | {r["mIoU"]:.4f} | {r["mAcc"]:.4f} | {r["aAcc"]:.4f} |\n'
        if i == best_idx:
            line = f'| **{n}** | **{e}** | **{p}** | **{tta}** | **{r["mIoU"]:.4f}** | **{r["mAcc"]:.4f}** | **{r["aAcc"]:.4f}** |\n'
        L.append(line)

    L.append('\n## 2. PlantSeg 论文 Baselines (官方对照)\n\n')
    L.append('| Method | Encoder | mIoU | mAcc |\n|---|---|---|---|\n')
    paper_rows = [
        ('DeepLabV3', 'ResNet50', 17.24, 37.95),
        ('DeepLabV3', 'ResNet101', 20.72, 40.63),
        ('DeepLabV3+', 'ResNet50', 25.08, 40.66),
        ('DeepLabV3+', 'ResNet101', 27.18, 42.29),
        ('SAN', 'ViT-B/16', 34.79, 50.19),
        ('SAN', 'ViT-L/14', 36.91, 52.81),
        ('SegNext', 'MSCAN-L', 44.52, 59.95),
    ]
    for n, e, m, a in paper_rows:
        L.append(f'| {n} | {e} | {m:.2f} | {a:.2f} |\n')

    L.append('\n## 3. 我们的方法 vs Baselines\n\n')
    if bt_swin_t:
        miou = bt_swin_t['mIoU'] * 100
        macc = bt_swin_t['mAcc'] * 100
        L.append(f'我们的最佳 **Baseline UNet (Swin-T) + TTA** mIoU = **{miou:.2f}%**, mAcc = **{macc:.2f}%**\n\n')
        all_methods = [(n, m, a) for (n, _, m, a) in paper_rows] + \
                      [('**本文 Swin-T + TTA**', miou, macc)]
        all_methods.sort(key=lambda x: x[1])
        L.append('| 排名 | Method | mIoU | mAcc |\n|---|---|---|---|\n')
        for i, (n, m, a) in enumerate(all_methods):
            L.append(f'| {i+1} | {n} | {m:.2f} | {a:.2f} |\n')

    L.append('\n## 4. 验证集训练曲线 (best epoch)\n\n')
    L.append('| 模型 | best epoch | val mIoU | val mAcc |\n|---|---|---|---|\n')
    for name, h in [('Baseline R50', bh_r50), ('Baseline Swin-T', bh_swin),
                    ('Baseline Swin-B', bh_swinb),
                    ('RBP R50', rh_r50), ('RBP Swin-T', rh_swin)]:
        b = best_from(h)
        if b:
            L.append(f'| {name} | {b["epoch"]} | {b["mIoU"]:.4f} | {b["mAcc"]:.4f} |\n')

    L.append('\n## 5. 关键发现\n\n')
    L.append('1. **R50-UNet baseline mIoU = 20.98**, 已超 DeepLabV3 R50 (17.24).\n')
    L.append('2. **Swin-T 替换 ResNet50 提升显著**: baseline mIoU 20.98 → 31.31 (+10.33).\n')
    L.append('3. **多任务 RBP loss 在该数据集上拖累主分支**: '
             'R50 上 -4.49 mIoU, Swin-T 上 -7.21 mIoU. '
             '原因是 115 类极不均衡, 边界/原型辅助任务分散主任务. **这是真实负面发现**.\n')
    L.append('4. **TTA (3 scales + flip) 提升约 0.5-1.0 mIoU**: 多尺度集成有效.\n')
    L.append('5. **Swin-B (95M) 因 epoch 不够 (35) 未充分收敛**, 仅 27.30 mIoU. 完整训练 (60+ epoch) 预计可超 Swin-T.\n')
    L.append('6. **最终结果: Swin-T baseline + TTA = 31.89 mIoU**, 超过所有 DeepLabV3 系列 (DeepLabV3+ R101 27.18 高 +4.71).\n')

    L.append('\n## 6. 训练配置\n\n')
    L.append('- Optimizer: SGD (momentum=0.9, nesterov, weight_decay=1e-4)\n')
    L.append('- LR: 0.005-0.01, poly schedule (power=0.9), backbone lr×0.1\n')
    L.append('- Batch size: 8-12, Image size: 320×320, Epochs: 35-50\n')
    L.append('- Loss: CE + Dice (+ Focal Tversky for RBP)\n')
    L.append('- Aug: hflip, vflip, colorjitter; RBP 额外计算 SDF / 边界 mask\n')
    L.append('- TTA: 3 scales (0.75, 1.0, 1.25) + horizontal flip\n')

    L.append('\n## 7. 复现\n\n')
    L.append('```bash\ncd plant_disease_seg\npip install -r requirements.txt\n')
    L.append('python3 code/train/train_baseline_v2.py --epochs 40\n')
    L.append('python3 code/train/train_swin_baseline.py --epochs 40\n')
    L.append('python3 code/train/train_rbp_v2.py --epochs 50 --lambda_b 0.3 --lambda_p 0.1\n')
    L.append('python3 code/train/train_swin_rbp.py --epochs 40\n')
    L.append('python3 code/evaluation/eval_test_swin_baseline.py --split test\n```\n')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.writelines(L)
    print(f'saved: {OUT}')


if __name__ == '__main__':
    main()
