"""植物病斑分割项目统一入口.

Usage:
    python main.py <command> [extra args]

Commands:
    train-baseline   训练 BaselineUNet (CE+Dice)
    train-rbp        训练完整 RBP-UNet (区域+边界+一致性+原型)
    eval-baseline    用 best baseline checkpoint 在 test 上评估
    eval-rbp         用 best RBP checkpoint 在 test 上评估
    pipeline         baseline + rbp 顺序训练
"""
import os, sys, argparse, runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def run(mod, extra):
    sys.argv = [mod] + (extra or [])
    runpy.run_module(mod, run_name='__main__')


def cmd_train_baseline(args):
    run('train.train_baseline', args.extra)

def cmd_train_rbp(args):
    run('train.train_rbp', args.extra)

def cmd_eval_baseline(args):
    run('evaluation.eval_test', ['--ckpt_kind', 'baseline'] + (args.extra or []))

def cmd_eval_rbp(args):
    run('evaluation.eval_test', ['--ckpt_kind', 'rbp'] + (args.extra or []))

def cmd_pipeline(args):
    print('[1/2] BASELINE'); cmd_train_baseline(args)
    print('[2/2] RBP-UNet'); cmd_train_rbp(args)


COMMANDS = {
    'train-baseline': cmd_train_baseline,
    'train-rbp': cmd_train_rbp,
    'eval-baseline': cmd_eval_baseline,
    'eval-rbp': cmd_eval_rbp,
    'pipeline': cmd_pipeline,
}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument('command', choices=list(COMMANDS.keys()))
    ap.add_argument('extra', nargs=argparse.REMAINDER, help='passed to underlying script')
    args = ap.parse_args()
    COMMANDS[args.command](args)


if __name__ == '__main__':
    main()
