"""
Master Attack Pipeline -- runs all 6 evaluation phases in sequence.

Results are saved to organized subfolders:
  results/01_physical/    -- NUAA printed-photo attack baseline
  results/02_fgsm/        -- Fast Gradient Sign Method sweep
  results/03_pgd/         -- Projected Gradient Descent sweep
  results/04_patch/       -- Adversarial ocular-region patch
  results/05_gradcam/     -- Grad-CAM XAI attribution maps
  results/06_comparative/ -- Cross-attack summary figures & tables

Usage:
    python main_attack.py                     # run all phases
    python main_attack.py --skip 05           # skip Grad-CAM (slow)
    python main_attack.py --only 01 06        # only physical + comparative
    python main_attack.py --only 06           # re-run comparative only (needs 01-04)
"""

import os
import sys
import time
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PHASES = {
    '01': ('01_physical_attack.py',   'Physical Print Attack Baseline'),
    '02': ('02_fgsm_attack.py',       'FGSM Digital Attack Sweep'),
    '03': ('03_pgd_attack.py',        'PGD Digital Attack Sweep'),
    '04': ('04_adversarial_patch.py', 'Adversarial Ocular-Region Patch'),
    '05': ('05_xai_gradcam.py',       'Grad-CAM XAI Attribution Maps'),
    '06': ('06_comparative_analysis.py', 'Comparative Analysis & Thesis Figures'),
}


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h}h {m:02d}m {s:02d}s' if h else f'{m}m {s:02d}s'


def run_phase(phase_id, script_name, description):
    script_path = os.path.join(BASE_DIR, script_name)
    print(f'\n{"=" * 65}')
    print(f'  PHASE {phase_id}: {description}')
    print(f'{"=" * 65}')

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f'\n  [OK] Phase {phase_id} completed in {fmt_time(elapsed)}')
        return True, elapsed
    else:
        print(f'\n  [FAIL] Phase {phase_id} exited with code {result.returncode} '
              f'after {fmt_time(elapsed)}')
        return False, elapsed


def print_output_tree():
    """Print a tree of all generated result files."""
    results_root = os.path.join(BASE_DIR, 'results')
    if not os.path.isdir(results_root):
        return
    print('\n  Generated outputs:')
    total_files = 0
    for subfolder in sorted(os.listdir(results_root)):
        sub_path = os.path.join(results_root, subfolder)
        if not os.path.isdir(sub_path):
            continue
        files = sorted(os.listdir(sub_path))
        if not files:
            continue
        print(f'\n    results/{subfolder}/')
        for f in files:
            fpath = os.path.join(sub_path, f)
            size  = os.path.getsize(fpath)
            size_str = f'{size/1024:.0f} KB' if size > 1024 else f'{size} B'
            print(f'      {f:<55} {size_str:>8}')
            total_files += 1
    print(f'\n  Total: {total_files} files')


def main():
    parser = argparse.ArgumentParser(description='Run adversarial attack evaluation pipeline.')
    group  = parser.add_mutually_exclusive_group()
    group.add_argument('--skip', nargs='+', metavar='ID',
                       help='Phase IDs to skip (e.g. --skip 05)')
    group.add_argument('--only', nargs='+', metavar='ID',
                       help='Phase IDs to run exclusively (e.g. --only 01 06)')
    args = parser.parse_args()

    if args.only:
        phases_to_run = {k: v for k, v in PHASES.items() if k in args.only}
    elif args.skip:
        phases_to_run = {k: v for k, v in PHASES.items() if k not in args.skip}
    else:
        phases_to_run = PHASES

    print('=' * 65)
    print('  ADVERSARIAL ATTACK EVALUATION -- BACHELOR THESIS')
    print('  Tilburg University -- Cagan Kilinc')
    print('=' * 65)
    print(f'\n  Phases to run: {list(phases_to_run.keys())}')
    print(f'  Scripts from:  {BASE_DIR}')
    print(f'  Results to:    {os.path.join(BASE_DIR, "results")}\n')

    pipeline_start = time.time()
    phase_log = []

    for phase_id in sorted(phases_to_run.keys()):
        script_name, description = phases_to_run[phase_id]
        ok, elapsed = run_phase(phase_id, script_name, description)
        phase_log.append((phase_id, description, ok, elapsed))
        if not ok:
            print(f'\n  [ABORT] Phase {phase_id} failed -- stopping pipeline.')
            print('  Fix the error and re-run with --only {' + ' '.join(
                k for k, _ in sorted(phases_to_run.items()) if k >= phase_id) + '}')
            break

    total_time = time.time() - pipeline_start

    # -- Final summary ---------------------------------------------------------
    print(f'\n{"=" * 65}')
    print('  PIPELINE SUMMARY')
    print(f'{"=" * 65}')
    all_ok = True
    for phase_id, desc, ok, elapsed in phase_log:
        status = '[OK  ]' if ok else '[FAIL]'
        if not ok:
            all_ok = False
        print(f'  {status} Phase {phase_id}: {desc:<40} {fmt_time(elapsed):>10}')
    print(f'{"=" * 65}')
    print(f'  Total elapsed: {fmt_time(total_time)}')

    if all_ok:
        print_output_tree()
        print('\n  All phases complete. Results are ready for thesis writing.')
    else:
        print('\n  Pipeline did not finish -- see error above.')

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
