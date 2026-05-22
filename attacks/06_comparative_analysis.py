"""
Comparative Analysis -- All Attacks x All Models.

Reads the JSON result files produced by scripts 01-04 and generates
the thesis-ready summary figures and tables.

Run AFTER scripts 01-04 have completed:
    python 06_comparative_analysis.py

Outputs -> results/06_comparative/
    06_summary_table.txt                -- human-readable master table
    06_summary_table_latex.txt          -- LaTeX tabular for thesis inclusion
    06_accuracy_drop_bar.png            -- grouped bar: accuracy drop per attack
    06_robustness_radar.png             -- spider chart: robustness profile
    06_asr_heatmap.png                  -- heatmap: ASR across model x attack
    06_conf_drop_heatmap.png            -- heatmap: confidence drop
    06_combined_epsilon_curves.png      -- FGSM + PGD epsilon curves combined
    06_model_ranking.png                -- bar chart: overall robustness ranking
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR     = os.path.dirname(__file__)
RESULTS_BASE = os.path.join(BASE_DIR, 'results')
RESULTS_DIR  = os.path.join(RESULTS_BASE, '06_comparative')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paths to JSON outputs from each script
PHYS_JSON  = os.path.join(RESULTS_BASE, '01_physical', '01_physical_attack_results.json')
FGSM_JSON  = os.path.join(RESULTS_BASE, '02_fgsm',    '02_fgsm_results.json')
PGD_JSON   = os.path.join(RESULTS_BASE, '03_pgd',     '03_pgd_results.json')
PATCH_JSON = os.path.join(RESULTS_BASE, '04_patch',   '04_patch_results.json')

MODEL_NAMES  = ['MobileNetV3', 'GhostNet', 'EfficientNet']
ATTACK_ORDER = ['Physical', 'FGSM (8/255)', 'PGD (8/255, 40)', 'Patch (eye)']
COLORS       = ['#1f77b4', '#ff7f0e', '#2ca02c']
EPS_KEYS     = ['eps_1_255', 'eps_2_255', 'eps_4_255', 'eps_8_255', 'eps_16_255']
EPS_LABELS   = ['1/255', '2/255', '4/255', '8/255', '16/255']


# =============================================================================
# Data loading
# =============================================================================

def load_results():
    results = {m: {} for m in MODEL_NAMES}

    if os.path.isfile(PHYS_JSON):
        phys = json.load(open(PHYS_JSON))
        for m in MODEL_NAMES:
            if m in phys:
                r = phys[m]
                results[m]['Physical'] = {
                    'clean_accuracy':      r.get('genuine_accuracy', 0),
                    'attacked_accuracy':   r.get('physical_success_rate', 0),
                    'accuracy_drop':       r.get('genuine_accuracy', 0) - r.get('physical_success_rate', 0),
                    'attack_success_rate': r.get('physical_success_rate', 0),
                    'avg_conf_clean':      r.get('avg_conf_genuine', 0),
                    'avg_conf_attacked':   r.get('avg_conf_impostor', 0),
                }
    else:
        print(f'  [WARN] Missing {PHYS_JSON}')

    if os.path.isfile(FGSM_JSON):
        fgsm = json.load(open(FGSM_JSON))
        for m in MODEL_NAMES:
            if m in fgsm and 'eps_8_255' in fgsm[m]:
                results[m]['FGSM (8/255)'] = fgsm[m]['eps_8_255']
    else:
        print(f'  [WARN] Missing {FGSM_JSON}')

    if os.path.isfile(PGD_JSON):
        pgd = json.load(open(PGD_JSON))
        for m in MODEL_NAMES:
            eps_key = 'eps_8_255'
            if 'epsilon_sweep' in pgd and m in pgd['epsilon_sweep']:
                if eps_key in pgd['epsilon_sweep'][m]:
                    results[m]['PGD (8/255, 40)'] = pgd['epsilon_sweep'][m][eps_key]
    else:
        print(f'  [WARN] Missing {PGD_JSON}')

    if os.path.isfile(PATCH_JSON):
        patch = json.load(open(PATCH_JSON))
        for m in MODEL_NAMES:
            if m in patch and 'eye' in patch[m]:
                results[m]['Patch (eye)'] = patch[m]['eye']
    else:
        print(f'  [WARN] Missing {PATCH_JSON}')

    return results


def load_epsilon_curves():
    fgsm_curves, pgd_curves = {}, {}
    if os.path.isfile(FGSM_JSON):
        fgsm = json.load(open(FGSM_JSON))
        for m in MODEL_NAMES:
            if m in fgsm:
                fgsm_curves[m] = fgsm[m]
    if os.path.isfile(PGD_JSON):
        pgd = json.load(open(PGD_JSON))
        for m in MODEL_NAMES:
            if m in pgd.get('epsilon_sweep', {}):
                pgd_curves[m] = pgd['epsilon_sweep'][m]
    return fgsm_curves, pgd_curves


# =============================================================================
# Summary tables
# =============================================================================

def print_and_save_summary_table(results, save_path):
    """Human-readable summary table saved to .txt"""
    lines = ['=' * 80,
             'SUMMARY TABLE -- Recognition Accuracy (%) Under Each Attack',
             '  Canonical point: eps=8/255, PGD 40 steps, eye-region patch',
             '=' * 80,
             f"  {'Attack':<22} {'MobileNetV3':>14} {'GhostNet':>14} {'EfficientNet':>14}",
             '-' * 80]

    for attack in ATTACK_ORDER:
        row = f"  {attack:<22}"
        for m in MODEL_NAMES:
            val = results[m].get(attack, {}).get('attacked_accuracy', float('nan'))
            row += f" {val:>13.2f}%"
        lines.append(row)

    lines.append('-' * 80)
    row = f"  {'Clean (baseline)':<22}"
    for m in MODEL_NAMES:
        ca = next((v['clean_accuracy'] for v in results[m].values()), float('nan'))
        row += f" {ca:>13.2f}%"
    lines.append(row)
    lines.append('=' * 80)

    lines.append('\nAttack Success Rate (%):')
    lines.append(f"  {'Attack':<22} {'MobileNetV3':>14} {'GhostNet':>14} {'EfficientNet':>14}")
    lines.append('-' * 80)
    for attack in ATTACK_ORDER:
        row = f"  {attack:<22}"
        for m in MODEL_NAMES:
            val = results[m].get(attack, {}).get('attack_success_rate', float('nan'))
            row += f" {val:>13.2f}%"
        lines.append(row)

    lines.append('\nConfidence Drop (Clean - Attacked):')
    lines.append(f"  {'Attack':<22} {'MobileNetV3':>14} {'GhostNet':>14} {'EfficientNet':>14}")
    lines.append('-' * 80)
    for attack in ATTACK_ORDER:
        row = f"  {attack:<22}"
        for m in MODEL_NAMES:
            r    = results[m].get(attack, {})
            drop = r.get('avg_conf_clean', 0) - r.get('avg_conf_attacked', 0)
            row += f" {drop:>+13.4f}"
        lines.append(row)
    lines.append('=' * 80)

    txt = '\n'.join(lines)
    print(txt)
    with open(save_path, 'w', encoding='utf-8') as fh:
        fh.write(txt)
    print(f'\n  Saved -> {save_path}')
    return txt


def save_latex_table(results, save_path):
    """
    LaTeX tabular environment ready to paste into thesis.
    Shows accuracy under each attack and ASR for all model x attack combinations.
    """
    col_spec = 'l' + 'r' * len(MODEL_NAMES)
    header   = ' & '.join(['Attack'] + MODEL_NAMES)

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Recognition Accuracy (\%) and Attack Success Rate (\%) at canonical '
        r'operating point ($\varepsilon = 8/255$, PGD 40 steps, eye-region patch).}',
        r'\label{tab:attack_results}',
        r'\begin{tabular}{' + col_spec + '}',
        r'\toprule',
        header + r' \\',
        r'\midrule',
        r'\multicolumn{' + str(len(MODEL_NAMES) + 1) + r'}{l}{\textit{Accuracy under attack (\%)}} \\',
    ]

    for attack in ATTACK_ORDER:
        row_vals = []
        for m in MODEL_NAMES:
            val = results[m].get(attack, {}).get('attacked_accuracy', float('nan'))
            row_vals.append(f'{val:.2f}')
        lines.append(f'{attack} & ' + ' & '.join(row_vals) + r' \\')

    # Clean baseline
    clean_vals = []
    for m in MODEL_NAMES:
        ca = next((v['clean_accuracy'] for v in results[m].values()), float('nan'))
        clean_vals.append(f'{ca:.2f}')
    lines.append(r'\midrule')
    lines.append(r'Clean (no attack) & ' + ' & '.join(clean_vals) + r' \\')

    lines += [
        r'\midrule',
        r'\multicolumn{' + str(len(MODEL_NAMES) + 1) + r'}{l}{\textit{Attack Success Rate (\%)}} \\',
    ]
    for attack in ATTACK_ORDER:
        row_vals = []
        for m in MODEL_NAMES:
            val = results[m].get(attack, {}).get('attack_success_rate', float('nan'))
            row_vals.append(f'{val:.2f}')
        lines.append(f'{attack} & ' + ' & '.join(row_vals) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    with open(save_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f'  Saved -> {save_path}')


# =============================================================================
# Figures
# =============================================================================

def plot_accuracy_drop_bar(results, save_path):
    n_attacks = len(ATTACK_ORDER)
    x     = np.arange(n_attacks)
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, ylabel, title in zip(
        axes,
        ['attacked_accuracy', 'accuracy_drop'],
        ['Accuracy Under Attack (%)', 'Accuracy Drop vs. Clean (pp)'],
        ['Recognition Accuracy per Attack', 'Accuracy Drop per Attack']
    ):
        for i, (m, color) in enumerate(zip(MODEL_NAMES, COLORS)):
            vals = [results[m].get(a, {}).get(metric, 0) for a in ATTACK_ORDER]
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          label=m, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(ATTACK_ORDER, rotation=12, ha='right', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        if metric == 'attacked_accuracy':
            ax.set_ylim(0, 115)

    plt.suptitle('Adversarial Robustness -- All Attacks x All Models',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_radar(results, save_path):
    """Spider chart: post-attack accuracy per attack (higher = more robust)."""
    labels  = ATTACK_ORDER
    n_axes  = len(labels)
    angles  = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for m, color in zip(MODEL_NAMES, COLORS):
        vals  = [results[m].get(a, {}).get('attacked_accuracy', 0) for a in labels]
        vals += vals[:1]
        ax.plot(angles, vals, '-o', color=color, linewidth=2, markersize=6, label=m)
        ax.fill(angles, vals, color=color, alpha=0.10)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
    ax.set_title('Robustness Profile -- Post-Attack Accuracy (%)\n'
                 '(Larger area = more robust)',
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_asr_heatmap(results, save_path):
    """Heatmap: Attack Success Rate (%) for every (attack, model) pair."""
    data = np.array([
        [results[m].get(a, {}).get('attack_success_rate', 0) for m in MODEL_NAMES]
        for a in ATTACK_ORDER
    ])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap='RdYlGn_r', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(MODEL_NAMES)));  ax.set_xticklabels(MODEL_NAMES, fontsize=12)
    ax.set_yticks(range(len(ATTACK_ORDER))); ax.set_yticklabels(ATTACK_ORDER, fontsize=11)
    ax.set_title('Attack Success Rate Heatmap (%)\n(red = high vulnerability)', fontsize=12, fontweight='bold')
    for i in range(len(ATTACK_ORDER)):
        for j in range(len(MODEL_NAMES)):
            val = data[i, j]
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if val > 65 else 'black')
    plt.colorbar(im, ax=ax, label='ASR (%)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_conf_drop_heatmap(results, save_path):
    """Heatmap: Confidence drop (clean - attacked) per (attack, model) pair."""
    data = np.array([
        [results[m].get(a, {}).get('avg_conf_clean', 0) -
         results[m].get(a, {}).get('avg_conf_attacked', 0)
         for m in MODEL_NAMES]
        for a in ATTACK_ORDER
    ])
    vmax = max(abs(data.max()), abs(data.min()), 0.01)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap='RdYlBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(MODEL_NAMES)));  ax.set_xticklabels(MODEL_NAMES, fontsize=12)
    ax.set_yticks(range(len(ATTACK_ORDER))); ax.set_yticklabels(ATTACK_ORDER, fontsize=11)
    ax.set_title('Confidence Drop Heatmap (Clean - Attacked)\n'
                 '(red = large confidence drop; blue = confidence unexpectedly rises)',
                 fontsize=11, fontweight='bold')
    for i in range(len(ATTACK_ORDER)):
        for j in range(len(MODEL_NAMES)):
            val = data[i, j]
            ax.text(j, i, f'{val:+.3f}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if abs(val) > vmax * 0.6 else 'black')
    plt.colorbar(im, ax=ax, label='Confidence Drop')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_combined_epsilon_curves(fgsm_curves, pgd_curves, save_path):
    """FGSM vs PGD accuracy curves across epsilon for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, curves, method in zip(axes, [fgsm_curves, pgd_curves], ['FGSM', 'PGD (40 steps)']):
        for m, color in zip(MODEL_NAMES, COLORS):
            if m not in curves:
                continue
            accs = [curves[m].get(k, {}).get('attacked_accuracy', np.nan) for k in EPS_KEYS]
            ax.plot(EPS_LABELS, accs, '-o', color=color, linewidth=2, markersize=7, label=m)
        ax.set_xlabel('Perturbation Budget eps (L-inf)', fontsize=12)
        ax.set_ylabel('Recognition Accuracy (%)', fontsize=12)
        ax.set_title(f'{method}: Accuracy vs. eps', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylim(0, 105)
    plt.suptitle('Digital Attack Comparison: FGSM vs. PGD (40 steps)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_model_robustness_ranking(results, save_path):
    """
    Composite robustness score per model: mean post-attack accuracy across all attacks.
    Also shows the worst-case (minimum) accuracy per model.
    """
    avg_accs = []
    min_accs = []
    for m in MODEL_NAMES:
        accs = [results[m].get(a, {}).get('attacked_accuracy', 100.0) for a in ATTACK_ORDER]
        avg_accs.append(np.mean(accs))
        min_accs.append(np.min(accs))

    x = np.arange(len(MODEL_NAMES))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, avg_accs, w, label='Mean Attacked Accuracy', color='#1f77b4', alpha=0.85)
    bars2 = ax.bar(x + w/2, min_accs, w, label='Worst-Case Accuracy',    color='#d62728', alpha=0.85)

    for bar, v in list(zip(bars1, avg_accs)) + list(zip(bars2, min_accs)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Robustness Ranking\n'
                 '(Higher = more robust across all attacks)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_ylim(0, 115)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')

    # Print ranking
    ranked = sorted(zip(MODEL_NAMES, avg_accs, min_accs),
                    key=lambda x: x[1], reverse=True)
    print('\n  Model robustness ranking (by mean attacked accuracy):')
    for rank, (m, avg, worst) in enumerate(ranked, 1):
        print(f'    {rank}. {m:<15} mean={avg:.2f}%  worst-case={worst:.2f}%')


# =============================================================================
# Main
# =============================================================================

def main():
    print('Loading results...')
    results = load_results()
    fgsm_curves, pgd_curves = load_epsilon_curves()

    available_models  = [m for m in MODEL_NAMES if results[m]]
    available_attacks = [a for a in ATTACK_ORDER
                         if any(a in results[m] for m in MODEL_NAMES)]

    if not available_models:
        print('[ERROR] No result files found. Run scripts 01-04 first.')
        return

    print(f'  Models   : {available_models}')
    print(f'  Attacks  : {available_attacks}')

    # -- Tables ----------------------------------------------------------------
    print_and_save_summary_table(
        results, os.path.join(RESULTS_DIR, '06_summary_table.txt'))
    save_latex_table(
        results, os.path.join(RESULTS_DIR, '06_summary_table_latex.txt'))

    # -- Figures ---------------------------------------------------------------
    plot_accuracy_drop_bar(
        results, os.path.join(RESULTS_DIR, '06_accuracy_drop_bar.png'))
    plot_radar(
        results, os.path.join(RESULTS_DIR, '06_robustness_radar.png'))
    plot_asr_heatmap(
        results, os.path.join(RESULTS_DIR, '06_asr_heatmap.png'))
    plot_conf_drop_heatmap(
        results, os.path.join(RESULTS_DIR, '06_conf_drop_heatmap.png'))
    plot_model_robustness_ranking(
        results, os.path.join(RESULTS_DIR, '06_model_ranking.png'))

    if fgsm_curves or pgd_curves:
        plot_combined_epsilon_curves(
            fgsm_curves, pgd_curves,
            os.path.join(RESULTS_DIR, '06_combined_epsilon_curves.png'))

    print(f'\nAll outputs saved to: {RESULTS_DIR}')
    print('Comparative analysis complete.')


if __name__ == '__main__':
    main()
