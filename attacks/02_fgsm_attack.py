"""
FGSM Attack Evaluation -- Digital Perturbation Baseline.

Applies the Fast Gradient Sign Method (Goodfellow et al., 2014) at five
epsilon values to each of the three lightweight recognition models.

Outputs -> results/02_fgsm/
    02_fgsm_results.txt
    02_fgsm_results.json
    02_fgsm_accuracy_vs_epsilon.png      -- accuracy curve per model
    02_fgsm_asr_vs_epsilon.png           -- ASR curve per model
    02_fgsm_confidence_vs_epsilon.png    -- confidence collapse per model
    02_fgsm_perturbation_viz_{model}.png -- noise visualisation (amplified)
    02_fgsm_examples_{model}.png         -- clean vs adversarial image grid
    02_fgsm_conf_dist_{model}.png        -- confidence distribution at each eps
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(__file__))
from utils import (load_all_models, get_dataloader,
                   compute_attack_metrics, denormalize)
from fgsm import fgsm_evaluate, fgsm_attack

BASE_DIR    = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, 'results', '02_fgsm')
os.makedirs(RESULTS_DIR, exist_ok=True)

EPSILONS     = [1/255, 2/255, 4/255, 8/255, 16/255]
EPS_LABELS   = [f'{e*255:.0f}/255' for e in EPSILONS]
MODEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']


# =============================================================================
# Per-model sweep
# =============================================================================

def run_fgsm_sweep(model, dataloader, device, model_name):
    """Run FGSM at every epsilon value; return dict {eps: metrics}."""
    print(f'\n--- {model_name} ---')
    sweep = {}
    for eps in EPSILONS:
        true_l, clean_p, adv_p, clean_c, adv_c = fgsm_evaluate(
            model, dataloader, eps, device
        )
        metrics = compute_attack_metrics(true_l, clean_p, adv_p, clean_c, adv_c)
        # also store raw confidences for distribution plots
        metrics['_clean_conf'] = clean_c.tolist()
        metrics['_adv_conf']   = adv_c.tolist()
        sweep[eps] = metrics
        print(f'  eps={eps*255:.0f}/255  '
              f'clean={metrics["clean_accuracy"]:.1f}%  '
              f'attacked={metrics["attacked_accuracy"]:.1f}%  '
              f'drop={metrics["accuracy_drop"]:.1f}pp  '
              f'ASR={metrics["attack_success_rate"]:.1f}%')
    return sweep


# =============================================================================
# Plots -- aggregate (all models on one figure)
# =============================================================================

def plot_accuracy_vs_epsilon(all_sweeps, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, sweep), marker, color in zip(all_sweeps.items(), ['o', 's', '^'], MODEL_COLORS):
        clean_accs    = [sweep[e]['clean_accuracy']    for e in EPSILONS]
        attacked_accs = [sweep[e]['attacked_accuracy'] for e in EPSILONS]
        ax.plot(EPS_LABELS, clean_accs, linestyle='--', color=color, alpha=0.3, linewidth=1.2)
        ax.plot(EPS_LABELS, attacked_accs, marker=marker, color=color,
                linewidth=2, markersize=7, label=name)
    ax.set_xlabel('Perturbation Budget eps (L-inf)', fontsize=12)
    ax.set_ylabel('Recognition Accuracy (%)', fontsize=12)
    ax.set_title('FGSM: Accuracy vs. Perturbation Budget\n'
                 '(dashed = clean baseline, solid = under attack)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_asr_vs_epsilon(all_sweeps, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, sweep), marker, color in zip(all_sweeps.items(), ['o', 's', '^'], MODEL_COLORS):
        asrs = [sweep[e]['attack_success_rate'] for e in EPSILONS]
        ax.plot(EPS_LABELS, asrs, marker=marker, color=color,
                linewidth=2, markersize=7, label=name)
    ax.set_xlabel('Perturbation Budget eps (L-inf)', fontsize=12)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('FGSM: Attack Success Rate vs. Perturbation Budget', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_confidence_vs_epsilon(all_sweeps, save_path):
    """How average model confidence drops as epsilon increases."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for (name, sweep), marker, color in zip(all_sweeps.items(), ['o', 's', '^'], MODEL_COLORS):
        clean_confs = [sweep[e]['avg_conf_clean']    for e in EPSILONS]
        adv_confs   = [sweep[e]['avg_conf_attacked'] for e in EPSILONS]

        axes[0].plot(EPS_LABELS, clean_confs, linestyle='--', color=color, alpha=0.3, linewidth=1.2)
        axes[0].plot(EPS_LABELS, adv_confs,   marker=marker, color=color,
                     linewidth=2, markersize=7, label=name)

        conf_drops = [c - a for c, a in zip(clean_confs, adv_confs)]
        axes[1].plot(EPS_LABELS, conf_drops, marker=marker, color=color,
                     linewidth=2, markersize=7, label=name)

    axes[0].set_title('Average Confidence vs. Epsilon\n(dashed = clean, solid = attacked)',
                      fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Average Model Confidence', fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.4)

    axes[1].set_title('Confidence Drop (Clean - Attacked) vs. Epsilon', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Confidence Drop', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.4)

    for ax in axes:
        ax.set_xlabel('Perturbation Budget eps (L-inf)', fontsize=11)

    plt.suptitle('FGSM: How Model Confidence Collapses with Perturbation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


# =============================================================================
# Plots -- per-model
# =============================================================================

def plot_perturbation_noise(model, dataloader, device, model_name, save_path, eps=8/255):
    """
    Visualise the adversarial perturbation itself.
    Shows: Clean | Adversarial | Noise x10 | Noise x50
    This reveals the invisible gradient-sign pattern the attack uses.
    """
    images, class_ids, _, _ = next(iter(dataloader))
    img    = images[:1].to(device)
    label  = class_ids[:1].to(device)

    adv, _ = fgsm_attack(model, img, label, eps, device)

    noise  = (adv - img).detach().cpu()
    img_np  = (denormalize(img.squeeze(0).detach().cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')
    adv_np  = (denormalize(adv.squeeze(0).detach().cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')

    # Amplified noise: shift to [0,1] range for display
    noise_10  = (noise.squeeze(0).permute(1, 2, 0).numpy() * 10 + 0.5).clip(0, 1)
    noise_50  = (noise.squeeze(0).permute(1, 2, 0).numpy() * 50 + 0.5).clip(0, 1)
    noise_np  = (noise_10 * 255).astype('uint8')
    noise50_np = (noise_50 * 255).astype('uint8')

    l_inf = noise.abs().max().item()

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, img_show, title in zip(
        axes,
        [img_np, adv_np, noise_np, noise50_np],
        ['Clean Image', f'Adversarial (eps={eps*255:.0f}/255)',
         'Noise x10\n(near-invisible)', 'Noise x50\n(pattern visible)']
    ):
        ax.imshow(img_show)
        ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold')

    axes[1].set_xlabel(f'L-inf norm = {l_inf:.4f}  ({l_inf*255:.2f}/255)', fontsize=9)

    plt.suptitle(f'FGSM Perturbation Visualisation -- {model_name}\n'
                 f'The noise is invisible to humans but fools the neural network.',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_adversarial_examples(model, dataloader, device, model_name, save_path, n_samples=4):
    """Clean vs FGSM examples at multiple epsilon values."""
    epsilons_vis = [1/255, 4/255, 8/255, 16/255]
    images, class_ids, _, _ = next(iter(dataloader))
    images    = images[:n_samples]
    class_ids = class_ids[:n_samples]

    fig, axes = plt.subplots(n_samples, 1 + len(epsilons_vis),
                             figsize=(3.5 * (1 + len(epsilons_vis)), 3.5 * n_samples))
    if n_samples == 1:
        axes = [axes]

    col_titles = ['Clean'] + [f'FGSM\neps={e*255:.0f}/255' for e in epsilons_vis]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10, fontweight='bold')

    for row, (img, label) in enumerate(zip(images, class_ids)):
        img_np = (denormalize(img).permute(1, 2, 0).numpy() * 255).astype('uint8')
        axes[row][0].imshow(img_np)
        axes[row][0].axis('off')
        for col, eps in enumerate(epsilons_vis, start=1):
            adv, _ = fgsm_attack(model, img.unsqueeze(0), label.unsqueeze(0), eps, device)
            adv_np = (denormalize(adv.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')
            axes[row][col].imshow(adv_np)
            axes[row][col].axis('off')

    plt.suptitle(f'FGSM Adversarial Examples -- {model_name}',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_confidence_distributions_per_model(sweep, model_name, save_path):
    """
    For one model: violin plots of clean and adversarial confidence at each epsilon.
    Shows how the confidence distribution shifts as epsilon increases.
    """
    n_eps = len(EPSILONS)
    fig, axes = plt.subplots(1, n_eps, figsize=(4 * n_eps, 5), sharey=True)

    for ax, eps, label in zip(axes, EPSILONS, EPS_LABELS):
        clean_c = np.array(sweep[eps]['_clean_conf'])
        adv_c   = np.array(sweep[eps]['_adv_conf'])
        vp = ax.violinplot([clean_c, adv_c], positions=[1, 2],
                           showmedians=True, showextrema=True)
        for body in vp['bodies']:
            body.set_alpha(0.65)
        vp['cmedians'].set_color('red')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Clean', 'FGSM'], fontsize=9)
        ax.set_title(f'eps = {label}', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.text(1, clean_c.mean() + 0.03, f'{clean_c.mean():.2f}', ha='center', fontsize=8)
        ax.text(2, adv_c.mean() + 0.03,   f'{adv_c.mean():.2f}',   ha='center', fontsize=8)

    axes[0].set_ylabel('Confidence', fontsize=11)
    plt.suptitle(f'Confidence Distribution Under FGSM -- {model_name}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


# =============================================================================
# Text / JSON exports
# =============================================================================

def save_results_txt(all_sweeps, path):
    lines = ['=' * 70, 'FGSM ATTACK EVALUATION', '=' * 70, '']
    for name, sweep in all_sweeps.items():
        lines.append(f'Model: {name}')
        lines.append('-' * 40)
        lines.append(f"  {'eps':>8} {'Clean Acc':>10} {'Atk Acc':>10} "
                     f"{'Drop':>8} {'ASR':>8} {'Delta Conf':>10}")
        for eps, label in zip(EPSILONS, EPS_LABELS):
            m       = sweep[eps]
            delta_c = m['avg_conf_attacked'] - m['avg_conf_clean']
            lines.append(
                f"  {label:>8} {m['clean_accuracy']:>9.2f}% "
                f"{m['attacked_accuracy']:>9.2f}% "
                f"{m['accuracy_drop']:>7.2f}pp "
                f"{m['attack_success_rate']:>7.2f}% "
                f"{delta_c:>+9.4f}"
            )
        lines.append('')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f'  Saved -> {path}')


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\nLoading NUAA genuine test split...')
    dataloader = get_dataloader(split='test', include='genuine', batch_size=16)
    print(f'  Samples: {len(dataloader.dataset)}')

    print('\nLoading models...')
    models = load_all_models(device)

    all_sweeps = {}
    for name, model in models.items():
        all_sweeps[name] = run_fgsm_sweep(model, dataloader, device, name)

    # -- Aggregate plots -------------------------------------------------------
    print('\nGenerating aggregate plots...')
    plot_accuracy_vs_epsilon(
        all_sweeps, os.path.join(RESULTS_DIR, '02_fgsm_accuracy_vs_epsilon.png'))
    plot_asr_vs_epsilon(
        all_sweeps, os.path.join(RESULTS_DIR, '02_fgsm_asr_vs_epsilon.png'))
    plot_confidence_vs_epsilon(
        all_sweeps, os.path.join(RESULTS_DIR, '02_fgsm_confidence_vs_epsilon.png'))

    # -- Per-model plots -------------------------------------------------------
    print('\nGenerating per-model plots...')
    dataloader_vis = get_dataloader(split='test', include='genuine', batch_size=4)
    for name, model in models.items():
        lname = name.lower()
        plot_perturbation_noise(
            model, dataloader_vis, device, name,
            os.path.join(RESULTS_DIR, f'02_fgsm_perturbation_viz_{lname}.png'))
        plot_adversarial_examples(
            model, dataloader_vis, device, name,
            os.path.join(RESULTS_DIR, f'02_fgsm_examples_{lname}.png'))
        plot_confidence_distributions_per_model(
            all_sweeps[name], name,
            os.path.join(RESULTS_DIR, f'02_fgsm_conf_dist_{lname}.png'))

    # -- Text / JSON exports --------------------------------------------------
    save_results_txt(all_sweeps, os.path.join(RESULTS_DIR, '02_fgsm_results.txt'))

    json_data = {}
    for name, sweep in all_sweeps.items():
        json_data[name] = {}
        for eps, label in zip(EPSILONS, EPS_LABELS):
            key = f'eps_{int(eps*255)}_255'
            m   = {k: v for k, v in sweep[eps].items() if not k.startswith('_')}
            json_data[name][key] = m

    with open(os.path.join(RESULTS_DIR, '02_fgsm_results.json'), 'w', encoding='utf-8') as fh:
        json.dump(json_data, fh, indent=2)

    print(f'\nAll outputs saved to: {RESULTS_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
