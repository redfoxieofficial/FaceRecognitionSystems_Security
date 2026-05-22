"""
PGD Attack Evaluation -- Multi-Step Digital Perturbation.

Applies Projected Gradient Descent (Madry et al., 2018) to the three
lightweight recognition models.  PGD is strictly stronger than FGSM because
it takes multiple gradient-sign steps and projects back onto the eps-ball.

Outputs -> results/03_pgd/
    03_pgd_results.txt
    03_pgd_results.json
    03_pgd_accuracy_vs_steps.png         -- accuracy/ASR vs step count
    03_pgd_accuracy_vs_epsilon.png       -- accuracy vs budget at 40 steps
    03_pgd_fgsm_vs_pgd.png              -- direct FGSM vs PGD comparison
    03_pgd_steps_eps_heatmap_{model}.png -- 2D ASR grid (steps x epsilon)
    03_pgd_examples_{model}.png          -- clean vs PGD images at configs
    03_pgd_convergence_{model}.png       -- how attack loss evolves over steps
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
from pgd  import pgd_evaluate, pgd_attack
from fgsm import fgsm_evaluate

BASE_DIR    = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, 'results', '03_pgd')
os.makedirs(RESULTS_DIR, exist_ok=True)

EPSILON_MAIN   = 8 / 255
ALPHA          = 2 / 255
STEPS_LIST     = [1, 5, 10, 20, 40]
EPSILONS_SWEEP = [1/255, 2/255, 4/255, 8/255, 16/255]
EPS_LABELS     = [f'{e*255:.0f}/255' for e in EPSILONS_SWEEP]
STEPS_SWEEP    = 40
MODEL_COLORS   = ['#1f77b4', '#ff7f0e', '#2ca02c']


# =============================================================================
# Sweeps
# =============================================================================

def run_steps_sweep(model, dataloader, device, model_name):
    print(f'\n--- {model_name}: steps sweep (eps={EPSILON_MAIN*255:.0f}/255) ---')
    sweep = {}
    for steps in STEPS_LIST:
        true_l, clean_p, adv_p, clean_c, adv_c = pgd_evaluate(
            model, dataloader,
            epsilon=EPSILON_MAIN, alpha=ALPHA, num_steps=steps, device=device
        )
        m = compute_attack_metrics(true_l, clean_p, adv_p, clean_c, adv_c)
        sweep[steps] = m
        print(f'  steps={steps:>3}  clean={m["clean_accuracy"]:.1f}%  '
              f'attacked={m["attacked_accuracy"]:.1f}%  ASR={m["attack_success_rate"]:.1f}%')
    return sweep


def run_epsilon_sweep(model, dataloader, device, model_name):
    print(f'\n--- {model_name}: epsilon sweep (steps={STEPS_SWEEP}) ---')
    sweep = {}
    for eps in EPSILONS_SWEEP:
        true_l, clean_p, adv_p, clean_c, adv_c = pgd_evaluate(
            model, dataloader,
            epsilon=eps, alpha=eps / 4, num_steps=STEPS_SWEEP, device=device
        )
        m = compute_attack_metrics(true_l, clean_p, adv_p, clean_c, adv_c)
        sweep[eps] = m
        print(f'  eps={eps*255:.0f}/255  attacked={m["attacked_accuracy"]:.1f}%  '
              f'ASR={m["attack_success_rate"]:.1f}%')
    return sweep


def run_steps_eps_grid(model, dataloader, device, model_name,
                       steps_list=(1, 5, 10, 20, 40),
                       eps_list=(2/255, 4/255, 8/255, 16/255)):
    """Compute ASR for every (steps, epsilon) combination for a 2-D heatmap."""
    print(f'\n--- {model_name}: full (steps x eps) grid ---')
    grid = np.zeros((len(steps_list), len(eps_list)))
    for i, steps in enumerate(steps_list):
        for j, eps in enumerate(eps_list):
            true_l, clean_p, adv_p, clean_c, adv_c = pgd_evaluate(
                model, dataloader,
                epsilon=eps, alpha=eps/4, num_steps=steps, device=device
            )
            m = compute_attack_metrics(true_l, clean_p, adv_p, clean_c, adv_c)
            grid[i, j] = m['attack_success_rate']
            print(f'  steps={steps} eps={eps*255:.0f}/255  ASR={grid[i,j]:.1f}%')
    return grid, steps_list, eps_list


# =============================================================================
# Plots
# =============================================================================

def plot_accuracy_vs_steps(steps_sweeps, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, ylabel in zip(
        axes,
        ['attacked_accuracy', 'attack_success_rate'],
        ['Recognition Accuracy (%)', 'Attack Success Rate (%)']
    ):
        for (name, sweep), marker, color in zip(steps_sweeps.items(),
                                                ['o', 's', '^'], MODEL_COLORS):
            vals = [sweep[s][metric] for s in STEPS_LIST]
            ax.plot(STEPS_LIST, vals, marker=marker, color=color,
                    linewidth=2, markersize=7, label=name)
        ax.set_xlabel('Number of PGD Steps', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'PGD: {ylabel} vs. Steps\n(eps={EPSILON_MAIN*255:.0f}/255)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xticks(STEPS_LIST)
    plt.suptitle('PGD Attack -- Effect of Iteration Count', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_accuracy_vs_epsilon(eps_sweeps, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, sweep), marker, color in zip(eps_sweeps.items(),
                                            ['o', 's', '^'], MODEL_COLORS):
        accs = [sweep[e]['attacked_accuracy'] for e in EPSILONS_SWEEP]
        ax.plot(EPS_LABELS, accs, marker=marker, color=color,
                linewidth=2, markersize=7, label=name)
    ax.set_xlabel('Perturbation Budget eps (L-inf)', fontsize=12)
    ax.set_ylabel('Attacked Accuracy (%)', fontsize=12)
    ax.set_title(f'PGD: Accuracy vs. Budget (steps={STEPS_SWEEP})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_fgsm_vs_pgd(models, dataloader, device, save_path):
    """
    Direct comparison: FGSM vs PGD (40 steps) at the same epsilon values.
    Proves that multi-step optimisation finds strictly stronger examples.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    eps_list   = [2/255, 4/255, 8/255, 16/255]
    eps_labels = [f'{e*255:.0f}/255' for e in eps_list]

    for ax, metric, ylabel in zip(
        axes,
        ['attacked_accuracy', 'attack_success_rate'],
        ['Accuracy Under Attack (%)', 'Attack Success Rate (%)']
    ):
        for (name, model), color in zip(models.items(), MODEL_COLORS):
            fgsm_vals, pgd_vals = [], []
            for eps in eps_list:
                # FGSM
                true_l, clean_p, adv_p, clean_c, adv_c = fgsm_evaluate(
                    model, dataloader, eps, device)
                fgsm_vals.append(compute_attack_metrics(
                    true_l, clean_p, adv_p, clean_c, adv_c)[metric])
                # PGD (40 steps)
                true_l, clean_p, adv_p, clean_c, adv_c = pgd_evaluate(
                    model, dataloader, eps, eps/4, 40, device)
                pgd_vals.append(compute_attack_metrics(
                    true_l, clean_p, adv_p, clean_c, adv_c)[metric])

            ax.plot(eps_labels, fgsm_vals, '--o', color=color,
                    linewidth=1.8, markersize=6, alpha=0.7, label=f'{name} FGSM')
            ax.plot(eps_labels, pgd_vals,  '-s',  color=color,
                    linewidth=2.2, markersize=7, label=f'{name} PGD-40')

        ax.set_xlabel('Perturbation Budget eps (L-inf)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'FGSM vs PGD-40: {ylabel}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, linestyle='--', alpha=0.4)
        if metric == 'attacked_accuracy':
            ax.set_ylim(0, 105)

    plt.suptitle('FGSM vs. PGD (40 Steps): Multi-Step Attack is Strictly Stronger',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_steps_eps_heatmap(grid, steps_list, eps_list, model_name, save_path):
    """2-D heatmap of ASR at every (steps, epsilon) combination for one model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, cmap='RdYlGn_r', vmin=0, vmax=100, aspect='auto')

    ax.set_xticks(range(len(eps_list)))
    ax.set_xticklabels([f'{e*255:.0f}/255' for e in eps_list], fontsize=10)
    ax.set_yticks(range(len(steps_list)))
    ax.set_yticklabels([str(s) for s in steps_list], fontsize=10)
    ax.set_xlabel('Perturbation Budget eps (L-inf)', fontsize=11)
    ax.set_ylabel('Number of PGD Steps', fontsize=11)
    ax.set_title(f'PGD Attack Success Rate (%) -- {model_name}\n'
                 f'(red = high ASR, green = low ASR)', fontsize=11, fontweight='bold')

    for i in range(len(steps_list)):
        for j in range(len(eps_list)):
            val = grid[i, j]
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if val > 60 else 'black')

    plt.colorbar(im, ax=ax, label='Attack Success Rate (%)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_pgd_examples(model, dataloader, device, model_name, save_path, n_samples=3):
    """Clean vs PGD examples at several configurations."""
    images, class_ids, _, _ = next(iter(dataloader))
    images    = images[:n_samples]
    class_ids = class_ids[:n_samples]
    configs   = [(4/255, 10, '4/255 10-step'),
                 (8/255, 20, '8/255 20-step'),
                 (8/255, 40, '8/255 40-step'),
                 (16/255, 40, '16/255 40-step')]

    fig, axes = plt.subplots(n_samples, 1 + len(configs),
                             figsize=(3.5 * (1 + len(configs)), 3.5 * n_samples))
    if n_samples == 1:
        axes = [axes]

    col_titles = ['Clean'] + [f'PGD\n{c[2]}' for c in configs]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=9, fontweight='bold')

    for row, (img, label) in enumerate(zip(images, class_ids)):
        img_np = (denormalize(img).permute(1, 2, 0).numpy() * 255).astype('uint8')
        axes[row][0].imshow(img_np)
        axes[row][0].axis('off')
        for col, (eps, steps, _) in enumerate(configs, start=1):
            adv = pgd_attack(model, img.unsqueeze(0), label.unsqueeze(0),
                             epsilon=eps, alpha=eps/4, num_steps=steps, device=device)
            adv_np = (denormalize(adv.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')
            axes[row][col].imshow(adv_np)
            axes[row][col].axis('off')

    plt.suptitle(f'PGD Adversarial Examples -- {model_name}',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_attack_convergence(model, dataloader, device, model_name, save_path,
                            eps=8/255, alpha=2/255):
    """
    Track loss inside PGD over individual steps for a single batch.
    Shows how the attack converges and whether more steps always help.
    """
    import torch.nn.functional as F

    model.eval()
    images, class_ids, _, _ = next(iter(dataloader))
    images    = images[:8].to(device)   # single mini-batch
    class_ids = class_ids[:8].to(device)

    x = images + torch.zeros_like(images).uniform_(-eps, eps)
    x = torch.clamp(x, images - eps, images + eps).detach()

    losses = []
    accs   = []
    MAX_STEPS = 60

    with torch.enable_grad():
        for _ in range(MAX_STEPS):
            x = x.detach().requires_grad_(True)
            logits = model(x)
            loss   = F.cross_entropy(logits, class_ids)
            model.zero_grad()
            loss.backward()
            x = x + alpha * x.grad.sign()
            x = torch.max(torch.min(x, images + eps), images - eps)
            x = torch.clamp(x, -3.0, 3.0).detach()

            with torch.no_grad():
                logits2 = model(x)
                ce   = F.cross_entropy(logits2, class_ids).item()
                acc  = (logits2.argmax(dim=1) == class_ids).float().mean().item() * 100
            losses.append(ce)
            accs.append(acc)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    steps = list(range(1, MAX_STEPS + 1))

    axes[0].plot(steps, losses, color='#d62728', linewidth=2)
    axes[0].set_xlabel('PGD Step', fontsize=12)
    axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12)
    axes[0].set_title('Attack Loss Convergence\n(higher = more effective attack)',
                      fontsize=11, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.4)

    axes[1].plot(steps, accs, color='#1f77b4', linewidth=2)
    axes[1].set_xlabel('PGD Step', fontsize=12)
    axes[1].set_ylabel('Accuracy on Mini-Batch (%)', fontsize=12)
    axes[1].set_title('Accuracy Drop per Step\n(lower = more effective attack)',
                      fontsize=11, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].set_ylim(0, 105)

    plt.suptitle(f'PGD Convergence Behaviour -- {model_name}\n'
                 f'(eps={eps*255:.0f}/255, alpha={alpha*255:.0f}/255)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


# =============================================================================
# Text / JSON exports
# =============================================================================

def save_results_txt(steps_sweeps, eps_sweeps, path):
    lines = ['=' * 70, 'PGD ATTACK EVALUATION', '=' * 70, '']
    lines.append(f'A) Steps sweep  (eps={EPSILON_MAIN*255:.0f}/255, alpha={ALPHA*255:.0f}/255)')
    lines.append('-' * 70)
    for name, sweep in steps_sweeps.items():
        lines.append(f'  {name}')
        for steps in STEPS_LIST:
            m = sweep[steps]
            lines.append(f'    steps={steps:>3}  '
                         f'atk_acc={m["attacked_accuracy"]:.2f}%  '
                         f'ASR={m["attack_success_rate"]:.2f}%  '
                         f'conf_drop={m["avg_conf_attacked"]-m["avg_conf_clean"]:+.4f}')
        lines.append('')
    lines.append(f'B) Epsilon sweep  (steps={STEPS_SWEEP}, alpha=eps/4)')
    lines.append('-' * 70)
    for name, sweep in eps_sweeps.items():
        lines.append(f'  {name}')
        for eps in EPSILONS_SWEEP:
            m = sweep[eps]
            lines.append(f'    eps={eps*255:.0f}/255  '
                         f'atk_acc={m["attacked_accuracy"]:.2f}%  '
                         f'ASR={m["attack_success_rate"]:.2f}%')
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

    steps_sweeps = {}
    eps_sweeps   = {}

    for name, model in models.items():
        steps_sweeps[name] = run_steps_sweep(model, dataloader, device, name)
        eps_sweeps[name]   = run_epsilon_sweep(model, dataloader, device, name)

    # -- Aggregate plots -------------------------------------------------------
    print('\nGenerating aggregate plots...')
    plot_accuracy_vs_steps(
        steps_sweeps, os.path.join(RESULTS_DIR, '03_pgd_accuracy_vs_steps.png'))
    plot_accuracy_vs_epsilon(
        eps_sweeps, os.path.join(RESULTS_DIR, '03_pgd_accuracy_vs_epsilon.png'))
    plot_fgsm_vs_pgd(
        models, dataloader, device,
        os.path.join(RESULTS_DIR, '03_pgd_fgsm_vs_pgd.png'))

    # -- Per-model plots -------------------------------------------------------
    print('\nGenerating per-model plots...')
    dataloader_vis  = get_dataloader(split='test', include='genuine', batch_size=4)
    dataloader_conv = get_dataloader(split='test', include='genuine', batch_size=8)

    GRID_STEPS = [1, 5, 10, 20, 40]
    GRID_EPS   = [2/255, 4/255, 8/255, 16/255]

    for name, model in models.items():
        lname = name.lower()
        plot_pgd_examples(
            model, dataloader_vis, device, name,
            os.path.join(RESULTS_DIR, f'03_pgd_examples_{lname}.png'))
        plot_attack_convergence(
            model, dataloader_conv, device, name,
            os.path.join(RESULTS_DIR, f'03_pgd_convergence_{lname}.png'))
        grid, sl, el = run_steps_eps_grid(
            model, dataloader, device, name, GRID_STEPS, GRID_EPS)
        plot_steps_eps_heatmap(
            grid, sl, el, name,
            os.path.join(RESULTS_DIR, f'03_pgd_steps_eps_heatmap_{lname}.png'))

    # -- Text / JSON exports --------------------------------------------------
    save_results_txt(
        steps_sweeps, eps_sweeps,
        os.path.join(RESULTS_DIR, '03_pgd_results.txt'))

    json_data = {
        'steps_sweep':   {name: {str(s): m for s, m in sw.items()}
                          for name, sw in steps_sweeps.items()},
        'epsilon_sweep': {name: {f'eps_{int(e*255)}_255': m for e, m in sw.items()}
                          for name, sw in eps_sweeps.items()},
    }
    with open(os.path.join(RESULTS_DIR, '03_pgd_results.json'), 'w', encoding='utf-8') as fh:
        json.dump(json_data, fh, indent=2)

    print(f'\nAll outputs saved to: {RESULTS_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
