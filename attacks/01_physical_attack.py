"""
Physical Attack Evaluation -- NUAA Printed-Photograph Baseline.

The NUAA ImposterFace subset consists of photographs of the 15 subjects that were
printed on paper and then re-photographed.  This is the canonical physical
presentation attack (Anjos & Marcel, 2011).

Attack success: if the model classifies a printed photo as the correct subject
identity -> the attacker can spoof the system without any digital manipulation.

Outputs -> results/01_physical/
    01_physical_attack_results.txt       -- human-readable summary
    01_physical_attack_results.json      -- machine-readable for script 06
    01_physical_confidence_dist.png      -- violin: genuine vs impostor confidence
    01_physical_confidence_hist.png      -- overlaid histograms per model
    01_physical_per_subject.png          -- per-subject attack success bar chart
    01_physical_per_subject_conf.png     -- per-subject confidence boxplots
    01_physical_sample_images.png        -- genuine vs impostor visual samples
    01_physical_per_subject_stats.csv    -- per-subject CSV for thesis table
"""

import os
import sys
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(__file__))
from utils import (load_all_models, get_dataloader, CLASS_NAMES,
                   compute_attack_metrics, denormalize)

BASE_DIR    = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, 'results', '01_physical')
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_COLORS = {'MobileNetV3': '#1f77b4', 'GhostNet': '#ff7f0e', 'EfficientNet': '#2ca02c'}


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_physical(model, genuine_loader, impostor_loader, device):
    model.eval()

    def collect(loader):
        true_l, pred_l, conf_l = [], [], []
        for images, class_ids, _, _ in loader:
            logits = model(images.to(device))
            probs  = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            true_l.extend(class_ids.numpy().tolist())
            pred_l.extend(pred.cpu().numpy().tolist())
            conf_l.extend(conf.cpu().numpy().tolist())
        return np.array(true_l), np.array(pred_l), np.array(conf_l)

    true_g, pred_g, conf_g = collect(genuine_loader)
    true_i, pred_i, conf_i = collect(impostor_loader)

    genuine_acc      = (pred_g == true_g).mean() * 100
    physical_success = (pred_i == true_i).mean() * 100

    return {
        'genuine_accuracy':        round(float(genuine_acc),              2),
        'physical_success_rate':   round(float(physical_success),         2),
        'physical_rejection_rate': round(float(100.0 - physical_success), 2),
        'avg_conf_genuine':        round(float(conf_g.mean()),            4),
        'avg_conf_impostor':       round(float(conf_i.mean()),            4),
        'conf_drop':               round(float(conf_g.mean() - conf_i.mean()), 4),
        # raw arrays kept for visualisation (stripped before JSON export)
        '_true_impostor': true_i.tolist(),
        '_pred_impostor': pred_i.tolist(),
        '_conf_impostor': conf_i.tolist(),
        '_true_genuine':  true_g.tolist(),
        '_pred_genuine':  pred_g.tolist(),
        '_conf_genuine':  conf_g.tolist(),
    }


# =============================================================================
# Plots
# =============================================================================

def plot_sample_images(genuine_loader, impostor_loader, save_path, n=5):
    """Side-by-side grid of genuine and impostor (printed photo) samples."""
    gen_imgs  = []
    imp_imgs  = []
    for images, _, _, _ in genuine_loader:
        gen_imgs.extend([images[i] for i in range(min(n, len(images)))])
        if len(gen_imgs) >= n:
            break
    for images, _, _, _ in impostor_loader:
        imp_imgs.extend([images[i] for i in range(min(n, len(images)))])
        if len(imp_imgs) >= n:
            break

    gen_imgs = gen_imgs[:n]
    imp_imgs = imp_imgs[:n]

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7))
    labels = ['Genuine (real face)', 'Impostor (printed photo)']
    for row, (imgs, label) in enumerate(zip([gen_imgs, imp_imgs], labels)):
        axes[row][0].set_ylabel(label, fontsize=12, fontweight='bold')
        for col, img in enumerate(imgs):
            img_np = (denormalize(img).permute(1, 2, 0).numpy() * 255).astype('uint8')
            axes[row][col].imshow(img_np)
            axes[row][col].axis('off')

    plt.suptitle('NUAA Dataset: Genuine Faces vs. Physical Print Attack Samples',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_confidence_distributions(all_results, save_path):
    """Violin plot of confidence scores: genuine vs. impostor per model."""
    model_names = list(all_results.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        r   = all_results[name]
        gen = np.array(r['_conf_genuine'])
        imp = np.array(r['_conf_impostor'])

        vp = ax.violinplot([gen, imp], positions=[1, 2],
                           showmedians=True, showextrema=True)
        for body in vp['bodies']:
            body.set_alpha(0.7)
        vp['cmedians'].set_color('red')
        vp['cmedians'].set_linewidth(2)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Genuine\n(real face)', 'Impostor\n(print)'], fontsize=11)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_ylabel('Model Confidence', fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.text(1, gen.mean() + 0.03, f'mean={gen.mean():.3f}', ha='center', fontsize=9, color='navy')
        ax.text(2, imp.mean() + 0.03, f'mean={imp.mean():.3f}', ha='center', fontsize=9, color='darkred')

    plt.suptitle('Model Confidence Distribution: Genuine vs. Physical Print Attack',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_confidence_histograms(all_results, save_path):
    """Overlaid histograms of genuine and impostor confidence per model."""
    model_names = list(all_results.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        gen = np.array(all_results[name]['_conf_genuine'])
        imp = np.array(all_results[name]['_conf_impostor'])

        bins = np.linspace(0, 1, 40)
        ax.hist(gen, bins=bins, alpha=0.65, color='#2ca02c', label='Genuine', density=True)
        ax.hist(imp, bins=bins, alpha=0.65, color='#d62728', label='Impostor', density=True)

        ax.axvline(gen.mean(), color='#2ca02c', linestyle='--', linewidth=2,
                   label=f'Gen mean={gen.mean():.3f}')
        ax.axvline(imp.mean(), color='#d62728', linestyle='--', linewidth=2,
                   label=f'Imp mean={imp.mean():.3f}')

        ax.set_xlabel('Confidence Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.suptitle('Confidence Histograms: Genuine vs. Impostor\n'
                 '(Separation = discriminability of the model under print attack)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_per_subject_success(all_results, save_path):
    """Bar chart: physical attack success rate per subject, per model."""
    model_names = list(all_results.keys())
    n_subjects  = len(CLASS_NAMES)
    x     = np.arange(n_subjects)
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 5))
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (name, color) in enumerate(zip(model_names, colors)):
        true_i = np.array(all_results[name]['_true_impostor'])
        pred_i = np.array(all_results[name]['_pred_impostor'])
        rates  = []
        for subj_idx in range(n_subjects):
            mask = true_i == subj_idx
            rates.append((pred_i[mask] == true_i[mask]).mean() * 100 if mask.sum() > 0 else 0.0)
        ax.bar(x + i * width, rates, width, label=name, color=color, alpha=0.85)

    ax.set_xlabel('Subject ID', fontsize=12)
    ax.set_ylabel('Physical Attack Success Rate (%)', fontsize=12)
    ax.set_title('Per-Subject Physical Attack Success Rate\n'
                 '(Higher = model is fooled by that subject\'s printed photo)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% chance')
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_ylim(0, 115)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_per_subject_confidence_box(all_results, save_path):
    """
    For each model: boxplots of impostor confidence per subject.
    Shows which subjects' printed photos the model is most/least confident about.
    """
    model_names = list(all_results.keys())
    n_models    = len(model_names)
    n_subjects  = len(CLASS_NAMES)
    colors      = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, name, color in zip(axes, model_names, colors):
        true_i = np.array(all_results[name]['_true_impostor'])
        conf_i = np.array(all_results[name]['_conf_impostor'])
        data   = [conf_i[true_i == s] for s in range(n_subjects)]
        data   = [d if len(d) > 0 else np.array([0.0]) for d in data]

        bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color='red', linewidth=2))
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(range(1, n_subjects + 1))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Subject ID', fontsize=10)
        ax.set_ylabel('Impostor Confidence', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.suptitle('Per-Subject Impostor Confidence Distribution\n'
                 '(High confidence = model more easily fooled by that subject\'s print)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


# =============================================================================
# Text / CSV exports
# =============================================================================

def save_results_txt(all_results, path):
    lines = ['=' * 70,
             'PHYSICAL ATTACK EVALUATION -- NUAA PRINTED PHOTOGRAPH',
             '=' * 70, '']
    for name, r in all_results.items():
        lines += [
            f'Model: {name}',
            '-' * 40,
            f"  Genuine accuracy (ClientFace test):    {r['genuine_accuracy']:>6.2f}%",
            f"  Physical attack success rate:          {r['physical_success_rate']:>6.2f}%",
            f"    (model correctly identifies printed photo -> attacker succeeds)",
            f"  Physical attack rejection rate:        {r['physical_rejection_rate']:>6.2f}%",
            f"    (model fails to match printed photo  -> model is robust)",
            f"  Avg confidence -- genuine:              {r['avg_conf_genuine']:>6.4f}",
            f"  Avg confidence -- impostor:             {r['avg_conf_impostor']:>6.4f}",
            f"  Confidence drop (genuine - impostor):  {r['conf_drop']:>+6.4f}",
            '',
        ]
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f'  Saved -> {path}')


def save_per_subject_csv(all_results, path):
    """CSV table: subject x model with genuine acc and attack success rate."""
    model_names = list(all_results.keys())
    rows = []
    for subj_idx, subj_id in enumerate(CLASS_NAMES):
        row = {'subject': subj_id}
        for name in model_names:
            true_g = np.array(all_results[name]['_true_genuine'])
            pred_g = np.array(all_results[name]['_pred_genuine'])
            true_i = np.array(all_results[name]['_true_impostor'])
            pred_i = np.array(all_results[name]['_pred_impostor'])

            mask_g = true_g == subj_idx
            mask_i = true_i == subj_idx
            gen_acc   = (pred_g[mask_g] == subj_idx).mean() * 100 if mask_g.sum() > 0 else float('nan')
            atk_succ  = (pred_i[mask_i] == subj_idx).mean() * 100 if mask_i.sum() > 0 else float('nan')
            row[f'{name}_genuine_acc']  = round(gen_acc,  2)
            row[f'{name}_attack_succ']  = round(atk_succ, 2)
        rows.append(row)

    fieldnames = ['subject'] + [f'{n}_{s}' for n in model_names for s in ('genuine_acc', 'attack_succ')]
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Saved -> {path}')


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\nLoading NUAA test split...')
    genuine_loader  = get_dataloader(split='test', include='genuine',  batch_size=32)
    impostor_loader = get_dataloader(split='test', include='impostor', batch_size=32)
    print(f'  Genuine test : {len(genuine_loader.dataset)}')
    print(f'  Impostor test: {len(impostor_loader.dataset)}')

    print('\nLoading models...')
    models = load_all_models(device)

    all_results = {}
    for name, model in models.items():
        print(f'\n--- {name} ---')
        r = evaluate_physical(model, genuine_loader, impostor_loader, device)
        all_results[name] = r
        print(f"  Genuine accuracy:        {r['genuine_accuracy']:.2f}%")
        print(f"  Physical success rate:   {r['physical_success_rate']:.2f}%")
        print(f"  Physical rejection rate: {r['physical_rejection_rate']:.2f}%")
        print(f"  Confidence drop:         {r['conf_drop']:+.4f}")

    # -- Visualisations --------------------------------------------------------
    print('\nGenerating plots...')
    plot_sample_images(
        genuine_loader, impostor_loader,
        os.path.join(RESULTS_DIR, '01_physical_sample_images.png')
    )
    plot_confidence_distributions(
        all_results,
        os.path.join(RESULTS_DIR, '01_physical_confidence_dist.png')
    )
    plot_confidence_histograms(
        all_results,
        os.path.join(RESULTS_DIR, '01_physical_confidence_hist.png')
    )
    plot_per_subject_success(
        all_results,
        os.path.join(RESULTS_DIR, '01_physical_per_subject.png')
    )
    plot_per_subject_confidence_box(
        all_results,
        os.path.join(RESULTS_DIR, '01_physical_per_subject_conf.png')
    )

    # -- Text / CSV / JSON exports --------------------------------------------
    save_results_txt(all_results, os.path.join(RESULTS_DIR, '01_physical_attack_results.txt'))
    save_per_subject_csv(all_results, os.path.join(RESULTS_DIR, '01_physical_per_subject_stats.csv'))

    json_results = {
        name: {k: v for k, v in r.items() if not k.startswith('_')}
        for name, r in all_results.items()
    }
    with open(os.path.join(RESULTS_DIR, '01_physical_attack_results.json'),
              'w', encoding='utf-8') as fh:
        json.dump(json_results, fh, indent=2)

    print(f'\nAll outputs saved to: {RESULTS_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
