"""
Adversarial Patch Attack Evaluation -- Landmark-Guided Ocular-Region Patch.

A small rectangular patch optimised to maximise misclassification loss when
placed over the eye region.  Hypothesis H3: the ocular region patch causes a
disproportionate accuracy drop because lightweight models rely heavily on
periocular features.

Two placement strategies are compared:
  landmark -- patch placed at each image's actual eye coordinates (from NUAA files)
  fixed    -- patch placed at the dataset-wide mean eye location (weaker adversary)

Ablation: a random-region patch of the same size (lower face) is compared to
prove that eye-region placement is what drives the drop.

Outputs -> results/04_patch/
    04_patch_{model}_eye.pt / .png        -- saved eye-region patch
    04_patch_{model}_random.pt / .png     -- saved random-region patch
    04_patch_training_curves.png          -- loss over optimisation steps
    04_patch_learned_patches.png          -- visual of all learned patches
    04_patch_accuracy_comparison.png      -- clean / eye / random accuracy bar chart
    04_patch_examples_{model}.png         -- visual examples per model
    04_patch_results.txt
    04_patch_results.json
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(__file__))
from utils import (load_all_models, get_dataloader,
                   compute_attack_metrics, CLASS_NAMES, denormalize)
from adversarial_patch import (AdversarialPatch, eye_bbox_from_landmarks,
                               apply_patch_fixed, apply_patch_landmark)

BASE_DIR    = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, 'results', '04_patch')
os.makedirs(RESULTS_DIR, exist_ok=True)

PATCH_SIZE       = (50, 110)
PATCH_ITERATIONS = 500
PATCH_LR         = 0.02
EYE_PADDING      = 12

# Fixed eye bbox (y0, y1, x0, x1) in 224x224 space: eyes at ~32-44% height
FIXED_EYE_BBOX    = (65, 105, 35, 190)
# Same-area patch on lower face / chin for ablation
FIXED_RANDOM_BBOX = (140, 180, 35, 190)


# =============================================================================
# Training
# =============================================================================

def train_patch(model, train_loader, device, mode, fixed_bbox=None):
    if fixed_bbox:
        patch_h = fixed_bbox[1] - fixed_bbox[0]
        patch_w = fixed_bbox[3] - fixed_bbox[2]
    else:
        patch_h, patch_w = PATCH_SIZE
    attacker = AdversarialPatch(patch_size=(patch_h, patch_w), device=device)
    if mode == 'fixed':
        attacker.set_fixed_bbox(fixed_bbox)
    attacker.optimize(model, train_loader,
                      num_iterations=PATCH_ITERATIONS,
                      lr=PATCH_LR, mode=mode, log_every=100)
    return attacker


# =============================================================================
# Plots
# =============================================================================

def plot_training_curves(losses_dict, save_path):
    """
    Loss over optimisation steps for each (model, region) combination.
    Rising loss = the patch is becoming a better adversarial perturbation.
    """
    model_names = list(losses_dict.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = {'eye': '#d62728', 'random': '#1f77b4'}

    for ax, name in zip(axes, model_names):
        for region, color in colors.items():
            losses = losses_dict[name].get(region, [])
            if not losses:
                continue
            window   = max(1, len(losses) // 40)
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            raw_x    = list(range(len(losses)))
            smooth_x = list(range(window - 1, len(losses)))
            ax.plot(raw_x, losses, alpha=0.2, color=color, linewidth=1)
            ax.plot(smooth_x[:len(smoothed)], smoothed,
                    color=color, linewidth=2.5, label=f'{region} patch')

        ax.set_xlabel('Optimisation Step', fontsize=11)
        ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle('Adversarial Patch Training Curves\n'
                 '(Higher loss = patch more effectively fools the model)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_patch_visualization(attackers_eye, attackers_rand, save_path):
    """Show the learned patch tensors side-by-side for all models."""
    model_names = list(attackers_eye.keys())
    n = len(model_names)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    row_labels = ['Eye-Region Patch', 'Random-Region Patch']
    for row, (attackers, label) in enumerate(
        zip([attackers_eye, attackers_rand], row_labels)
    ):
        for col, name in enumerate(model_names):
            patch = attackers[name].patch.detach().cpu()
            img   = (denormalize(patch).permute(1, 2, 0).numpy() * 255).astype('uint8')
            axes[row][col].imshow(img)
            axes[row][col].axis('off')
            if row == 0:
                axes[row][col].set_title(name, fontsize=12, fontweight='bold')
        axes[row][0].set_ylabel(label, fontsize=11, fontweight='bold', rotation=90)

    plt.suptitle('Learned Adversarial Patches\n'
                 '(Optimised to maximise misclassification loss when placed on a face)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_patched_examples(model, test_loader, attacker_eye, attacker_rand,
                          model_name, device, save_path, n_samples=4):
    """Show clean | eye-patch | random-patch for n images."""
    images, class_ids, _, landmarks = next(iter(test_loader))
    images = images[:n_samples].to(device)
    lm_sub = landmarks[:n_samples]

    bboxes_eye = eye_bbox_from_landmarks(lm_sub, padding=EYE_PADDING)
    patch_eye  = attacker_eye.patch.detach()
    patch_rand = attacker_rand.patch.detach()

    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3.5 * n_samples))
    if n_samples == 1:
        axes = [axes]

    titles = ['Clean Image', 'Eye-Region Patch\n(red box)', 'Random-Region Patch\n(blue box)']
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=11, fontweight='bold')

    for row, (img, bbox_e) in enumerate(zip(images, bboxes_eye)):
        img_cpu  = img.cpu()
        clean_np = (denormalize(img_cpu).permute(1, 2, 0).numpy() * 255).astype('uint8')
        axes[row][0].imshow(clean_np)
        axes[row][0].axis('off')

        # Eye patch with red border
        eye_patched = apply_patch_landmark(img.unsqueeze(0), patch_eye, [bbox_e])
        eye_np = (denormalize(eye_patched.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')
        y0e, y1e, x0e, x1e = bbox_e
        eye_np[y0e:y0e+3, x0e:x1e] = [255, 0, 0]
        eye_np[y1e-3:y1e, x0e:x1e] = [255, 0, 0]
        eye_np[y0e:y1e, x0e:x0e+3] = [255, 0, 0]
        eye_np[y0e:y1e, x1e-3:x1e] = [255, 0, 0]
        axes[row][1].imshow(eye_np)
        axes[row][1].axis('off')

        # Random patch with blue border
        rand_patched = apply_patch_fixed(img.unsqueeze(0), patch_rand, FIXED_RANDOM_BBOX)
        rand_np = (denormalize(rand_patched.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')
        y0r, y1r, x0r, x1r = FIXED_RANDOM_BBOX
        rand_np[y0r:y0r+3, x0r:x1r] = [0, 0, 255]
        rand_np[y1r-3:y1r, x0r:x1r] = [0, 0, 255]
        rand_np[y0r:y1r, x0r:x0r+3] = [0, 0, 255]
        rand_np[y0r:y1r, x1r-3:x1r] = [0, 0, 255]
        axes[row][2].imshow(rand_np)
        axes[row][2].axis('off')

    plt.suptitle(f'Adversarial Patch Examples -- {model_name}',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


def plot_accuracy_comparison(all_results, save_path):
    """Grouped bar chart: clean / eye-patch / random-patch accuracy and drop."""
    model_names = list(all_results.keys())
    x = np.arange(len(model_names))
    w = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, ylabel, title in zip(
        axes,
        ['attacked_accuracy', 'accuracy_drop'],
        ['Accuracy (%)', 'Accuracy Drop (pp)'],
        ['Recognition Accuracy per Patch Region', 'Accuracy Drop vs. Clean Baseline']
    ):
        if metric == 'attacked_accuracy':
            clean_vals = [all_results[m]['eye']['clean_accuracy'] for m in model_names]
            eye_vals   = [all_results[m]['eye']['attacked_accuracy'] for m in model_names]
            rand_vals  = [all_results[m]['random']['attacked_accuracy'] for m in model_names]
            ax.bar(x - w, clean_vals, w, label='Clean',               color='#2ca02c', alpha=0.85)
            ax.bar(x,     eye_vals,   w, label='Eye-Region Patch',    color='#d62728', alpha=0.85)
            ax.bar(x + w, rand_vals,  w, label='Random-Region Patch', color='#1f77b4', alpha=0.85)
            for xi, (ca, ea, ra) in enumerate(zip(clean_vals, eye_vals, rand_vals)):
                ax.text(xi - w, ca + 1, f'{ca:.1f}', ha='center', fontsize=8)
                ax.text(xi,     ea + 1, f'{ea:.1f}', ha='center', fontsize=8)
                ax.text(xi + w, ra + 1, f'{ra:.1f}', ha='center', fontsize=8)
            ax.set_ylim(0, 115)
        else:
            eye_vals  = [all_results[m]['eye']['accuracy_drop']    for m in model_names]
            rand_vals = [all_results[m]['random']['accuracy_drop'] for m in model_names]
            ax.bar(x,     eye_vals,  w, label='Eye-Region Patch',    color='#d62728', alpha=0.85)
            ax.bar(x + w, rand_vals, w, label='Random-Region Patch', color='#1f77b4', alpha=0.85)
            for xi, (ea, ra) in enumerate(zip(eye_vals, rand_vals)):
                ax.text(xi,     ea + 0.3, f'{ea:.1f}', ha='center', fontsize=8)
                ax.text(xi + w, ra + 0.3, f'{ra:.1f}', ha='center', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.suptitle('Adversarial Patch: Eye-Region vs. Random-Region Ablation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


# =============================================================================
# Text / JSON exports
# =============================================================================

def save_results_txt(all_results, path):
    lines = ['=' * 70, 'ADVERSARIAL PATCH ATTACK EVALUATION', '=' * 70, '']
    for name, res in all_results.items():
        lines.append(f'Model: {name}')
        lines.append('-' * 40)
        for region in ('eye', 'random'):
            m  = res[region]
            dc = m['avg_conf_attacked'] - m['avg_conf_clean']
            lines.append(
                f"  [{region:>6} patch]  "
                f"clean={m['clean_accuracy']:.2f}%  "
                f"attacked={m['attacked_accuracy']:.2f}%  "
                f"drop={m['accuracy_drop']:.2f}pp  "
                f"ASR={m['attack_success_rate']:.2f}%  "
                f"Delta_conf={dc:+.4f}"
            )
        eye_drop  = res['eye']['accuracy_drop']
        rand_drop = res['random']['accuracy_drop']
        lines.append(f"  Eye vs Random drop advantage: {eye_drop - rand_drop:+.2f}pp")
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

    train_loader = get_dataloader(split='train', include='genuine',
                                  batch_size=16, shuffle=True)
    test_loader  = get_dataloader(split='test',  include='genuine', batch_size=16)
    print(f'Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}')

    print('\nLoading models...')
    models = load_all_models(device)

    all_results    = {}
    losses_dict    = {}
    attackers_eye  = {}
    attackers_rand = {}

    for name, model in models.items():
        print(f'\n{"=" * 50}')
        print(f'  {name}')
        print(f'{"=" * 50}')
        losses_dict[name] = {}

        # -- Eye-region patch --------------------------------------------------
        print('\n  [1/2] Optimising eye-region patch (landmark placement)...')
        att_eye = train_patch(model, train_loader, device, mode='landmark')
        losses_dict[name]['eye'] = att_eye.loss_history
        att_eye.save(os.path.join(RESULTS_DIR, f'04_patch_{name.lower()}_eye.pt'))
        att_eye.save_image(os.path.join(RESULTS_DIR, f'04_patch_{name.lower()}_eye.png'))
        attackers_eye[name] = att_eye

        true_l, clean_p, adv_p, clean_c, adv_c = att_eye.evaluate(
            model, test_loader, mode='landmark')
        eye_metrics = compute_attack_metrics(true_l, clean_p, adv_p, clean_c, adv_c)
        print(f"  Eye patch -> clean={eye_metrics['clean_accuracy']:.2f}%  "
              f"attacked={eye_metrics['attacked_accuracy']:.2f}%  "
              f"ASR={eye_metrics['attack_success_rate']:.2f}%")

        # -- Random-region patch (ablation) ------------------------------------
        print('\n  [2/2] Optimising random-region (chin) patch...')
        att_rand = AdversarialPatch(patch_size=PATCH_SIZE, device=device)
        att_rand.set_fixed_bbox(FIXED_RANDOM_BBOX)
        att_rand.optimize(model, train_loader,
                          num_iterations=PATCH_ITERATIONS,
                          lr=PATCH_LR, mode='fixed', log_every=100)
        losses_dict[name]['random'] = att_rand.loss_history
        att_rand.save(os.path.join(RESULTS_DIR, f'04_patch_{name.lower()}_random.pt'))
        att_rand.save_image(os.path.join(RESULTS_DIR, f'04_patch_{name.lower()}_random.png'))
        attackers_rand[name] = att_rand

        true_l2, clean_p2, adv_p2, clean_c2, adv_c2 = att_rand.evaluate(
            model, test_loader, mode='fixed')
        rand_metrics = compute_attack_metrics(true_l2, clean_p2, adv_p2, clean_c2, adv_c2)
        print(f"  Random patch -> clean={rand_metrics['clean_accuracy']:.2f}%  "
              f"attacked={rand_metrics['attacked_accuracy']:.2f}%  "
              f"ASR={rand_metrics['attack_success_rate']:.2f}%")

        all_results[name] = {'eye': eye_metrics, 'random': rand_metrics}

        # -- Per-model example visualisation -----------------------------------
        plot_patched_examples(
            model, test_loader, att_eye, att_rand, name, device,
            os.path.join(RESULTS_DIR, f'04_patch_examples_{name.lower()}.png'))

    # -- Summary plots ---------------------------------------------------------
    print('\nGenerating summary plots...')
    plot_training_curves(losses_dict,
                         os.path.join(RESULTS_DIR, '04_patch_training_curves.png'))
    plot_patch_visualization(attackers_eye, attackers_rand,
                             os.path.join(RESULTS_DIR, '04_patch_learned_patches.png'))
    plot_accuracy_comparison(all_results,
                             os.path.join(RESULTS_DIR, '04_patch_accuracy_comparison.png'))

    # -- Text / JSON -----------------------------------------------------------
    save_results_txt(all_results, os.path.join(RESULTS_DIR, '04_patch_results.txt'))
    with open(os.path.join(RESULTS_DIR, '04_patch_results.json'), 'w', encoding='utf-8') as fh:
        json.dump(all_results, fh, indent=2)

    print(f'\nAll outputs saved to: {RESULTS_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
