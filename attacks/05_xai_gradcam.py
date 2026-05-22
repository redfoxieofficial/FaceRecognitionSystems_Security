"""
XAI -- Grad-CAM Feature Attribution Under Adversarial Attacks.

Generates side-by-side Grad-CAM heatmaps showing how each model's internal
attention shifts when presented with clean vs. adversarially perturbed faces.

Outputs -> results/05_gradcam/
    05_gradcam_{model}_digital.png     -- clean vs FGSM vs PGD
    05_gradcam_{model}_physical.png    -- genuine vs impostor (print attack)
    05_gradcam_{model}_patch.png       -- clean vs adversarial patch
    05_gradcam_all_models_clean.png    -- all 3 models on same clean image
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from utils import (load_all_models, get_dataloader,
                   NUAADataset, TEST_TRANSFORM, CLASS_NAMES, denormalize)
from gradcam import build_gradcam, overlay_heatmap
from fgsm import fgsm_attack
from pgd  import pgd_attack
from adversarial_patch import (AdversarialPatch, eye_bbox_from_landmarks,
                               apply_patch_landmark)

BASE_DIR    = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, 'results', '05_gradcam')
os.makedirs(RESULTS_DIR, exist_ok=True)

FGSM_EPS  = 8 / 255
PGD_EPS   = 8 / 255
PGD_ALPHA = 2 / 255
PGD_STEPS = 40
N_SAMPLES = 4
PATCH_ITERATIONS = 300


# =============================================================================
# Sample selection
# =============================================================================

def get_diverse_samples(dataloader, n):
    """Return n samples covering different subjects."""
    seen, samples = set(), []
    for images, class_ids, _, landmarks in dataloader:
        for i in range(len(images)):
            cid = class_ids[i].item()
            if cid not in seen and len(samples) < n:
                samples.append((images[i].unsqueeze(0), cid, landmarks[i]))
                seen.add(cid)
        if len(samples) >= n:
            break
    return samples


# =============================================================================
# Visualisation builders
# =============================================================================

def gradcam_clean_vs_digital(model, model_name, samples, device, save_path):
    """6-column grid: Clean | CAM(Clean) | FGSM | CAM(FGSM) | PGD | CAM(PGD)"""
    gradcam = build_gradcam(model, model_name)
    n = len(samples)
    fig, axes = plt.subplots(n, 6, figsize=(21, 3.5 * n))
    if n == 1:
        axes = [axes]

    col_titles = ['Clean', 'Grad-CAM\n(Clean)',
                  f'FGSM eps={FGSM_EPS*255:.0f}/255', 'Grad-CAM\n(FGSM)',
                  f'PGD eps={PGD_EPS*255:.0f}/255\n{PGD_STEPS} steps', 'Grad-CAM\n(PGD)']
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=9, fontweight='bold')

    model.eval()
    for row, (img_t, cid, lm) in enumerate(samples):
        img_t  = img_t.to(device)
        img_np = (denormalize(img_t.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')

        axes[row][0].imshow(img_np);  axes[row][0].axis('off')
        cam_c = gradcam.generate(img_t.requires_grad_(True), cid)
        axes[row][1].imshow(overlay_heatmap(img_np.copy(), cam_c));  axes[row][1].axis('off')

        adv_f, _ = fgsm_attack(model, img_t, torch.tensor([cid], device=device), FGSM_EPS, device)
        adv_f_np = (denormalize(adv_f.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')
        axes[row][2].imshow(adv_f_np); axes[row][2].axis('off')
        cam_f = gradcam.generate(adv_f.requires_grad_(True), cid)
        axes[row][3].imshow(overlay_heatmap(adv_f_np.copy(), cam_f)); axes[row][3].axis('off')

        adv_p = pgd_attack(model, img_t, torch.tensor([cid], device=device),
                           PGD_EPS, PGD_ALPHA, PGD_STEPS, device)
        adv_p_np = (denormalize(adv_p.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')
        axes[row][4].imshow(adv_p_np); axes[row][4].axis('off')
        cam_p = gradcam.generate(adv_p.requires_grad_(True), cid)
        axes[row][5].imshow(overlay_heatmap(adv_p_np.copy(), cam_p)); axes[row][5].axis('off')

        axes[row][0].set_ylabel(f'Subject {CLASS_NAMES[cid]}', fontsize=9, rotation=90)

    plt.suptitle(f'Grad-CAM Attribution -- {model_name}: Clean vs. Digital Attacks',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')
    gradcam.remove_hooks()


def gradcam_clean_vs_patch(model, model_name, samples, device, patch_attacker, save_path):
    """4-column grid: Clean | CAM(Clean) | Patched | CAM(Patched)"""
    gradcam = build_gradcam(model, model_name)
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = [axes]

    col_titles = ['Clean', 'Grad-CAM\n(Clean)', 'Eye-Region\nPatch', 'Grad-CAM\n(Patch)']
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=10, fontweight='bold')

    model.eval()
    for row, (img_t, cid, lm) in enumerate(samples):
        img_t  = img_t.to(device)
        img_np = (denormalize(img_t.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')

        axes[row][0].imshow(img_np); axes[row][0].axis('off')
        cam_c = gradcam.generate(img_t.requires_grad_(True), cid)
        axes[row][1].imshow(overlay_heatmap(img_np.copy(), cam_c)); axes[row][1].axis('off')

        bbox    = eye_bbox_from_landmarks([lm])
        patched = apply_patch_landmark(img_t, patch_attacker.patch.detach(), bbox)
        pat_np  = (denormalize(patched.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')
        axes[row][2].imshow(pat_np); axes[row][2].axis('off')
        cam_pat = gradcam.generate(patched.requires_grad_(True), cid)
        axes[row][3].imshow(overlay_heatmap(pat_np.copy(), cam_pat)); axes[row][3].axis('off')

        axes[row][0].set_ylabel(f'Subject {CLASS_NAMES[cid]}', fontsize=9, rotation=90)

    plt.suptitle(f'Grad-CAM Attribution -- {model_name}: Clean vs. Adversarial Patch',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')
    gradcam.remove_hooks()


def gradcam_physical(model, model_name, genuine_samples, impostor_loader,
                     device, save_path):
    """4-column grid: Genuine | CAM(Genuine) | Printed Photo | CAM(Print)"""
    gradcam = build_gradcam(model, model_name)

    subject_ids = {s[1] for s in genuine_samples}
    impostor_by_subject = {}
    for images, class_ids, _, _ in impostor_loader:
        for i in range(len(images)):
            cid = class_ids[i].item()
            if cid in subject_ids and cid not in impostor_by_subject:
                impostor_by_subject[cid] = images[i].unsqueeze(0)
        if len(impostor_by_subject) >= len(subject_ids):
            break

    samples_with_impostors = [
        (img_t, cid, lm, impostor_by_subject.get(cid))
        for (img_t, cid, lm) in genuine_samples
        if cid in impostor_by_subject
    ]

    if not samples_with_impostors:
        print(f'  [SKIP] No matching impostors found for {model_name}')
        gradcam.remove_hooks()
        return

    n = len(samples_with_impostors)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = [axes]

    col_titles = ['Genuine Face', 'Grad-CAM\n(Genuine)',
                  'Printed Photo', 'Grad-CAM\n(Printed)']
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=10, fontweight='bold')

    model.eval()
    for row, (img_t, cid, lm, imp_t) in enumerate(samples_with_impostors):
        img_t  = img_t.to(device)
        img_np = (denormalize(img_t.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')
        axes[row][0].imshow(img_np); axes[row][0].axis('off')
        cam_g = gradcam.generate(img_t.requires_grad_(True), cid)
        axes[row][1].imshow(overlay_heatmap(img_np.copy(), cam_g)); axes[row][1].axis('off')

        imp_t  = imp_t.to(device)
        imp_np = (denormalize(imp_t.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')
        axes[row][2].imshow(imp_np); axes[row][2].axis('off')
        cam_i = gradcam.generate(imp_t.requires_grad_(True), cid)
        axes[row][3].imshow(overlay_heatmap(imp_np.copy(), cam_i)); axes[row][3].axis('off')

        axes[row][0].set_ylabel(f'Subject {CLASS_NAMES[cid]}', fontsize=9, rotation=90)

    plt.suptitle(f'Grad-CAM Attribution -- {model_name}: Genuine vs. Physical Print Attack',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')
    gradcam.remove_hooks()


def gradcam_all_models_comparison(models, model_names, sample, device, save_path):
    """
    For a single image: show all 3 models' clean Grad-CAM side-by-side.
    Reveals which facial regions each architecture relies on.
    """
    img_t, cid, lm = sample
    img_t  = img_t.to(device)
    img_np = (denormalize(img_t.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')

    n = len(model_names)
    fig, axes = plt.subplots(2, n + 1, figsize=(4 * (n + 1), 8))

    axes[0][0].imshow(img_np);   axes[0][0].axis('off')
    axes[0][0].set_title('Original Image', fontsize=11, fontweight='bold')
    axes[1][0].axis('off')

    for col, (name, model) in enumerate(zip(model_names, models.values()), start=1):
        model.eval()
        gradcam = build_gradcam(model, name)

        cam_clean = gradcam.generate(img_t.requires_grad_(True), cid)
        axes[0][col].imshow(overlay_heatmap(img_np.copy(), cam_clean))
        axes[0][col].axis('off')
        axes[0][col].set_title(f'{name}\nClean', fontsize=10, fontweight='bold')

        # Also show FGSM-perturbed attention
        adv_f, _ = fgsm_attack(model, img_t, torch.tensor([cid], device=device), FGSM_EPS, device)
        adv_f_np  = (denormalize(adv_f.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')
        cam_adv   = gradcam.generate(adv_f.requires_grad_(True), cid)
        axes[1][col].imshow(overlay_heatmap(adv_f_np.copy(), cam_adv))
        axes[1][col].axis('off')
        axes[1][col].set_title(f'{name}\nFGSM', fontsize=10, fontweight='bold')

        gradcam.remove_hooks()

    axes[1][0].text(0.5, 0.5, f'Subject:\n{CLASS_NAMES[cid]}',
                    ha='center', va='center', fontsize=12, transform=axes[1][0].transAxes)

    plt.suptitle('Grad-CAM: Where Each Architecture Looks (Clean vs. FGSM)\n'
                 'Left column = original; each pair = clean attention / FGSM attention',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {save_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    genuine_loader  = get_dataloader(split='test', include='genuine',  batch_size=8)
    impostor_loader = get_dataloader(split='test', include='impostor', batch_size=8)

    samples = get_diverse_samples(genuine_loader, N_SAMPLES)
    print(f'Selected {len(samples)} diverse subjects for XAI visualisation.')

    print('\nLoading models...')
    models = load_all_models(device)

    train_loader = get_dataloader(split='train', include='genuine',
                                  batch_size=16, shuffle=True)

    for name, model in models.items():
        print(f'\n{"=" * 50}')
        print(f'  Grad-CAM: {name}')
        print(f'{"=" * 50}')

        gradcam_clean_vs_digital(
            model, name, samples, device,
            os.path.join(RESULTS_DIR, f'05_gradcam_{name.lower()}_digital.png'))

        gradcam_physical(
            model, name, samples, impostor_loader, device,
            os.path.join(RESULTS_DIR, f'05_gradcam_{name.lower()}_physical.png'))

        print(f'  Training demo patch for Grad-CAM ({PATCH_ITERATIONS} iters)...')
        attacker = AdversarialPatch(patch_size=(50, 110), device=device)
        attacker.optimize(model, train_loader,
                          num_iterations=PATCH_ITERATIONS,
                          lr=0.02, mode='landmark', log_every=100)

        gradcam_clean_vs_patch(
            model, name, samples, device, attacker,
            os.path.join(RESULTS_DIR, f'05_gradcam_{name.lower()}_patch.png'))

    # Cross-model comparison on a single sample
    if samples:
        gradcam_all_models_comparison(
            models, list(models.keys()), samples[0], device,
            os.path.join(RESULTS_DIR, '05_gradcam_all_models_comparison.png'))

    print(f'\nAll outputs saved to: {RESULTS_DIR}')
    print('Done.')


if __name__ == '__main__':
    main()
