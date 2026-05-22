"""
Attack Visualisation -- Publication-Quality Comparison Figures.

Standalone script that generates clean before/after images for every attack
type used in the thesis.  Run independently; no prior results needed.

Output folder: code/visualizations/

Generated figures
-----------------
  01_physical_genuine_vs_impostor.png   -- side-by-side real vs. printed photo
  02_fgsm_epsilon_progression.png       -- clean -> eps=1 -> 4 -> 8 -> 16/255
  03_fgsm_noise_anatomy.png             -- the invisible noise amplified
  04_pgd_steps_progression.png          -- clean -> 1-step -> 10-step -> 40-step PGD
  05_fgsm_vs_pgd_comparison.png         -- FGSM vs PGD at same epsilon
  06_patch_eye_vs_random.png            -- eye-region vs random-region patch
  07_all_attacks_master.png             -- ONE figure: all attacks on same subject
  08_three_models_gradcam.png           -- Grad-CAM attention: clean vs FGSM per model

Usage:
    python visualize_attacks.py
"""

import os
import sys

# Resolve paths so this script can import from code/attacks/
THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
ATTACKS_DIR = os.path.join(THIS_DIR, 'attacks')
sys.path.insert(0, ATTACKS_DIR)

OUT_DIR = os.path.join(THIS_DIR, 'visualizations')
os.makedirs(OUT_DIR, exist_ok=True)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from utils import (
    load_all_models, get_dataloader, CLASS_NAMES, denormalize
)
from fgsm import fgsm_attack
from pgd  import pgd_attack
from adversarial_patch import (
    AdversarialPatch, eye_bbox_from_landmarks,
    apply_patch_landmark, apply_patch_fixed
)
from gradcam import build_gradcam, overlay_heatmap

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.titlesize':  11,
    'axes.titleweight': 'bold',
    'figure.dpi':      150,
})

FGSM_EPS   = 8 / 255
PGD_ALPHA  = 2 / 255
PGD_STEPS  = 40

# Fixed eye bbox (y0, y1, x0, x1) in 224x224 space
EYE_BBOX    = (65, 105, 35, 190)
RANDOM_BBOX = (140, 180, 35, 190)
EYE_PAD     = 12

SAVE_DPI = 250


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_np(tensor):
    """FloatTensor (C,H,W) in normalised space -> uint8 (H,W,3)."""
    return (denormalize(tensor.detach().cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')


def collect_samples(dataloader, n, unique_subjects=True):
    """Return up to n (image, class_id, landmarks) tuples."""
    seen, samples = set(), []
    for images, class_ids, _, landmarks in dataloader:
        for i in range(len(images)):
            cid = class_ids[i].item()
            if unique_subjects and cid in seen:
                continue
            samples.append((images[i], cid, landmarks[i]))
            seen.add(cid)
            if len(samples) >= n:
                return samples
    return samples


def ax_off(ax, img, title='', ylabel=''):
    ax.imshow(img)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold', rotation=0,
                      labelpad=60, va='center')


def save(fig, name, tight=True):
    path = os.path.join(OUT_DIR, name)
    if tight:
        fig.savefig(path, dpi=SAVE_DPI, bbox_inches='tight', facecolor='white')
    else:
        fig.savefig(path, dpi=SAVE_DPI, facecolor='white')
    plt.close(fig)
    print(f'  Saved -> {path}')


# ===========================================================================
# Figure 1 -- Physical Attack: Genuine vs Impostor
# ===========================================================================

def fig_physical(n_subjects=6):
    gen_loader  = get_dataloader(split='test', include='genuine',  batch_size=32)
    imp_loader  = get_dataloader(split='test', include='impostor', batch_size=32)

    gen_by_subj = {}
    for imgs, cids, _, _ in gen_loader:
        for i in range(len(imgs)):
            cid = cids[i].item()
            if cid not in gen_by_subj:
                gen_by_subj[cid] = imgs[i]
            if len(gen_by_subj) >= n_subjects:
                break
        if len(gen_by_subj) >= n_subjects:
            break

    imp_by_subj = {}
    for imgs, cids, _, _ in imp_loader:
        for i in range(len(imgs)):
            cid = cids[i].item()
            if cid in gen_by_subj and cid not in imp_by_subj:
                imp_by_subj[cid] = imgs[i]
            if len(imp_by_subj) >= n_subjects:
                break
        if len(imp_by_subj) >= n_subjects:
            break

    subjects = sorted(set(gen_by_subj) & set(imp_by_subj))[:n_subjects]
    n = len(subjects)

    fig, axes = plt.subplots(2, n, figsize=(2.8 * n, 6.2))
    fig.patch.set_facecolor('#f8f8f8')

    row_labels = ['Genuine\n(real face)', 'Impostor\n(printed photo)']
    for row, (src, label) in enumerate([(gen_by_subj, row_labels[0]),
                                         (imp_by_subj, row_labels[1])]):
        for col, subj in enumerate(subjects):
            img = to_np(src[subj])
            axes[row][col].imshow(img)
            axes[row][col].axis('off')
            if row == 0:
                axes[row][col].set_title(f'Subject {CLASS_NAMES[subj]}',
                                         fontsize=9, fontweight='bold')
        axes[row][0].set_ylabel(label, fontsize=10, fontweight='bold',
                                rotation=0, labelpad=70, va='center')

    # Coloured border to distinguish rows
    for col in range(n):
        for spine_ax, color in [(axes[0][col], '#2ca02c'), (axes[1][col], '#d62728')]:
            for sp in spine_ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(3)
                sp.set_edgecolor(color)

    fig.suptitle('Physical Print Attack  |  NUAA Dataset\n'
                 'Green border = genuine live face  |  Red border = printed photograph',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    save(fig, '01_physical_genuine_vs_impostor.png')


# ===========================================================================
# Figure 2 -- FGSM: Clean -> Epsilon Progression
# ===========================================================================

def fig_fgsm_progression(model, device, n_subjects=4):
    loader   = get_dataloader(split='test', include='genuine', batch_size=32)
    samples  = collect_samples(loader, n_subjects)
    epsilons = [0, 1/255, 4/255, 8/255, 16/255]
    labels   = ['Clean', 'eps = 1/255\n(~invisible)', 'eps = 4/255',
                 'eps = 8/255\n(typical)', 'eps = 16/255\n(noticeable)']

    n_eps = len(epsilons)
    fig, axes = plt.subplots(n_subjects, n_eps,
                              figsize=(3.2 * n_eps, 3.2 * n_subjects))
    fig.patch.set_facecolor('#f8f8f8')

    for col, label in enumerate(labels):
        axes[0][col].set_title(label, fontsize=10, fontweight='bold', pad=6)

    for row, (img, cid, _) in enumerate(samples):
        img_d = img.to(device)
        for col, eps in enumerate(epsilons):
            if eps == 0:
                vis = to_np(img)
            else:
                adv, _ = fgsm_attack(model, img_d.unsqueeze(0),
                                     torch.tensor([cid], device=device), eps, device)
                vis = to_np(adv.squeeze(0))
            axes[row][col].imshow(vis)
            axes[row][col].axis('off')
        axes[row][0].set_ylabel(f'Subject\n{CLASS_NAMES[cid]}',
                                fontsize=9, rotation=0, labelpad=55, va='center')

    fig.suptitle('FGSM Attack: Effect of Perturbation Budget (epsilon)\n'
                 'Each column increases the L-inf perturbation norm',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, '02_fgsm_epsilon_progression.png')


# ===========================================================================
# Figure 3 -- FGSM: Noise Anatomy (what the attack actually adds)
# ===========================================================================

def fig_fgsm_noise(model, device, n_subjects=3):
    loader  = get_dataloader(split='test', include='genuine', batch_size=32)
    samples = collect_samples(loader, n_subjects)
    eps     = 8 / 255

    col_titles = ['Original\nImage',
                  'Adversarial\nImage (eps=8/255)',
                  'Added Noise\n(amplified x10)',
                  'Added Noise\n(amplified x50)',
                  'Noise\nPattern Only']

    fig, axes = plt.subplots(n_subjects, 5,
                              figsize=(3.2 * 5, 3.2 * n_subjects))
    fig.patch.set_facecolor('#f8f8f8')

    for col, t in enumerate(col_titles):
        axes[0][col].set_title(t, fontsize=10, fontweight='bold', pad=6)

    for row, (img, cid, _) in enumerate(samples):
        img_d = img.to(device)
        adv, _ = fgsm_attack(model, img_d.unsqueeze(0),
                              torch.tensor([cid], device=device), eps, device)
        noise = (adv.squeeze(0) - img_d).detach().cpu()

        clean_np = to_np(img)
        adv_np   = to_np(adv.squeeze(0))

        # Noise amplified: shift [min,max] -> [0,1] then scale
        noise_np  = noise.permute(1, 2, 0).numpy()
        n10 = np.clip(noise_np * 10 + 0.5, 0, 1)
        n50 = np.clip(noise_np * 50 + 0.5, 0, 1)
        # Pattern: sign of noise (shows gradient direction)
        sign = (np.sign(noise_np) + 1) / 2   # 0 or 1 per channel

        for col, vis in enumerate([clean_np,
                                    adv_np,
                                    (n10 * 255).astype('uint8'),
                                    (n50 * 255).astype('uint8'),
                                    (sign * 255).astype('uint8')]):
            axes[row][col].imshow(vis if vis.dtype == np.uint8 else vis)
            axes[row][col].axis('off')

        axes[row][0].set_ylabel(f'Subject\n{CLASS_NAMES[cid]}',
                                fontsize=9, rotation=0, labelpad=55, va='center')

    fig.suptitle('FGSM Noise Anatomy  |  eps = 8/255\n'
                 'The perturbation is imperceptible to humans '
                 'but shifts every pixel by the gradient sign',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, '03_fgsm_noise_anatomy.png')


# ===========================================================================
# Figure 4 -- PGD: Steps Progression
# ===========================================================================

def fig_pgd_steps(model, device, n_subjects=4):
    loader  = get_dataloader(split='test', include='genuine', batch_size=32)
    samples = collect_samples(loader, n_subjects)

    configs = [(0,  0,    'Clean'),
               (1,  8/255, 'PGD\n1 step'),
               (5,  8/255, 'PGD\n5 steps'),
               (10, 8/255, 'PGD\n10 steps'),
               (20, 8/255, 'PGD\n20 steps'),
               (40, 8/255, 'PGD\n40 steps')]

    fig, axes = plt.subplots(n_subjects, len(configs),
                              figsize=(3.2 * len(configs), 3.2 * n_subjects))
    fig.patch.set_facecolor('#f8f8f8')

    for col, (_, _, label) in enumerate(configs):
        axes[0][col].set_title(label, fontsize=10, fontweight='bold', pad=6)

    for row, (img, cid, _) in enumerate(samples):
        img_d = img.to(device)
        for col, (steps, eps, _) in enumerate(configs):
            if steps == 0:
                vis = to_np(img)
            else:
                adv = pgd_attack(model, img_d.unsqueeze(0),
                                  torch.tensor([cid], device=device),
                                  eps, eps / 4, steps, device)
                vis = to_np(adv.squeeze(0))
            axes[row][col].imshow(vis)
            axes[row][col].axis('off')
        axes[row][0].set_ylabel(f'Subject\n{CLASS_NAMES[cid]}',
                                fontsize=9, rotation=0, labelpad=55, va='center')

    fig.suptitle('PGD Attack: Effect of Number of Gradient Steps  |  eps = 8/255\n'
                 'More steps -> stronger attack -> harder for model to resist',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, '04_pgd_steps_progression.png')


# ===========================================================================
# Figure 5 -- FGSM vs PGD side-by-side
# ===========================================================================

def fig_fgsm_vs_pgd(model, device, n_subjects=4):
    loader  = get_dataloader(split='test', include='genuine', batch_size=32)
    samples = collect_samples(loader, n_subjects)
    eps     = 8 / 255

    col_titles = ['Clean', 'FGSM\neps=8/255\n(1 step)',
                  'PGD-10\neps=8/255',
                  'PGD-40\neps=8/255\n(strongest)']

    fig, axes = plt.subplots(n_subjects, 4,
                              figsize=(3.2 * 4, 3.2 * n_subjects))
    fig.patch.set_facecolor('#f8f8f8')

    for col, t in enumerate(col_titles):
        axes[0][col].set_title(t, fontsize=10, fontweight='bold', pad=6)

    for row, (img, cid, _) in enumerate(samples):
        img_d = img.to(device)
        label = torch.tensor([cid], device=device)

        adv_fgsm, _ = fgsm_attack(model, img_d.unsqueeze(0), label, eps, device)
        adv_pgd10   = pgd_attack(model, img_d.unsqueeze(0), label, eps, eps/4, 10, device)
        adv_pgd40   = pgd_attack(model, img_d.unsqueeze(0), label, eps, eps/4, 40, device)

        for col, vis_t in enumerate([img, adv_fgsm.squeeze(0),
                                      adv_pgd10.squeeze(0), adv_pgd40.squeeze(0)]):
            axes[row][col].imshow(to_np(vis_t))
            axes[row][col].axis('off')
        axes[row][0].set_ylabel(f'Subject\n{CLASS_NAMES[cid]}',
                                fontsize=9, rotation=0, labelpad=55, va='center')

    fig.suptitle('FGSM vs. PGD Attack Comparison  |  eps = 8/255\n'
                 'Same budget, more steps = strictly stronger adversarial example',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, '05_fgsm_vs_pgd_comparison.png')


# ===========================================================================
# Figure 6 -- Adversarial Patch: Eye vs Random Region
# ===========================================================================

def fig_patch(model, device, n_subjects=4):
    train_loader = get_dataloader(split='train', include='genuine',
                                   batch_size=16, shuffle=True)
    test_loader  = get_dataloader(split='test',  include='genuine', batch_size=32)

    # Train a quick patch
    print('  Training eye-region patch (300 steps)...')
    att_eye = AdversarialPatch(patch_size=(40, 100), device=device)
    att_eye.optimize(model, train_loader, num_iterations=300,
                     lr=0.02, mode='landmark', log_every=100)

    print('  Training random-region patch (300 steps)...')
    att_rand = AdversarialPatch(patch_size=(40, 100), device=device)
    att_rand.set_fixed_bbox(RANDOM_BBOX)
    att_rand.optimize(model, train_loader, num_iterations=300,
                      lr=0.02, mode='fixed', log_every=100)

    samples = collect_samples(test_loader, n_subjects)

    col_titles = ['Clean Image',
                  'Eye-Region Patch\n(optimised)',
                  'Patch Location\n(highlighted)',
                  'Random-Region Patch\n(chin area)']

    fig, axes = plt.subplots(n_subjects, 4,
                              figsize=(3.4 * 4, 3.4 * n_subjects))
    fig.patch.set_facecolor('#f8f8f8')

    for col, t in enumerate(col_titles):
        axes[0][col].set_title(t, fontsize=10, fontweight='bold', pad=6)

    for row, (img, cid, lm) in enumerate(samples):
        img_d  = img.to(device)
        bbox_e = eye_bbox_from_landmarks([lm], padding=EYE_PAD)[0]

        # Eye-patched
        eye_p = apply_patch_landmark(img_d.unsqueeze(0), att_eye.patch.detach(), [bbox_e])
        eye_np = to_np(eye_p.squeeze(0))

        # Highlighted version (semi-transparent red box)
        eye_hi = eye_np.copy().astype('float32')
        y0, y1, x0, x1 = bbox_e
        eye_hi[y0:y1, x0:x1] = eye_hi[y0:y1, x0:x1] * 0.5 + np.array([255, 0, 0]) * 0.5
        eye_hi = eye_hi.astype('uint8')
        # Draw border
        for thickness in range(3):
            eye_hi[y0+thickness, x0:x1] = [255, 50, 50]
            eye_hi[y1-1-thickness, x0:x1] = [255, 50, 50]
            eye_hi[y0:y1, x0+thickness] = [255, 50, 50]
            eye_hi[y0:y1, x1-1-thickness] = [255, 50, 50]

        # Random-patched
        rand_p = apply_patch_fixed(img_d.unsqueeze(0), att_rand.patch.detach(), RANDOM_BBOX)
        rand_np = to_np(rand_p.squeeze(0))
        y0r, y1r, x0r, x1r = RANDOM_BBOX
        for thickness in range(3):
            rand_np[y0r+thickness, x0r:x1r] = [50, 50, 255]
            rand_np[y1r-1-thickness, x0r:x1r] = [50, 50, 255]
            rand_np[y0r:y1r, x0r+thickness] = [50, 50, 255]
            rand_np[y0r:y1r, x1r-1-thickness] = [50, 50, 255]

        for col, vis in enumerate([to_np(img), eye_np, eye_hi, rand_np]):
            axes[row][col].imshow(vis)
            axes[row][col].axis('off')
        axes[row][0].set_ylabel(f'Subject\n{CLASS_NAMES[cid]}',
                                fontsize=9, rotation=0, labelpad=55, va='center')

    red_patch  = mpatches.Patch(color='#d62728', label='Eye-region patch (hypothesis region)')
    blue_patch = mpatches.Patch(color='#1f77b4', label='Random-region patch (ablation)')
    fig.legend(handles=[red_patch, blue_patch], fontsize=10,
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Adversarial Patch Attack: Eye Region vs. Random Region\n'
                 'Eye-region patch exploits the model\'s reliance on periocular features',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, '06_patch_eye_vs_random.png')

    return att_eye   # reuse for figure 7


# ===========================================================================
# Figure 7 -- Master: All Attacks on the Same Subject
# ===========================================================================

def fig_all_attacks_master(model, device, att_eye, n_subjects=3):
    gen_loader  = get_dataloader(split='test', include='genuine',  batch_size=32)
    imp_loader  = get_dataloader(split='test', include='impostor', batch_size=32)

    gen_samples = collect_samples(gen_loader, n_subjects)

    # Collect matching impostors
    needed = {s[1] for s in gen_samples}
    imp_map = {}
    for imgs, cids, _, _ in imp_loader:
        for i in range(len(imgs)):
            cid = cids[i].item()
            if cid in needed and cid not in imp_map:
                imp_map[cid] = imgs[i]
        if len(imp_map) >= len(needed):
            break

    col_titles = ['Clean\n(baseline)',
                  'Physical\n(print attack)',
                  'FGSM\neps=8/255',
                  'PGD-40\neps=8/255',
                  'Eye-Patch\n(optimised)']

    eps = 8 / 255
    fig, axes = plt.subplots(n_subjects, 5,
                              figsize=(3.4 * 5, 3.4 * n_subjects))
    fig.patch.set_facecolor('#f8f8f8')

    for col, t in enumerate(col_titles):
        axes[0][col].set_title(t, fontsize=10, fontweight='bold', pad=6)

    for row, (img, cid, lm) in enumerate(gen_samples):
        img_d = img.to(device)
        label = torch.tensor([cid], device=device)
        bbox_e = eye_bbox_from_landmarks([lm], padding=EYE_PAD)[0]

        # Physical: use matching impostor if available
        physical = imp_map.get(cid)
        phys_vis = to_np(physical) if physical is not None else np.zeros((224, 224, 3), dtype='uint8')

        adv_fgsm, _ = fgsm_attack(model, img_d.unsqueeze(0), label, eps, device)
        adv_pgd      = pgd_attack(model, img_d.unsqueeze(0), label, eps, eps/4, 40, device)
        adv_patch    = apply_patch_landmark(img_d.unsqueeze(0),
                                             att_eye.patch.detach(), [bbox_e])

        for col, vis_t in enumerate([img,
                                      physical if physical is not None else img,
                                      adv_fgsm.squeeze(0),
                                      adv_pgd.squeeze(0),
                                      adv_patch.squeeze(0)]):
            if isinstance(vis_t, torch.Tensor):
                axes[row][col].imshow(to_np(vis_t))
            else:
                axes[row][col].imshow(to_np(vis_t))
            axes[row][col].axis('off')

        axes[row][0].set_ylabel(f'Subject {CLASS_NAMES[cid]}',
                                fontsize=10, fontweight='bold',
                                rotation=0, labelpad=65, va='center')

    # Attack-type colour legend
    colors = ['#555555', '#d62728', '#ff7f0e', '#9467bd', '#1f77b4']
    patches = [mpatches.Patch(color=c, label=t)
               for c, t in zip(colors, col_titles)]
    fig.legend(handles=patches, fontsize=9, loc='lower center',
               ncol=5, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle('All Attack Types on the Same Subjects\n'
                 'Left to right: increasing sophistication of the adversary',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save(fig, '07_all_attacks_master.png')


# ===========================================================================
# Figure 8 -- Grad-CAM: Where Each Model Looks (clean vs attacked)
# ===========================================================================

def fig_gradcam_all_models(models, device):
    loader  = get_dataloader(split='test', include='genuine', batch_size=32)
    samples = collect_samples(loader, 2)   # 2 subjects

    model_names = list(models.keys())
    n_models    = len(model_names)
    n_subjects  = len(samples)
    eps = 8 / 255

    # 3 cols per model: image | clean CAM | FGSM CAM
    n_cols = 1 + n_models * 2   # first col = image, then pairs
    fig, axes = plt.subplots(n_subjects, n_cols,
                              figsize=(3.0 * n_cols, 3.5 * n_subjects))
    fig.patch.set_facecolor('#f8f8f8')
    if n_subjects == 1:
        axes = [axes]

    # Header
    axes[0][0].set_title('Image', fontsize=10, fontweight='bold', pad=6)
    col = 1
    for name in model_names:
        axes[0][col].set_title(f'{name}\nGrad-CAM (Clean)', fontsize=9, fontweight='bold', pad=6)
        axes[0][col+1].set_title(f'{name}\nGrad-CAM (FGSM)', fontsize=9, fontweight='bold', pad=6)
        col += 2

    for row, (img, cid, _) in enumerate(samples):
        img_d  = img.to(device)
        img_np = to_np(img)

        axes[row][0].imshow(img_np)
        axes[row][0].axis('off')
        axes[row][0].set_ylabel(f'Subject\n{CLASS_NAMES[cid]}',
                                fontsize=9, rotation=0, labelpad=55, va='center')

        col = 1
        for name, model in models.items():
            model.eval()
            gradcam = build_gradcam(model, name)

            cam_clean = gradcam.generate(img_d.unsqueeze(0).requires_grad_(True), cid)
            axes[row][col].imshow(overlay_heatmap(img_np.copy(), cam_clean))
            axes[row][col].axis('off')

            adv_f, _ = fgsm_attack(model, img_d.unsqueeze(0),
                                    torch.tensor([cid], device=device), eps, device)
            adv_f_np  = to_np(adv_f.squeeze(0))
            cam_adv   = gradcam.generate(adv_f.requires_grad_(True), cid)
            axes[row][col+1].imshow(overlay_heatmap(adv_f_np.copy(), cam_adv))
            axes[row][col+1].axis('off')

            gradcam.remove_hooks()
            col += 2

    fig.suptitle('Grad-CAM: Where Each Model Focuses Attention\n'
                 'Clean image (left pair) vs. FGSM-perturbed (right pair) per architecture',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, '08_three_models_gradcam.png')


# ===========================================================================
# Main
# ===========================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Output folder: {OUT_DIR}\n')

    print('Loading models...')
    models = load_all_models(device)
    # Use MobileNetV3 as the representative model for per-model figures
    # (fastest; EfficientNet would also work)
    primary_model = models['MobileNetV3']
    primary_model.eval()

    print('\n[1/8] Physical: genuine vs impostor samples...')
    fig_physical(n_subjects=6)

    print('\n[2/8] FGSM: epsilon progression...')
    fig_fgsm_progression(primary_model, device, n_subjects=4)

    print('\n[3/8] FGSM: noise anatomy (amplified perturbation)...')
    fig_fgsm_noise(primary_model, device, n_subjects=3)

    print('\n[4/8] PGD: steps progression...')
    fig_pgd_steps(primary_model, device, n_subjects=4)

    print('\n[5/8] FGSM vs PGD comparison...')
    fig_fgsm_vs_pgd(primary_model, device, n_subjects=4)

    print('\n[6/8] Adversarial patch: eye vs random region...')
    att_eye = fig_patch(primary_model, device, n_subjects=4)

    print('\n[7/8] Master figure: all attacks on same subjects...')
    fig_all_attacks_master(primary_model, device, att_eye, n_subjects=3)

    print('\n[8/8] Grad-CAM: attention maps for all three models...')
    fig_gradcam_all_models(models, device)

    print(f'\nAll 8 figures saved to: {OUT_DIR}')
    print('\nFile list:')
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f'  {f:<45} {size//1024:>5} KB')


if __name__ == '__main__':
    main()
