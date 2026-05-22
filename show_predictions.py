"""
Prediction Labels Visualisation -- What Does the Classifier Think After Each Attack?

For every attack type, shows a grid of:
  - The attacked image
  - True subject label (always shown in blue)
  - What the model actually predicts (green = correct, red = wrong + shows predicted name)
  - Model confidence score

Output folder: code/predictions/

Generated figures
-----------------
  pred_01_physical_{model}.png        -- what the model predicts on printed photos
  pred_02_fgsm_{model}.png            -- predicted labels under FGSM
  pred_03_pgd_{model}.png             -- predicted labels under PGD-40
  pred_04_patch_{model}.png           -- predicted labels with eye-region patch
  pred_05_all_attacks_summary.png     -- compact matrix: true → predicted per attack
  pred_06_confusion_shift.png         -- bar chart: how many flip to each wrong class

Usage:
    python show_predictions.py
"""

import os
import sys

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
ATTACKS_DIR = os.path.join(THIS_DIR, 'attacks')
sys.path.insert(0, ATTACKS_DIR)

OUT_DIR = os.path.join(THIS_DIR, 'predictions')
os.makedirs(OUT_DIR, exist_ok=True)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

from utils import (load_all_models, get_dataloader, CLASS_NAMES, denormalize)
from fgsm import fgsm_attack
from pgd  import pgd_attack
from adversarial_patch import (
    AdversarialPatch, eye_bbox_from_landmarks, apply_patch_landmark, apply_patch_fixed
)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 130})

FGSM_EPS    = 8 / 255
PGD_EPS     = 8 / 255
PGD_ALPHA   = 2 / 255
PGD_STEPS   = 40
EYE_BBOX    = (65, 105, 35, 190)
RANDOM_BBOX = (140, 180, 35, 190)
EYE_PAD     = 12
N_SUBJECTS  = 8
SAVE_DPI    = 220

# Colours used for annotation boxes
COL_CORRECT = '#1a7a1a'   # dark green
COL_WRONG   = '#c0392b'   # dark red
COL_TRUE    = '#154360'   # dark blue
COL_CONF    = '#6c3483'   # purple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_np(tensor):
    return (denormalize(tensor.detach().cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')


def collect_samples(loader, n):
    seen, samples = set(), []
    for images, class_ids, _, landmarks in loader:
        for i in range(len(images)):
            cid = class_ids[i].item()
            if cid not in seen:
                seen.add(cid)
                samples.append((images[i], cid, landmarks[i]))
            if len(samples) >= n:
                return samples
    return samples


@torch.no_grad()
def get_top3(model, img_tensor, device):
    """Returns [(class_idx, class_name, confidence), ...] top-3 predictions."""
    logits = model(img_tensor.to(device))
    probs  = torch.softmax(logits, dim=1).squeeze(0).cpu()
    top3   = probs.topk(3)
    return [(top3.indices[k].item(),
             CLASS_NAMES[top3.indices[k].item()],
             top3.values[k].item()) for k in range(3)]


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved -> {path}')


# ---------------------------------------------------------------------------
# Core figure builder
# ---------------------------------------------------------------------------

def fig_attack_predictions(attack_name, attacked_imgs_fn,
                            samples, model, model_name, device, fname):
    """
    One figure per (attack × model).

    Layout  (N_SUBJECTS rows):
      Col 0 : clean image
      Col 1 : clean label annotation (true / top-1 predicted / confidence)
      Col 2 : attacked image
      Col 3 : attacked label annotation (true / top-1 predicted / confidence)
    """
    n = len(samples)
    fig = plt.figure(figsize=(12, 2.8 * n), facecolor='white')
    fig.suptitle(f'{attack_name}  --  {model_name}\n'
                 f'Green = correct prediction | Red = attack succeeded (wrong label)',
                 fontsize=12, fontweight='bold', y=1.01)

    col_w = [1, 1.2, 1, 1.2]
    gs = fig.add_gridspec(n, 4, width_ratios=col_w,
                          hspace=0.35, wspace=0.05,
                          left=0.02, right=0.98, top=0.96, bottom=0.02)

    for row, (img, cid, lm) in enumerate(samples):
        img_d = img.to(device)
        adv   = attacked_imgs_fn(img_d, cid, lm)

        # ── Clean image ──────────────────────────────────────────────────────
        ax_img_c = fig.add_subplot(gs[row, 0])
        ax_img_c.imshow(to_np(img))
        ax_img_c.axis('off')
        if row == 0:
            ax_img_c.set_title('Clean Image', fontsize=10, fontweight='bold', pad=4)

        # ── Clean annotation ─────────────────────────────────────────────────
        ax_ann_c = fig.add_subplot(gs[row, 1])
        ax_ann_c.axis('off')
        if row == 0:
            ax_ann_c.set_title('Clean Prediction', fontsize=10, fontweight='bold', pad=4)
        top3_c = get_top3(model, img_d.unsqueeze(0), device)
        draw_annotation(ax_ann_c, CLASS_NAMES[cid], top3_c)

        # ── Attacked image ────────────────────────────────────────────────────
        ax_img_a = fig.add_subplot(gs[row, 2])
        ax_img_a.imshow(to_np(adv))
        ax_img_a.axis('off')
        if row == 0:
            ax_img_a.set_title('After Attack', fontsize=10, fontweight='bold', pad=4)

        # ── Attacked annotation ───────────────────────────────────────────────
        ax_ann_a = fig.add_subplot(gs[row, 3])
        ax_ann_a.axis('off')
        if row == 0:
            ax_ann_a.set_title('Attacked Prediction', fontsize=10, fontweight='bold', pad=4)
        top3_a = get_top3(model, adv.unsqueeze(0), device)
        draw_annotation(ax_ann_a, CLASS_NAMES[cid], top3_a)

        # Row label
        ax_img_c.set_ylabel(f'Subject\n{CLASS_NAMES[cid]}',
                             fontsize=9, rotation=0, labelpad=55, va='center')

    save(fig, fname)


def draw_annotation(ax, true_label, top3):
    """
    Draw a styled annotation panel on ax showing:
      TRUE LABEL  (always blue)
      Predicted #1  (green if matches true, else red)
      Predicted #2  (gray, smaller)
      Predicted #3  (gray, smaller)
    """
    pred_label = top3[0][1]
    pred_conf  = top3[0][2]
    correct    = (pred_label == true_label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # True label box
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.70), 0.90, 0.24,
        boxstyle='round,pad=0.02',
        facecolor='#d6eaf8', edgecolor=COL_TRUE, linewidth=1.5))
    ax.text(0.50, 0.82, f'TRUE:  Subject {true_label}',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=COL_TRUE, transform=ax.transAxes)

    # Top-1 prediction box
    box_color   = '#d5f5e3' if correct else '#fadbd8'
    edge_color  = COL_CORRECT if correct else COL_WRONG
    result_text = 'CORRECT' if correct else 'WRONG'
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.40), 0.90, 0.27,
        boxstyle='round,pad=0.02',
        facecolor=box_color, edgecolor=edge_color, linewidth=2.0))
    ax.text(0.50, 0.60,
            f'PRED:  Subject {pred_label}',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=edge_color, transform=ax.transAxes)
    ax.text(0.50, 0.46,
            f'{result_text}  |  conf = {pred_conf:.1%}',
            ha='center', va='center', fontsize=8,
            color=edge_color, transform=ax.transAxes)

    # Top-2 and top-3 (smaller, gray)
    for k, (cidx, cname, cconf) in enumerate(top3[1:], start=2):
        y = 0.34 - (k - 2) * 0.17
        ax.text(0.50, y,
                f'#{k}: Subject {cname}  ({cconf:.1%})',
                ha='center', va='center', fontsize=7.5,
                color='#555555', transform=ax.transAxes)

    # Confidence bar (thin, at bottom)
    bar_w = pred_conf * 0.90
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.02), bar_w, 0.07,
        boxstyle='round,pad=0.01',
        facecolor=edge_color, alpha=0.7, edgecolor='none'))
    ax.text(0.05 + bar_w + 0.02, 0.055, f'{pred_conf:.0%}',
            va='center', fontsize=7.5, color='#333333',
            transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Summary figure: compact matrix  true -> predicted for all attacks
# ---------------------------------------------------------------------------

def fig_summary_matrix(all_preds, samples, save_fname):
    """
    Grid: rows = subjects, cols = (model × attack)
    Each cell = tiny image with colour-coded border + label text.
    """
    model_names  = list(all_preds.keys())
    attack_names = list(all_preds[model_names[0]].keys())
    n_subjects   = len(samples)
    n_cols       = len(model_names) * len(attack_names)

    fig, axes = plt.subplots(n_subjects, n_cols,
                              figsize=(2.4 * n_cols, 2.8 * n_subjects),
                              facecolor='white')
    fig.suptitle('Prediction Labels After Each Attack -- All Models x All Attacks\n'
                 'Green border = correct  |  Red border = wrong (model fooled)',
                 fontsize=13, fontweight='bold', y=1.01)

    if n_subjects == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    col_idx = 0
    for m_name in model_names:
        for a_name in attack_names:
            if n_subjects > 0:
                axes[0][col_idx].set_title(
                    f'{m_name}\n{a_name}', fontsize=8, fontweight='bold', pad=4)
            for row, (img, cid, lm) in enumerate(samples):
                pred_info = all_preds[m_name][a_name][row]  # (adv_img_np, pred_label, conf)
                adv_np, pred_label, conf = pred_info
                correct = (pred_label == CLASS_NAMES[cid])

                ax = axes[row][col_idx]
                ax.imshow(adv_np)
                ax.axis('off')

                # Coloured border
                border_c = COL_CORRECT if correct else COL_WRONG
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_linewidth(3.5)
                    sp.set_edgecolor(border_c)

                # Label overlay
                label_txt = f'T:{CLASS_NAMES[cid]}\nP:{pred_label}\n{conf:.0%}'
                ax.text(0.02, 0.02, label_txt,
                        transform=ax.transAxes, fontsize=6.5,
                        va='bottom', ha='left',
                        color='white', fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.55,
                                  boxstyle='round,pad=0.2', edgecolor='none'))

            col_idx += 1

    # Row labels
    for row, (_, cid, _) in enumerate(samples):
        axes[row][0].set_ylabel(f'Subject\n{CLASS_NAMES[cid]}',
                                fontsize=8, rotation=0, labelpad=50, va='center')

    plt.tight_layout()
    save(fig, save_fname)


# ---------------------------------------------------------------------------
# Confusion shift: where do wrong predictions go?
# ---------------------------------------------------------------------------

def fig_confusion_shift(all_preds, samples, save_fname):
    """
    For each (model, attack) combination: bar chart showing which wrong
    classes the adversarial examples get classified as.
    Reveals whether a single subject is a common 'attractor' for fooled predictions.
    """
    model_names  = list(all_preds.keys())
    attack_names = list(next(iter(all_preds.values())).keys())
    n_models     = len(model_names)
    n_attacks    = len(attack_names)

    fig, axes = plt.subplots(n_models, n_attacks,
                              figsize=(5 * n_attacks, 4 * n_models),
                              facecolor='white')
    if n_models == 1:
        axes = [axes]
    if n_attacks == 1:
        axes = [[ax] for ax in axes]

    for mi, m_name in enumerate(model_names):
        for ai, a_name in enumerate(attack_names):
            ax = axes[mi][ai]
            wrong_preds = {}
            n_correct = 0
            for row, (_, cid, _) in enumerate(samples):
                _, pred_label, _ = all_preds[m_name][a_name][row]
                true_label = CLASS_NAMES[cid]
                if pred_label == true_label:
                    n_correct += 1
                else:
                    wrong_preds[pred_label] = wrong_preds.get(pred_label, 0) + 1

            if wrong_preds:
                labels = list(wrong_preds.keys())
                counts = [wrong_preds[l] for l in labels]
                bar_colors = plt.cm.Reds(np.linspace(0.5, 0.9, len(labels)))
                ax.bar(labels, counts, color=bar_colors, edgecolor='white', linewidth=0.5)
                ax.set_ylabel('# images fooled into this class', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'All correct!\n(no misclassifications)',
                        ha='center', va='center', fontsize=11,
                        color=COL_CORRECT, transform=ax.transAxes, fontweight='bold')

            ax.set_title(f'{m_name} | {a_name}\n'
                         f'({n_correct}/{len(samples)} correct after attack)',
                         fontsize=9, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            if mi == 0 and ai == 0:
                ax.set_xlabel('Predicted Subject (wrong)', fontsize=9)

    fig.suptitle('Confusion Shift: Where Do Fooled Predictions Land?\n'
                 '(Which subjects does the model confuse each victim with?)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, save_fname)


# ===========================================================================
# Main
# ===========================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Output folder: {OUT_DIR}\n')

    print('Loading models...')
    models = load_all_models(device)

    test_loader  = get_dataloader(split='test',  include='genuine',  batch_size=32)
    imp_loader   = get_dataloader(split='test',  include='impostor', batch_size=32)
    train_loader = get_dataloader(split='train', include='genuine',
                                   batch_size=16, shuffle=True)

    samples = collect_samples(test_loader, N_SUBJECTS)
    print(f'Subjects selected: {[CLASS_NAMES[s[1]] for s in samples]}\n')

    # -- Build impostor map for physical attack --------------------------------
    needed = {s[1] for s in samples}
    imp_map = {}
    for imgs, cids, _, _ in imp_loader:
        for i in range(len(imgs)):
            cid = cids[i].item()
            if cid in needed and cid not in imp_map:
                imp_map[cid] = imgs[i]
        if len(imp_map) >= len(needed):
            break

    # -- Train a quick patch (shared across all models for speed) -------------
    print('Training eye-region patch (MobileNetV3, 300 steps)...')
    primary_model = models['MobileNetV3']
    att_eye = AdversarialPatch(patch_size=(40, 100), device=device)
    att_eye.optimize(primary_model, train_loader,
                     num_iterations=300, lr=0.02, mode='landmark', log_every=100)

    # Each model gets its own patch for accurate predictions
    patches = {}
    for name, model in models.items():
        print(f'Training patch for {name} (300 steps)...')
        att = AdversarialPatch(patch_size=(40, 100), device=device)
        att.optimize(model, train_loader,
                     num_iterations=300, lr=0.02, mode='landmark', log_every=100)
        patches[name] = att

    # -- Define attack functions per model ------------------------------------
    def make_attacks(model, device, att):
        def physical(img_d, cid, lm):
            return imp_map.get(cid, img_d).to(device)

        def fgsm(img_d, cid, lm):
            adv, _ = fgsm_attack(model, img_d.unsqueeze(0),
                                  torch.tensor([cid], device=device), FGSM_EPS, device)
            return adv.squeeze(0)

        def pgd(img_d, cid, lm):
            adv = pgd_attack(model, img_d.unsqueeze(0),
                              torch.tensor([cid], device=device),
                              PGD_EPS, PGD_ALPHA, PGD_STEPS, device)
            return adv.squeeze(0)

        def patch(img_d, cid, lm):
            bbox = eye_bbox_from_landmarks([lm], padding=EYE_PAD)[0]
            adv  = apply_patch_landmark(img_d.unsqueeze(0),
                                         att.patch.detach(), [bbox])
            return adv.squeeze(0)

        return {'Physical': physical, 'FGSM (8/255)': fgsm,
                'PGD-40 (8/255)': pgd, 'Eye Patch': patch}

    # -- Per-model per-attack prediction figures ------------------------------
    all_preds = {}   # {model_name: {attack_name: [(adv_np, pred_label, conf), ...]}}

    for name, model in models.items():
        print(f'\n=== {name} ===')
        model.eval()
        attacks     = make_attacks(model, device, patches[name])
        all_preds[name] = {}

        for a_name, attack_fn in attacks.items():
            print(f'  {a_name}...')
            all_preds[name][a_name] = []

            fig_attack_predictions(
                attack_name=a_name,
                attacked_imgs_fn=attack_fn,
                samples=samples,
                model=model,
                model_name=name,
                device=device,
                fname=f'pred_{a_name.lower().replace(" ", "_").replace("/", "").replace("-", "")}_{name.lower()}.png'
            )

            # Collect for summary figures
            for img, cid, lm in samples:
                img_d = img.to(device)
                adv   = attack_fn(img_d, cid, lm)
                top3  = get_top3(model, adv.unsqueeze(0), device)
                all_preds[name][a_name].append(
                    (to_np(adv), top3[0][1], top3[0][2])
                )

    # -- Summary matrix -------------------------------------------------------
    print('\nGenerating summary matrix...')
    fig_summary_matrix(all_preds, samples, 'pred_05_all_attacks_summary.png')

    # -- Confusion shift ------------------------------------------------------
    print('Generating confusion shift chart...')
    fig_confusion_shift(all_preds, samples, 'pred_06_confusion_shift.png')

    # -- Final listing --------------------------------------------------------
    print(f'\nAll files saved to: {OUT_DIR}\n')
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f'  {f:<65} {size//1024:>5} KB')


if __name__ == '__main__':
    main()
