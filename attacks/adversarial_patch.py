"""
Adversarial Patch Attack -- Brown et al., 2017; inspired by AdvHat (Komkov & Petiushko, 2021).

A localized patch is optimised (via Adam) to maximise the model's cross-entropy loss
when placed over a target facial region.  The patch is then frozen and applied to all
test images to measure the accuracy drop.

Two placement strategies are supported:
  landmark  -- uses per-image eye coordinates from the NUAA text files, so the patch
               sits precisely over the actual eye region of each subject.
  fixed     -- places the patch at a fixed bounding box derived from the dataset-wide
               mean eye location (faster; useful when landmark info is unavailable).
"""

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import os

_CLAMP_MIN, _CLAMP_MAX = -3.0, 3.0


# ---------------------------------------------------------------------------------
# Landmark -> bounding box helpers
# ---------------------------------------------------------------------------------

def eye_bbox_from_landmarks(landmarks_batch, image_size=224, padding=10):
    """
    Convert per-image landmark dicts to (y0, y1, x0, x1) bounding boxes
    that span the eye region with some padding.

    landmarks_batch : list of dicts with 'left_eye' and 'right_eye' keys
                      (coordinates already scaled to image_size x image_size)
    Returns:
        list of (y0, y1, x0, x1) tuples -- one per image in the batch
    """
    bboxes = []
    for lm in landmarks_batch:
        lx, ly = lm['left_eye']
        rx, ry = lm['right_eye']

        x0 = max(0,          int(min(lx, rx)) - padding)
        x1 = min(image_size, int(max(lx, rx)) + padding)
        y0 = max(0,          int(min(ly, ry)) - padding)
        y1 = min(image_size, int(max(ly, ry)) + padding)

        if (x1 - x0) < 10:
            x0, x1 = max(0, x0 - 5), min(image_size, x1 + 5)
        if (y1 - y0) < 10:
            y0, y1 = max(0, y0 - 5), min(image_size, y1 + 5)

        bboxes.append((y0, y1, x0, x1))
    return bboxes


def mean_eye_bbox(dataset, image_size=224, padding=10):
    """
    Compute the average eye bounding box across the full dataset.
    Used for fixed-placement attacks.
    Returns: (y0, y1, x0, x1)
    """
    all_x0, all_x1, all_y0, all_y1 = [], [], [], []
    for s in dataset.samples:
        lx, ly = s['left_eye']
        rx, ry = s['right_eye']
        all_x0.append(min(lx, rx))
        all_x1.append(max(lx, rx))
        all_y0.append(min(ly, ry))
        all_y1.append(max(ly, ry))

    sample_img_path = dataset.samples[0]['path']
    from PIL import Image as PILImage
    w, h = PILImage.open(sample_img_path).size
    sx, sy = image_size / w, image_size / h

    x0 = max(0,          int(np.mean(all_x0) * sx) - padding)
    x1 = min(image_size, int(np.mean(all_x1) * sx) + padding)
    y0 = max(0,          int(np.mean(all_y0) * sy) - padding)
    y1 = min(image_size, int(np.mean(all_y1) * sy) + padding)
    return (y0, y1, x0, x1)


# ---------------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------------

def apply_patch_fixed(images, patch, bbox):
    """
    Paste a single patch tensor onto all images at the same bounding box.

    images : FloatTensor (N, C, H, W)
    patch  : FloatTensor (C, ph, pw)
    bbox   : (y0, y1, x0, x1)
    """
    y0, y1, x0, x1 = bbox
    ph, pw = y1 - y0, x1 - x0

    patch_resized = F.interpolate(
        patch.unsqueeze(0), size=(ph, pw), mode='bilinear', align_corners=False
    ).squeeze(0)
    patch_resized = torch.clamp(patch_resized, _CLAMP_MIN, _CLAMP_MAX)

    patched = images.clone()
    patched[:, :, y0:y1, x0:x1] = patch_resized.unsqueeze(0)
    return patched


def apply_patch_landmark(images, patch, bboxes):
    """
    Paste the patch onto each image at its individual landmark-derived bbox.

    images : FloatTensor (N, C, H, W)
    patch  : FloatTensor (C, patch_h, patch_w)
    bboxes : list of (y0, y1, x0, x1) -- one per image
    """
    patched = images.clone()
    for i, (y0, y1, x0, x1) in enumerate(bboxes):
        ph, pw = y1 - y0, x1 - x0
        if ph <= 0 or pw <= 0:
            continue
        resized = F.interpolate(
            patch.unsqueeze(0), size=(ph, pw), mode='bilinear', align_corners=False
        ).squeeze(0)
        patched[i, :, y0:y1, x0:x1] = torch.clamp(resized, _CLAMP_MIN, _CLAMP_MAX)
    return patched


# ---------------------------------------------------------------------------------
# Patch optimisation
# ---------------------------------------------------------------------------------

class AdversarialPatch:
    """
    Optimise a universal adversarial patch for a given model.

    The patch is trained to maximise the cross-entropy loss when applied
    over the eye region, causing the model to misidentify subjects.
    """

    def __init__(self, patch_size=(60, 120), device='cpu'):
        self.patch_size   = patch_size
        self.device       = device
        self._fixed_bbox  = None
        self.loss_history = []   # CE loss per optimisation step (for training curves)

        ph, pw = patch_size
        self.patch = torch.empty(3, ph, pw, device=device).uniform_(_CLAMP_MIN, _CLAMP_MAX)
        self.patch.requires_grad_(True)

    # -- training ---------------------------------------------------------------

    def optimize(self, model, dataloader, num_iterations=500,
                 lr=0.01, mode='landmark', log_every=100):
        """
        Optimise the patch on the training split.

        model         : PyTorch model (eval mode enforced internally)
        dataloader    : must yield (images, class_ids, liveness, landmarks)
        num_iterations: total gradient steps
        lr            : Adam learning rate
        mode          : 'landmark' or 'fixed'
        """
        model.eval()
        optimizer = torch.optim.Adam([self.patch], lr=lr)
        self.loss_history = []

        data_iter = iter(dataloader)
        step = 0

        while step < num_iterations:
            try:
                images, class_ids, _, landmarks = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                images, class_ids, _, landmarks = next(data_iter)

            images    = images.to(self.device)
            class_ids = class_ids.to(self.device)

            if mode == 'landmark':
                bboxes  = eye_bbox_from_landmarks(landmarks)
                patched = apply_patch_landmark(images, self.patch, bboxes)
            else:
                if self._fixed_bbox is None:
                    raise RuntimeError("Call set_fixed_bbox() before optimize() with mode='fixed'.")
                patched = apply_patch_fixed(images, self.patch, self._fixed_bbox)

            logits = model(patched)
            loss   = -F.cross_entropy(logits, class_ids)   # maximise loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.patch.clamp_(_CLAMP_MIN, _CLAMP_MAX)

            ce_loss = -loss.item()   # positive CE value
            self.loss_history.append(ce_loss)

            step += 1
            if step % log_every == 0:
                print(f'  [Patch] step {step:>4}/{num_iterations}  loss={ce_loss:.4f}')

    def set_fixed_bbox(self, bbox):
        self._fixed_bbox = bbox

    # -- evaluation -------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, model, dataloader, mode='landmark'):
        """
        Apply the optimised patch to the test set and collect predictions.

        Returns:
            true_labels, clean_preds, adv_preds, clean_confidences, adv_confidences
        """
        model.eval()
        true_l, clean_p, adv_p, clean_c, adv_c = [], [], [], [], []

        for images, class_ids, _, landmarks in dataloader:
            images = images.to(self.device)

            logits_clean = model(images)
            probs_clean  = torch.softmax(logits_clean, dim=1)
            conf_clean, pred_clean = probs_clean.max(dim=1)

            if mode == 'landmark':
                bboxes  = eye_bbox_from_landmarks(landmarks)
                patched = apply_patch_landmark(images, self.patch.detach(), bboxes)
            else:
                patched = apply_patch_fixed(images, self.patch.detach(), self._fixed_bbox)

            logits_adv = model(patched)
            probs_adv  = torch.softmax(logits_adv, dim=1)
            conf_adv, pred_adv = probs_adv.max(dim=1)

            true_l.extend(class_ids.numpy().tolist())
            clean_p.extend(pred_clean.cpu().numpy().tolist())
            adv_p.extend(pred_adv.cpu().numpy().tolist())
            clean_c.extend(conf_clean.cpu().numpy().tolist())
            adv_c.extend(conf_adv.cpu().numpy().tolist())

        return (
            np.array(true_l), np.array(clean_p), np.array(adv_p),
            np.array(clean_c), np.array(adv_c),
        )

    # -- persistence ------------------------------------------------------------

    def save(self, path):
        torch.save({'patch': self.patch.detach().cpu(),
                    'patch_size': self.patch_size,
                    'fixed_bbox': self._fixed_bbox,
                    'loss_history': self.loss_history}, path)
        print(f'  Patch saved -> {path}')

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.patch        = state['patch'].to(self.device).requires_grad_(True)
        self.patch_size   = state['patch_size']
        self._fixed_bbox  = state.get('fixed_bbox')
        self.loss_history = state.get('loss_history', [])
        print(f'  Patch loaded <- {path}')

    def save_image(self, path):
        """Save a human-readable PNG of the patch (denormalised to [0,1])."""
        from utils import denormalize
        patch_vis = denormalize(self.patch.detach().cpu())
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        vutils.save_image(patch_vis, path)
        print(f'  Patch image saved -> {path}')
