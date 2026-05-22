"""
Gradient-weighted Class Activation Mapping (Grad-CAM) -- Selvaraju et al., 2017.

Generates a heatmap that highlights which spatial regions most influenced
the model's prediction.  Used in this thesis for XAI (Section 6 / Chapter 7)
to visualise how adversarial attacks shift the model's attention away from
identity-relevant facial features.

Supports all three architectures used in this study:
  MobileNetV3-Small  -> target layer: features[-1]   (the last convolutional block)
  GhostNet-100       -> target layer: blocks[-1]      (last ghost bottleneck)
  EfficientNetV2-S   -> target layer: features[-1]   (last MBConv block)
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


# -----------------------------------------------------------------------------
# Grad-CAM core
# -----------------------------------------------------------------------------

class GradCAM:
    """
    Hook-based Grad-CAM.  Attach to a convolutional layer and call generate()
    to obtain a saliency map for a single input image.
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.activations  = None
        self.gradients    = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_idx=None):
        """
        Compute Grad-CAM heatmap for one image.

        image_tensor : FloatTensor (1, C, H, W) -- ImageNet-normalised
        class_idx    : int or None; if None uses the top predicted class

        Returns:
            cam : np.ndarray (H, W), values in [0, 1]
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global-average-pool the gradients over spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam     = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=image_tensor.shape[-2:],
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# -----------------------------------------------------------------------------
# Architecture-specific target layer selection
# -----------------------------------------------------------------------------

def get_target_layer(model, model_name):
    """
    Return the last convolutional block for each architecture.

    model_name : one of 'MobileNetV3', 'GhostNet', 'EfficientNet'
    """
    name = model_name.lower()
    if 'mobilenet' in name:
        # MobileNetV3-Small: model.features is a Sequential of InvertedResidual blocks
        return model.features[-1]
    elif 'ghost' in name:
        # GhostNet (timm): model.blocks is a Sequential of GhostBottleneck blocks
        return model.blocks[-1]
    elif 'efficient' in name:
        # EfficientNetV2-S: model.features is a Sequential of MBConv blocks
        return model.features[-1]
    else:
        raise ValueError(f"Unknown model name: '{model_name}'. "
                         "Expected 'MobileNetV3', 'GhostNet', or 'EfficientNet'.")


def build_gradcam(model, model_name):
    """Convenience factory: returns a GradCAM instance for the given model."""
    target_layer = get_target_layer(model, model_name)
    return GradCAM(model, target_layer)


# -----------------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------------

def overlay_heatmap(image_np, cam, alpha=0.45, colormap='jet'):
    """
    Overlay a Grad-CAM heatmap on a uint8 RGB image.

    image_np : np.ndarray (H, W, 3), dtype uint8
    cam      : np.ndarray (H, W), values in [0, 1]
    alpha    : blending weight for the heatmap
    Returns  : np.ndarray (H, W, 3), dtype uint8
    """
    cmap    = cm.get_cmap(colormap)
    heatmap = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)   # drop alpha channel
    blended = ((1 - alpha) * image_np + alpha * heatmap).astype(np.uint8)
    return blended


def plot_gradcam_comparison(model, model_name, device, samples,
                            fgsm_fn=None, pgd_fn=None,
                            save_path=None):
    """
    Plot a grid of rows (one per sample):
      [clean image | Grad-CAM clean | FGSM image | Grad-CAM FGSM | PGD image | Grad-CAM PGD]

    model    : trained PyTorch model
    samples  : list of (image_tensor, class_idx) pairs
                 image_tensor : (1, C, H, W) FloatTensor, normalised
    fgsm_fn  : callable(images, labels) -> adv_images  (or None to skip column)
    pgd_fn   : callable(images, labels) -> adv_images  (or None to skip column)
    save_path: file path to save the figure (PNG); if None, plt.show() is called
    """
    from utils import denormalize

    gradcam = build_gradcam(model, model_name)

    n_cols   = 2 + (2 if fgsm_fn else 0) + (2 if pgd_fn else 0)
    col_titles = ['Clean Image', 'Grad-CAM (Clean)']
    if fgsm_fn:
        col_titles += ['FGSM Image', 'Grad-CAM (FGSM)']
    if pgd_fn:
        col_titles += ['PGD Image', 'Grad-CAM (PGD)']

    fig, axes = plt.subplots(len(samples), n_cols,
                             figsize=(3.5 * n_cols, 3.5 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    for row_idx, (img_tensor, class_idx) in enumerate(samples):
        img_tensor = img_tensor.to(device)
        img_np     = (denormalize(img_tensor.squeeze(0)).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        col = 0

        # -- Clean image + Grad-CAM ------------------------------------------
        axes[row_idx][col].imshow(img_np);  axes[row_idx][col].axis('off')
        if row_idx == 0:
            axes[row_idx][col].set_title(col_titles[col], fontsize=11, fontweight='bold')
        col += 1

        cam_clean = gradcam.generate(img_tensor.detach().clone().requires_grad_(True), class_idx)
        overlay   = overlay_heatmap(img_np.copy(), cam_clean)
        axes[row_idx][col].imshow(overlay); axes[row_idx][col].axis('off')
        if row_idx == 0:
            axes[row_idx][col].set_title(col_titles[col], fontsize=11, fontweight='bold')
        col += 1

        # -- FGSM image + Grad-CAM -------------------------------------------
        if fgsm_fn:
            label_t      = torch.tensor([class_idx], device=device)
            adv_fgsm     = fgsm_fn(img_tensor, label_t)
            adv_fgsm_np  = (denormalize(adv_fgsm.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            axes[row_idx][col].imshow(adv_fgsm_np); axes[row_idx][col].axis('off')
            if row_idx == 0:
                axes[row_idx][col].set_title(col_titles[col], fontsize=11, fontweight='bold')
            col += 1

            cam_fgsm = gradcam.generate(adv_fgsm.requires_grad_(True), class_idx)
            axes[row_idx][col].imshow(overlay_heatmap(adv_fgsm_np.copy(), cam_fgsm))
            axes[row_idx][col].axis('off')
            if row_idx == 0:
                axes[row_idx][col].set_title(col_titles[col], fontsize=11, fontweight='bold')
            col += 1

        # -- PGD image + Grad-CAM --------------------------------------------
        if pgd_fn:
            label_t    = torch.tensor([class_idx], device=device)
            adv_pgd    = pgd_fn(img_tensor, label_t)
            adv_pgd_np = (denormalize(adv_pgd.squeeze(0).cpu()).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            axes[row_idx][col].imshow(adv_pgd_np); axes[row_idx][col].axis('off')
            if row_idx == 0:
                axes[row_idx][col].set_title(col_titles[col], fontsize=11, fontweight='bold')
            col += 1

            cam_pgd = gradcam.generate(adv_pgd.requires_grad_(True), class_idx)
            axes[row_idx][col].imshow(overlay_heatmap(adv_pgd_np.copy(), cam_pgd))
            axes[row_idx][col].axis('off')
            if row_idx == 0:
                axes[row_idx][col].set_title(col_titles[col], fontsize=11, fontweight='bold')

    plt.suptitle(f'Grad-CAM Feature Attribution -- {model_name}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'  Saved Grad-CAM figure -> {save_path}')
        plt.close()
    else:
        plt.show()

    gradcam.remove_hooks()
