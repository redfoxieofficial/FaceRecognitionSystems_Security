"""
Fast Gradient Sign Method (FGSM) -- Goodfellow et al., 2014.

Generates adversarial examples by adding a single-step perturbation in the
direction of the loss gradient w.r.t. the input:

    x_adv = x + eps * sign(grad_x L(f(x), y))

For face recognition, a successful attack causes the model to misidentify
the subject's identity (wrong class prediction).
"""

import torch
import torch.nn.functional as F


# Per-channel valid range after ImageNet normalisation (~= +/-3 SD)
_CLAMP_MIN, _CLAMP_MAX = -3.0, 3.0


def fgsm_attack(model, images, labels, epsilon, device):
    """
    Apply FGSM to a batch of images.

    Args:
        model   : PyTorch model in eval mode (outputs class logits)
        images  : FloatTensor (N, C, H, W) -- ImageNet-normalised
        labels  : LongTensor (N,) -- true class indices
        epsilon : perturbation magnitude in normalised pixel space
        device  : torch.device

    Returns:
        adv_images : FloatTensor (N, C, H, W), same shape as images
        loss_value : scalar float -- CE loss on clean inputs (for logging)
    """
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    images.requires_grad_(True)

    logits = model(images)
    loss   = F.cross_entropy(logits, labels)

    model.zero_grad()
    loss.backward()

    # Perturbation: eps * sign(grad_x L)
    perturbation = epsilon * images.grad.detach().sign()
    adv_images   = torch.clamp(images.detach() + perturbation, _CLAMP_MIN, _CLAMP_MAX)

    return adv_images, loss.item()


@torch.no_grad()
def _predict(model, images, device):
    model.eval()
    logits = model(images.to(device))
    probs  = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred.cpu(), conf.cpu()


def fgsm_evaluate(model, dataloader, epsilon, device):
    """
    Evaluate one model under FGSM at a single epsilon value.

    Returns:
        true_labels       : np.ndarray (N,)
        clean_preds       : np.ndarray (N,)
        adv_preds         : np.ndarray (N,)
        clean_confidences : np.ndarray (N,)
        adv_confidences   : np.ndarray (N,)
    """
    import numpy as np

    true_labels_list       = []
    clean_preds_list       = []
    adv_preds_list         = []
    clean_confidences_list = []
    adv_confidences_list   = []

    for images, class_ids, _, _ in dataloader:
        # Clean predictions
        clean_pred, clean_conf = _predict(model, images, device)

        # Adversarial predictions
        adv_images, _ = fgsm_attack(model, images, class_ids, epsilon, device)
        adv_pred, adv_conf = _predict(model, adv_images, device)

        true_labels_list.extend(class_ids.numpy().tolist())
        clean_preds_list.extend(clean_pred.numpy().tolist())
        adv_preds_list.extend(adv_pred.numpy().tolist())
        clean_confidences_list.extend(clean_conf.numpy().tolist())
        adv_confidences_list.extend(adv_conf.numpy().tolist())

    return (
        np.array(true_labels_list),
        np.array(clean_preds_list),
        np.array(adv_preds_list),
        np.array(clean_confidences_list),
        np.array(adv_confidences_list),
    )


def fgsm_sweep(model, dataloader, epsilons, device):
    """
    Run FGSM evaluation across multiple epsilon values.

    Returns:
        results : dict  {epsilon -> metrics_dict}
    """
    from utils import compute_attack_metrics

    sweep_results = {}
    for eps in epsilons:
        true_l, clean_p, adv_p, clean_c, adv_c = fgsm_evaluate(
            model, dataloader, eps, device
        )
        sweep_results[eps] = compute_attack_metrics(true_l, clean_p, adv_p, clean_c, adv_c)
        print(f'  eps={eps:.3f}  clean={sweep_results[eps]["clean_accuracy"]:.1f}%  '
              f'attacked={sweep_results[eps]["attacked_accuracy"]:.1f}%  '
              f'ASR={sweep_results[eps]["attack_success_rate"]:.1f}%')

    return sweep_results
