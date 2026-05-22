"""
Projected Gradient Descent (PGD) -- Madry et al., 2018.

A multi-step iterative attack that is strictly stronger than FGSM.
Each step is a scaled gradient-sign update, and the result is projected
back into the eps-ball around the original input (L-inf norm):

    x^0   = x + Uniform(-eps, eps)          # optional random start
    x^t+1 = Proj_{||delta||_inf <= eps} [ x^t + alpha * sign(grad_{x^t} L) ]

For face recognition, the goal is to make the model misidentify the subject.
"""

import torch
import torch.nn.functional as F

_CLAMP_MIN, _CLAMP_MAX = -3.0, 3.0


def pgd_attack(model, images, labels, epsilon, alpha, num_steps, device,
               random_start=True):
    """
    Apply PGD to a batch of images.

    Args:
        model       : PyTorch model in eval mode
        images      : FloatTensor (N, C, H, W) -- ImageNet-normalised
        labels      : LongTensor (N,) -- true class indices
        epsilon     : L-inf ball radius (perturbation budget)
        alpha       : step size per iteration
        num_steps   : number of gradient steps
        device      : torch.device
        random_start: if True, initialise with random noise inside eps-ball

    Returns:
        adv_images : FloatTensor (N, C, H, W)
    """
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    if random_start:
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(images)

    for _ in range(num_steps):
        delta = delta.detach().requires_grad_(True)

        logits = model(images + delta)
        loss   = F.cross_entropy(logits, labels)
        loss.backward()

        with torch.no_grad():
            # Gradient-sign step
            delta = delta + alpha * delta.grad.sign()
            # Project back onto the eps-ball (L-inf)
            delta = torch.clamp(delta, -epsilon, epsilon)

    adv_images = torch.clamp(images + delta.detach(), _CLAMP_MIN, _CLAMP_MAX)
    return adv_images.detach()


@torch.no_grad()
def _predict(model, images, device):
    model.eval()
    logits = model(images.to(device))
    probs  = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred.cpu(), conf.cpu()


def pgd_evaluate(model, dataloader, epsilon, alpha, num_steps, device,
                 random_start=True):
    """
    Evaluate one model under PGD.

    Returns:
        true_labels, clean_preds, adv_preds, clean_confidences, adv_confidences
        -- all np.ndarray of shape (N,)
    """
    import numpy as np

    true_l, clean_p, adv_p, clean_c, adv_c = [], [], [], [], []

    for images, class_ids, _, _ in dataloader:
        clean_pred, clean_conf = _predict(model, images, device)

        adv_images = pgd_attack(
            model, images, class_ids,
            epsilon, alpha, num_steps, device, random_start,
        )
        adv_pred, adv_conf = _predict(model, adv_images, device)

        true_l.extend(class_ids.numpy().tolist())
        clean_p.extend(clean_pred.numpy().tolist())
        adv_p.extend(adv_pred.numpy().tolist())
        clean_c.extend(clean_conf.numpy().tolist())
        adv_c.extend(adv_conf.numpy().tolist())

    return (
        np.array(true_l), np.array(clean_p), np.array(adv_p),
        np.array(clean_c), np.array(adv_c),
    )


def pgd_sweep(model, dataloader, epsilon_list, alpha, num_steps, device):
    """
    Run PGD evaluation across multiple epsilon values (fixed steps/alpha).

    Returns:
        results : dict  {epsilon -> metrics_dict}
    """
    from utils import compute_attack_metrics

    sweep_results = {}
    for eps in epsilon_list:
        true_l, clean_p, adv_p, clean_c, adv_c = pgd_evaluate(
            model, dataloader, eps, alpha, num_steps, device
        )
        sweep_results[eps] = compute_attack_metrics(true_l, clean_p, adv_p, clean_c, adv_c)
        print(f'  eps={eps:.3f}  clean={sweep_results[eps]["clean_accuracy"]:.1f}%  '
              f'attacked={sweep_results[eps]["attacked_accuracy"]:.1f}%  '
              f'ASR={sweep_results[eps]["attack_success_rate"]:.1f}%')

    return sweep_results
