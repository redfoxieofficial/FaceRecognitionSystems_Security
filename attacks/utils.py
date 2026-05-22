"""
Shared utilities for adversarial attack evaluation.

Loads all three trained models (MobileNetV3-Small, GhostNet-100, EfficientNetV2-Small)
and the NUAA dataset (ClientFace genuine + ImposterFace printed-photo attacks),
using the official split text files which also carry per-image landmark coordinates.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -----------------------------------------------------------------------------
# Paths (relative to this file's location: code/attacks/)
# -----------------------------------------------------------------------------
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_PATHS = {
    'mobilenetv3':  os.path.join(BASE_DIR, 'code', 'MobileNetV3',  'mobilenet_v3_face_recognition.pth'),
    'ghostnet':     os.path.join(BASE_DIR, 'code', 'GhostNet',     'ghostnet_face_recognition.pth'),
    'efficientnet': os.path.join(BASE_DIR, 'code', 'EfficientNetV2', 'efficientnet_v2_face_recognition.pth'),
}
CLIENT_FACE_DIR   = os.path.join(DATASET_DIR, 'ClientFace')
IMPOSTER_FACE_DIR = os.path.join(DATASET_DIR, 'ImposterFace')

# Build CLASS_NAMES from whatever subject folders actually exist in ClientFace
# (the models were trained on exactly these folders via ImageFolder).
CLASS_NAMES = sorted([
    d for d in os.listdir(CLIENT_FACE_DIR)
    if os.path.isdir(os.path.join(CLIENT_FACE_DIR, d))
])
NUM_CLASSES = len(CLASS_NAMES)

# -----------------------------------------------------------------------------
# Shared image transform (same as test transforms used during training)
# -----------------------------------------------------------------------------
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(tensor):
    """Reverse ImageNet normalization for visualization. Returns [0, 1] tensor."""
    return torch.clamp(tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Model loaders -- mirror the exact classifier heads from the training notebooks
# -----------------------------------------------------------------------------

def load_mobilenetv3(device='cpu'):
    model = tv_models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATHS['mobilenetv3'], map_location=device))
    return model.to(device).eval()


def load_ghostnet(device='cpu'):
    model = timm.create_model('ghostnet_100', pretrained=False)
    in_features = model.classifier.in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, NUM_CLASSES),
    )
    model.load_state_dict(torch.load(MODEL_PATHS['ghostnet'], map_location=device))
    return model.to(device).eval()


def load_efficientnet(device='cpu'):
    model = tv_models.efficientnet_v2_s(weights=None)
    model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATHS['efficientnet'], map_location=device))
    return model.to(device).eval()


def load_all_models(device='cpu'):
    """Return dict {name: model} for all three architectures."""
    return {
        'MobileNetV3': load_mobilenetv3(device),
        'GhostNet':    load_ghostnet(device),
        'EfficientNet': load_efficientnet(device),
    }


# -----------------------------------------------------------------------------
# NUAA dataset with landmark coordinates
# -----------------------------------------------------------------------------

def _parse_split_file(txt_path, base_dir, label):
    """
    Parse one of the four NUAA split text files.

    Each line:  relative\\path.jpg  lex ley  rex rey  nx ny
    Returns list of dicts with keys: path, label, left_eye, right_eye, nose
    """
    samples = []
    with open(txt_path, 'r') as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            rel_path   = parts[0].replace('\\', os.sep)
            full_path  = os.path.join(base_dir, rel_path)
            if not os.path.isfile(full_path):
                continue
            left_eye  = (float(parts[1]), float(parts[2]))   # (x, y)
            right_eye = (float(parts[3]), float(parts[4]))
            nose      = (float(parts[5]), float(parts[6]))
            # Subject id is the first component of the relative path
            subject = rel_path.split(os.sep)[0]   # e.g. '0001'
            if subject not in CLASS_NAMES:
                continue   # skip subjects the model was never trained on (e.g. '0016')
            class_idx = CLASS_NAMES.index(subject)
            samples.append({
                'path':      full_path,
                'label':     label,           # 1 = genuine, 0 = impostor
                'class_idx': class_idx,       # identity (0-14)
                'left_eye':  left_eye,
                'right_eye': right_eye,
                'nose':      nose,
            })
    return samples


class NUAADataset(Dataset):
    """
    Unified NUAA dataset for attack evaluation.

    split     : 'test' uses the official *_test_face.txt files
                'train' uses the official *_train_face.txt files
    include   : 'both'     -- genuine + impostor
                'genuine'  -- ClientFace only  (label = 1)
                'impostor' -- ImposterFace only (label = 0)
    transform : torchvision transform applied to PIL images
    """

    def __init__(self, split='test', include='both', transform=None):
        assert split   in ('train', 'test')
        assert include in ('both', 'genuine', 'impostor')
        self.transform = transform or TEST_TRANSFORM
        self.samples   = []

        if include in ('both', 'genuine'):
            txt = os.path.join(DATASET_DIR, f'client_{split}_face.txt')
            self.samples += _parse_split_file(txt, CLIENT_FACE_DIR, label=1)

        if include in ('both', 'impostor'):
            txt = os.path.join(DATASET_DIR, f'imposter_{split}_face.txt')
            self.samples += _parse_split_file(txt, IMPOSTER_FACE_DIR, label=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        image = Image.open(s['path']).convert('RGB')

        # Scale landmark coordinates to 224x224 for downstream patch placement
        orig_w, orig_h = image.size
        scale_x = 224.0 / orig_w
        scale_y = 224.0 / orig_h
        landmarks = {
            'left_eye':  (s['left_eye'][0]  * scale_x, s['left_eye'][1]  * scale_y),
            'right_eye': (s['right_eye'][0] * scale_x, s['right_eye'][1] * scale_y),
            'nose':      (s['nose'][0]      * scale_x, s['nose'][1]      * scale_y),
        }

        if self.transform:
            image = self.transform(image)

        return image, s['class_idx'], s['label'], landmarks


def collate_with_landmarks(batch):
    """Custom collate that keeps landmarks as a list of dicts (not tensored)."""
    images    = torch.stack([b[0] for b in batch])
    class_ids = torch.tensor([b[1] for b in batch], dtype=torch.long)
    liveness  = torch.tensor([b[2] for b in batch], dtype=torch.long)
    landmarks = [b[3] for b in batch]
    return images, class_ids, liveness, landmarks


def get_dataloader(split='test', include='both', batch_size=32, shuffle=False):
    ds = NUAADataset(split=split, include=include)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_with_landmarks, num_workers=0)


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

@torch.no_grad()
def get_predictions(model, dataloader, device):
    """
    Run model on dataloader, return arrays of true labels and predictions.

    Returns:
        true_labels : np.ndarray of class indices
        pred_labels : np.ndarray of predicted class indices
        confidences : np.ndarray of max-softmax scores
    """
    model.eval()
    true_labels, pred_labels, confidences = [], [], []

    for images, class_ids, _, _ in dataloader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        true_labels.extend(class_ids.numpy().tolist())
        pred_labels.extend(pred.cpu().numpy().tolist())
        confidences.extend(conf.cpu().numpy().tolist())

    return (np.array(true_labels),
            np.array(pred_labels),
            np.array(confidences))


def compute_attack_metrics(true_labels, clean_preds, attacked_preds, confidences_clean, confidences_attacked):
    """
    Compute standard attack evaluation metrics.

    Returns a dict with:
        clean_accuracy       : baseline recognition accuracy (%)
        attacked_accuracy    : accuracy after attack (%)
        accuracy_drop        : clean - attacked (percentage points)
        attack_success_rate  : fraction of originally-correct samples now misclassified (%)
        avg_conf_clean       : mean confidence before attack
        avg_conf_attacked    : mean confidence after attack
    """
    clean_correct   = (clean_preds   == true_labels)
    attacked_correct = (attacked_preds == true_labels)

    clean_acc    = clean_correct.mean()   * 100
    attacked_acc = attacked_correct.mean() * 100
    acc_drop     = clean_acc - attacked_acc

    # ASR: among images the model got right before, how many flip after attack?
    originally_correct = clean_correct
    if originally_correct.sum() == 0:
        asr = 0.0
    else:
        asr = (~attacked_correct[originally_correct]).mean() * 100

    return {
        'clean_accuracy':      round(float(clean_acc),    2),
        'attacked_accuracy':   round(float(attacked_acc), 2),
        'accuracy_drop':       round(float(acc_drop),     2),
        'attack_success_rate': round(float(asr),          2),
        'avg_conf_clean':      round(float(confidences_clean.mean()),    4),
        'avg_conf_attacked':   round(float(confidences_attacked.mean()), 4),
    }


def print_metrics_table(results: dict, title: str = 'Attack Results'):
    """Pretty-print a {model_name: metrics_dict} table."""
    print(f'\n{"=" * 70}')
    print(f'  {title}')
    print(f'{"=" * 70}')
    header = f"{'Model':<14} {'Clean Acc':>10} {'Atk Acc':>10} {'Drop':>8} {'ASR':>8} {'Delta Conf':>9}"
    print(header)
    print('-' * 70)
    for model_name, m in results.items():
        delta_conf = m['avg_conf_attacked'] - m['avg_conf_clean']
        print(f"{model_name:<14} "
              f"{m['clean_accuracy']:>9.2f}% "
              f"{m['attacked_accuracy']:>9.2f}% "
              f"{m['accuracy_drop']:>7.2f}pp "
              f"{m['attack_success_rate']:>7.2f}% "
              f"{delta_conf:>+9.4f}")
    print('=' * 70)
