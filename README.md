This repository contains the full implementation for the bachelor thesis evaluating the adversarial robustness of three lightweight face recognition models — MobileNetV3, GhostNet, and EfficientNetV2 — against four attack conditions: a physical printed photograph attack, FGSM, PGD, and an adversarial patch.

---

## Requirements

Python 3.10 is required. Install all dependencies with:

```bash
pip install torch==2.10.0 torchvision==0.25.0 timm==1.0.17 scikit-learn==1.7.2 matplotlib==3.10.6 seaborn==0.13.2
```

Training was conducted on an NVIDIA GeForce RTX 5070 with CUDA 13.0. CPU inference is supported but significantly slower.

---

## Dataset

This project uses the **NUAA Photograph Imposter Database** (Tan et al., 2010).

1. Download the dataset from the official source
2. Place it in a `/dataset` folder at the root of the repository with the following structure:

```
dataset/
├── ClientFace/
│   ├── 0001/
│   ├── 0002/
│   └── ...
├── ImposterFace/
│   ├── 0001/
│   ├── 0002/
│   └── ...
├── client_train_face.txt
├── client_test_face.txt
├── imposter_train_face.txt
└── imposter_test_face.txt
```

---

## Trained Models

Place the trained model `.pth` files in the following locations:

```
code/
├── MobileNetV3/
│   └── mobilenet_v3_face_recognition.pth
├── GhostNet/
│   └── ghostnet_face_recognition.pth
└── EfficientNetV2/
    └── efficientnet_v2_face_recognition.pth
```

---

## Usage

### Run the full pipeline

```bash
python main_attack.py
```

### Run specific phases only

```bash
python main_attack.py --only 01 02       # Physical + FGSM only
python main_attack.py --only 06          # Comparative analysis only
```

### Skip specific phases

```bash
python main_attack.py --skip 05          # Skip Grad-CAM (slow)
```

All results are saved to organised subfolders under `results/`.

---

## Scripts

| Script | Description |
|---|---|
| `main_attack.py` | Master pipeline — runs all phases in sequence |
| `utils.py` | Shared utilities: dataset loader, model loaders, metric functions |
| `fgsm.py` | FGSM implementation (Goodfellow et al., 2015) |
| `pgd.py` | PGD implementation (Madry et al., 2018) |
| `adversarial_patch.py` | Adversarial patch implementation (Brown et al., 2017) |
| `gradcam.py` | Grad-CAM implementation (Selvaraju et al., 2017) |
| `01_physical_attack.py` | Physical printed photograph attack evaluation |
| `02_fgsm_attack.py` | FGSM sweep across five epsilon values |
| `03_pgd_attack.py` | PGD sweep across step counts and epsilon values |
| `04_adversarial_patch.py` | Adversarial patch training and ablation |
| `05_xai_gradcam.py` | Grad-CAM heatmap generation |
| `06_comparative_analysis.py` | Cross-attack summary figures and tables |

---

## Results Structure

After running the pipeline, results are saved as follows:

```
results/
├── 01_physical/     # Physical attack results and figures
├── 02_fgsm/         # FGSM sweep results and figures
├── 03_pgd/          # PGD sweep results and figures
├── 04_patch/        # Adversarial patch results and figures
├── 05_gradcam/      # Grad-CAM heatmaps
└── 06_comparative/  # Cross-attack summary figures and tables
```

---

## References

- Tan, X., Li, Y., Liu, J., & Jiang, L. (2010). Face liveness detection from a single image with sparse low rank bilinear discriminative model. *ECCV 2010*.
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. *ICLR 2015*.
- Madry, A., et al. (2018). Towards deep learning models resistant to adversarial attacks. *ICLR 2018*.
- Brown, T. B., et al. (2017). Adversarial patch. *arXiv:1712.09665*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV 2017*.
