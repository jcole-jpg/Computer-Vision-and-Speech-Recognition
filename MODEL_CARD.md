# Model card — MILK10k 11-class lesion classifier (`EADA-CV26-JC`)

**Owner:** Joshua Cole · Computer Vision & Speech Recognition, EADA · November 2026
**Code:** `src/milk10k/final.py` (`FinalModel`, `predict`, `make_submission`)

## What it is
An image classifier that assigns a dermoscopic photo of a skin lesion to one or more of the 11
MILK10k diagnostic classes (AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC) and
reports P(malignant) = P(BCC) + P(MEL) + P(SCCKA) + P(MAL_OTH).

* **Architecture:** average of two probability vectors — (1) ResNet-18, ImageNet weights, every layer
  fine-tuned, head `Linear(512, 11)`; (2) ViT-B/16, ImageNet weights, frozen, class token (768) →
  standardisation → multinomial logistic regression (C = 0.1, balanced class weights). Weight 0.5 each.
* **Decision rule:** class c is predicted when its probability ≥ a per-class threshold tuned on
  validation (MEL 0.25, BCC 0.30, AKIEC 0.35, BEN_OTH 0.35, BKL 0.40, INF 0.40, NV 0.45, SCCKA 0.45,
  VASC 0.45, MAL_OTH 0.50, DF 0.55). A lesion may get 0, 1 or several classes.
* **Input:** one RGB dermoscopic image, any size; resized (stretched) to 224 × 224 with PIL bilinear,
  ImageNet mean/std. No metadata.
* **Weights:** `data/processed/final/resnet18_11class.pt` (45 MB), `data/processed/final/vit_head.joblib`;
  the ViT backbone is torchvision's `ViT_B_16_Weights.IMAGENET1K_V1`.

## Intended use
Decision support for **lesions a clinician already finds suspicious** (the population MILK10k was
collected from), e.g. to prioritise review. Not for self-screening by the public, not a diagnosis,
not for use without a dermatologist.

## Training data
MILK10k (ISIC Archive, CC-BY-NC 4.0, MILK study team): 3,742 training lesions of the seed-42
lesion-level split (`artifacts/splits/train.csv`), dermoscopic image only. Biopsy-enriched: ≈ 69 %
malignant, BCC 48 %, MAL_OTH 9 lesions in the whole dataset. Skin tones 0–1 < 1 %.

## Evaluation (internal test split, 749 lesions never used for any choice)
| Metric | Value |
|---|---|
| Macro-F1, 11 classes (leaderboard metric) | **0.510** (95 % bootstrap CI 0.43–0.56) |
| Macro-F1, argmax instead of thresholds | 0.463 |
| Accuracy (argmax) | 0.685 |
| Malignant vs benign AUC (P(malignant)) | 0.833 |
| Malignant sensitivity / specificity (argmax) | 0.83 / 0.69 |
| Official MILK10k leaderboard (479 lesions) | ____ |

Per-class test F1 (thresholds): BCC 0.84, VASC 0.77, DF 0.71, NV 0.71, SCCKA 0.62, AKIEC 0.50,
MEL 0.45, BEN_OTH 0.44, BKL 0.42, INF 0.14, MAL_OTH 0.00.

## Known limitations and risks
* **MAL_OTH is never predicted** (2 test lesions, 9 in the whole dataset). INF is nearly always missed.
* **Melanoma vs nevus:** 22 of 65 test melanomas were called nevus by the argmax rule — the most
  dangerous error. The lower MEL threshold (0.25) recovers some at the cost of false alarms.
* **Population shift:** trained on biopsied lesions; on unselected lesions precision would drop sharply.
* **Skin tone:** test tones 1 (18 lesions) to 5; tone 0 absent; tone 2 is the weakest group (accuracy
  0.60 vs 0.69 overall, within noise). Performance on dark skin is unknown.
* **One view only:** the clinical close-up and patient history are ignored.
* Confident errors cluster on pink/vascular non-pigmented lesions, featureless melanomas and lesions
  covered by hair or scale (`reports/figures/final_misclassified.png`).

## How to reproduce
See `reports/final_report.md` §8 and README §5c. Seeds: split 42, training 0.
