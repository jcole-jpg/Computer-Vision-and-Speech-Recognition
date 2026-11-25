"""Diagnose one dermoscopic image with the final model.

    python scripts/predict.py path/to/ISIC_xxxxxxx.jpg

Prints the predicted class, P(malignant) and the 11 probabilities. Uses exactly the validation
transform (milk10k.final.eval_tensor), the model in eval mode and no gradients.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from milk10k.final import predict

if len(sys.argv) != 2:
    sys.exit(__doc__)
print(json.dumps(predict(sys.argv[1]), indent=1))
