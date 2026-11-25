"""Train the final 11-class model (ResNet-18 fine-tuned + frozen-ViT head), evaluate it on
val/test through the deployment path and write the leaderboard submission.
Needs data/processed/final/frozen_features.npz (scripts/extract_frozen_features.py).

    python scripts/train_final.py --images data/raw/milk10k/images \
        --test-images data/raw/MILK10k_Test_Input --test-metadata data/raw/MILK10k_Test_Metadata.csv
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from milk10k import final
from milk10k.config import IMAGE_DIR

p = argparse.ArgumentParser()
p.add_argument("--images", default=str(IMAGE_DIR))
p.add_argument("--test-images", required=True)
p.add_argument("--test-metadata", required=True)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--skip-train", action="store_true")
a = p.parse_args()
torch.manual_seed(0)
t0 = time.time()
if not a.skip_train:
    res = final.train(a.images, epochs=a.epochs)          # 1. fine-tune ResNet-18, 11 outputs
    print(json.dumps({k: res[k] for k in ("best_epoch", "val", "test")}, indent=1))
final.train_vit_head()                                    # 2. ViT head on frozen train features
model = final.FinalModel()                                # 3. the average of the two
print(json.dumps(final.evaluate(a.images, model), indent=1))
print(final.tune_thresholds().to_dict())                  # 4. per-class thresholds on VALIDATION
sub = final.make_submission(a.test_images, a.test_metadata, model=model)
print("submission", sub.shape, "->", final.RESULT_DIR / "submission.csv", f"{time.time()-t0:.0f}s")
